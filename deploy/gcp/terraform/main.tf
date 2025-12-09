# TinyForgeAI GCP Infrastructure with Terraform
# This creates a GKE cluster with all necessary components

terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  # Uncomment for remote state storage
  # backend "gcs" {
  #   bucket = "tinyforge-terraform-state"
  #   prefix = "gke/terraform.tfstate"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Data sources
data "google_project" "project" {
  project_id = var.project_id
}

data "google_compute_zones" "available" {
  region = var.region
}

# Locals
locals {
  cluster_name = "${var.environment}-tinyforge-gke"
  zones        = slice(data.google_compute_zones.available.names, 0, 2)

  labels = {
    project     = "tinyforgeai"
    environment = var.environment
    managed-by  = "terraform"
  }
}

# Enable required APIs
resource "google_project_service" "services" {
  for_each = toset([
    "container.googleapis.com",
    "containerregistry.googleapis.com",
    "artifactregistry.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "storage.googleapis.com",
  ])

  project = var.project_id
  service = each.value

  disable_on_destroy = false
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "${var.environment}-tinyforge-vpc"
  auto_create_subnetworks = false
  project                 = var.project_id

  depends_on = [google_project_service.services]
}

# Subnet for GKE
resource "google_compute_subnetwork" "gke" {
  name          = "${var.environment}-tinyforge-gke-subnet"
  ip_cidr_range = var.subnet_cidr
  region        = var.region
  network       = google_compute_network.vpc.id
  project       = var.project_id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_cidr
  }

  private_ip_google_access = true
}

# Cloud NAT for private nodes
resource "google_compute_router" "router" {
  name    = "${var.environment}-tinyforge-router"
  region  = var.region
  network = google_compute_network.vpc.id
  project = var.project_id
}

resource "google_compute_router_nat" "nat" {
  name                               = "${var.environment}-tinyforge-nat"
  router                             = google_compute_router.router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  project                            = var.project_id

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Service Account for GKE nodes
resource "google_service_account" "gke_nodes" {
  account_id   = "${var.environment}-tinyforge-gke-nodes"
  display_name = "TinyForgeAI GKE Node Service Account"
  project      = var.project_id
}

resource "google_project_iam_member" "gke_nodes_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/storage.objectViewer",
    "roles/artifactregistry.reader",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# GKE Cluster
resource "google_container_cluster" "primary" {
  provider = google-beta

  name     = local.cluster_name
  location = var.region
  project  = var.project_id

  # Use regional cluster for HA
  node_locations = local.zones

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  # Network configuration
  network    = google_compute_network.vpc.id
  subnetwork = google_compute_subnetwork.gke.id

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = var.master_cidr
  }

  # Master authorized networks
  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "All networks"
    }
  }

  # Cluster features
  release_channel {
    channel = var.release_channel
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Addons
  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
  }

  # Logging and monitoring
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }

  # Maintenance window
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }

  resource_labels = local.labels

  depends_on = [
    google_project_service.services,
    google_compute_router_nat.nat,
  ]
}

# Primary Node Pool
resource "google_container_node_pool" "primary" {
  name       = "primary"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  project    = var.project_id

  initial_node_count = var.node_desired_size

  autoscaling {
    min_node_count = var.node_min_size
    max_node_count = var.node_max_size
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = var.node_machine_type
    disk_size_gb = 50
    disk_type    = "pd-ssd"

    preemptible = var.use_preemptible

    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    labels = merge(local.labels, {
      role = "general"
    })

    tags = ["gke-node", local.cluster_name]
  }
}

# GPU Node Pool (optional)
resource "google_container_node_pool" "gpu" {
  count = var.enable_gpu_nodes ? 1 : 0

  name       = "gpu"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  project    = var.project_id

  initial_node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = var.gpu_node_max_size
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = "n1-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"

    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    labels = merge(local.labels, {
      role = "gpu"
    })

    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    tags = ["gke-node", local.cluster_name, "gpu"]
  }
}

# Artifact Registry for container images
resource "google_artifact_registry_repository" "inference" {
  location      = var.region
  repository_id = "${var.environment}-tinyforge-inference"
  format        = "DOCKER"
  project       = var.project_id

  labels = local.labels

  depends_on = [google_project_service.services]
}

# GCS Bucket for model storage
resource "google_storage_bucket" "models" {
  name          = "${var.environment}-tinyforge-models-${var.project_id}"
  location      = var.region
  project       = var.project_id
  force_destroy = var.environment != "production"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 3
    }
    action {
      type = "Delete"
    }
  }

  labels = local.labels
}

# Service Account for TinyForgeAI workload
resource "google_service_account" "tinyforge" {
  account_id   = "${var.environment}-tinyforge-inference"
  display_name = "TinyForgeAI Inference Service Account"
  project      = var.project_id
}

# Grant model storage access
resource "google_storage_bucket_iam_member" "tinyforge_models" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.tinyforge.email}"
}

# Workload Identity binding
resource "google_service_account_iam_member" "tinyforge_workload_identity" {
  service_account_id = google_service_account.tinyforge.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[tinyforge/tinyforge]"
}

# Kubernetes provider configuration
data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.primary.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
  }
}

# Namespace for TinyForgeAI
resource "kubernetes_namespace" "tinyforge" {
  depends_on = [google_container_node_pool.primary]

  metadata {
    name = "tinyforge"

    labels = {
      name = "tinyforge"
    }
  }
}

# Deploy TinyForgeAI using Helm
resource "helm_release" "tinyforge" {
  depends_on = [kubernetes_namespace.tinyforge]

  name       = "tinyforge"
  namespace  = "tinyforge"
  chart      = "${path.module}/../../helm-chart"

  values = [
    yamlencode({
      replicaCount = var.inference_replicas

      image = {
        repository = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.inference.name}/inference"
        tag        = var.image_tag
        pullPolicy = "Always"
      }

      resources = {
        limits = {
          cpu    = var.inference_cpu_limit
          memory = var.inference_memory_limit
        }
        requests = {
          cpu    = var.inference_cpu_request
          memory = var.inference_memory_request
        }
      }

      autoscaling = {
        enabled                        = true
        minReplicas                    = var.inference_min_replicas
        maxReplicas                    = var.inference_max_replicas
        targetCPUUtilizationPercentage = 70
      }

      persistence = {
        enabled      = true
        storageClass = "premium-rwo"
        size         = var.model_storage_size
      }

      env = {
        GCP_PROJECT        = var.project_id
        MODEL_STORAGE_BUCKET = google_storage_bucket.models.name
        INFERENCE_PORT     = "8000"
        LOG_LEVEL          = var.environment == "production" ? "INFO" : "DEBUG"
      }

      serviceAccount = {
        create = true
        annotations = {
          "iam.gke.io/gcp-service-account" = google_service_account.tinyforge.email
        }
      }
    })
  ]
}
