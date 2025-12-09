# TinyForgeAI Azure Infrastructure with Terraform
# This creates an AKS cluster with all necessary components

terraform {
  required_version = ">= 1.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 2.45"
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
  # backend "azurerm" {
  #   resource_group_name  = "terraform-state-rg"
  #   storage_account_name = "tinyforgetfstate"
  #   container_name       = "tfstate"
  #   key                  = "aks/terraform.tfstate"
  # }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# Data sources
data "azurerm_client_config" "current" {}

data "azuread_client_config" "current" {}

# Locals
locals {
  cluster_name = "${var.environment}-tinyforge-aks"

  tags = {
    Project     = "TinyForgeAI"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${var.environment}-tinyforge-rg"
  location = var.location

  tags = local.tags
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "${var.environment}-tinyforge-vnet"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = [var.vnet_cidr]

  tags = local.tags
}

# Subnet for AKS
resource "azurerm_subnet" "aks" {
  name                 = "aks-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.aks_subnet_cidr]
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.environment}-tinyforge-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = var.log_retention_days

  tags = local.tags
}

# Azure Container Registry
resource "azurerm_container_registry" "main" {
  name                = "${var.environment}tinyforgeacr"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = var.acr_sku
  admin_enabled       = true

  tags = local.tags
}

# User Assigned Identity for AKS
resource "azurerm_user_assigned_identity" "aks" {
  name                = "${var.environment}-tinyforge-aks-identity"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  tags = local.tags
}

# Role assignment for ACR pull
resource "azurerm_role_assignment" "aks_acr" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = local.cluster_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "${var.environment}-tinyforge"
  kubernetes_version  = var.kubernetes_version

  default_node_pool {
    name                = "default"
    node_count          = var.node_desired_size
    vm_size             = var.node_vm_size
    vnet_subnet_id      = azurerm_subnet.aks.id
    enable_auto_scaling = true
    min_count           = var.node_min_size
    max_count           = var.node_max_size
    os_disk_size_gb     = 50
    os_disk_type        = "Managed"

    node_labels = {
      role = "general"
    }

    tags = local.tags
  }

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.aks.id]
  }

  network_profile {
    network_plugin    = "azure"
    network_policy    = "calico"
    load_balancer_sku = "standard"
    service_cidr      = var.service_cidr
    dns_service_ip    = var.dns_service_ip
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  }

  azure_active_directory_role_based_access_control {
    managed                = true
    azure_rbac_enabled     = true
    admin_group_object_ids = var.admin_group_object_ids
  }

  auto_scaler_profile {
    balance_similar_node_groups = true
    scale_down_delay_after_add  = "10m"
    scale_down_unneeded         = "10m"
  }

  tags = local.tags
}

# GPU Node Pool (optional)
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  count = var.enable_gpu_nodes ? 1 : 0

  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = "Standard_NC6s_v3"  # NVIDIA V100
  node_count            = 0
  enable_auto_scaling   = true
  min_count             = 0
  max_count             = var.gpu_node_max_size
  os_disk_size_gb       = 100
  vnet_subnet_id        = azurerm_subnet.aks.id

  node_labels = {
    role                     = "gpu"
    "nvidia.com/gpu.present" = "true"
  }

  node_taints = [
    "nvidia.com/gpu=true:NoSchedule"
  ]

  tags = local.tags
}

# Storage Account for model storage
resource "azurerm_storage_account" "models" {
  name                     = "${var.environment}tinyforgemodels"
  location                 = azurerm_resource_group.main.location
  resource_group_name      = azurerm_resource_group.main.name
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"

  blob_properties {
    versioning_enabled = true
  }

  tags = local.tags
}

resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.models.name
  container_access_type = "private"
}

# User Assigned Identity for TinyForgeAI workload
resource "azurerm_user_assigned_identity" "tinyforge" {
  name                = "${var.environment}-tinyforge-workload-identity"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  tags = local.tags
}

# Role assignment for storage access
resource "azurerm_role_assignment" "tinyforge_storage" {
  scope                = azurerm_storage_account.models.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.tinyforge.principal_id
}

# Federated credential for Workload Identity
resource "azurerm_federated_identity_credential" "tinyforge" {
  name                = "tinyforge-federated"
  resource_group_name = azurerm_resource_group.main.name
  parent_id           = azurerm_user_assigned_identity.tinyforge.id
  audience            = ["api://AzureADTokenExchange"]
  issuer              = azurerm_kubernetes_cluster.main.oidc_issuer_url
  subject             = "system:serviceaccount:tinyforge:tinyforge"
}

# Kubernetes provider configuration
provider "kubernetes" {
  host                   = azurerm_kubernetes_cluster.main.kube_config[0].host
  client_certificate     = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].client_certificate)
  client_key             = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].client_key)
  cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].cluster_ca_certificate)
}

# Helm provider configuration
provider "helm" {
  kubernetes {
    host                   = azurerm_kubernetes_cluster.main.kube_config[0].host
    client_certificate     = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].client_certificate)
    client_key             = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].client_key)
    cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].cluster_ca_certificate)
  }
}

# Namespace for TinyForgeAI
resource "kubernetes_namespace" "tinyforge" {
  depends_on = [azurerm_kubernetes_cluster.main]

  metadata {
    name = "tinyforge"

    labels = {
      name = "tinyforge"
    }
  }
}

# Deploy TinyForgeAI using Helm
resource "helm_release" "tinyforge" {
  depends_on = [
    kubernetes_namespace.tinyforge,
    azurerm_role_assignment.aks_acr,
  ]

  name       = "tinyforge"
  namespace  = "tinyforge"
  chart      = "${path.module}/../../helm-chart"

  values = [
    yamlencode({
      replicaCount = var.inference_replicas

      image = {
        repository = "${azurerm_container_registry.main.login_server}/tinyforge-inference"
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
        storageClass = "managed-premium"
        size         = var.model_storage_size
      }

      env = {
        AZURE_STORAGE_ACCOUNT = azurerm_storage_account.models.name
        AZURE_CONTAINER_NAME  = azurerm_storage_container.models.name
        INFERENCE_PORT        = "8000"
        LOG_LEVEL             = var.environment == "production" ? "INFO" : "DEBUG"
      }

      serviceAccount = {
        create = true
        annotations = {
          "azure.workload.identity/client-id" = azurerm_user_assigned_identity.tinyforge.client_id
        }
      }

      podLabels = {
        "azure.workload.identity/use" = "true"
      }
    })
  ]
}
