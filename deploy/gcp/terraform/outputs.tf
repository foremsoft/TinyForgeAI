# TinyForgeAI GCP Terraform Outputs

output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP region"
  value       = var.region
}

output "vpc_name" {
  description = "VPC network name"
  value       = google_compute_network.vpc.name
}

output "subnet_name" {
  description = "Subnet name"
  value       = google_compute_subnetwork.gke.name
}

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "GKE cluster API endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.primary.location
}

output "artifact_registry_url" {
  description = "Artifact Registry URL for container images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.inference.name}"
}

output "model_storage_bucket" {
  description = "GCS bucket name for model storage"
  value       = google_storage_bucket.models.name
}

output "workload_identity_email" {
  description = "Service account email for workload identity"
  value       = google_service_account.tinyforge.email
}

output "kubeconfig_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${var.region} --project ${var.project_id}"
}

output "docker_auth_command" {
  description = "Command to configure Docker authentication"
  value       = "gcloud auth configure-docker ${var.region}-docker.pkg.dev"
}

output "push_image_commands" {
  description = "Commands to build and push Docker image"
  value       = <<-EOT
    # Configure Docker authentication
    gcloud auth configure-docker ${var.region}-docker.pkg.dev

    # Build and tag the image
    docker build -t tinyforge-inference:latest .
    docker tag tinyforge-inference:latest ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.inference.name}/inference:latest

    # Push the image
    docker push ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.inference.name}/inference:latest
  EOT
}
