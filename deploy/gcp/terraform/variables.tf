# TinyForgeAI GCP Terraform Variables

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for deployment"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

# Network Configuration
variable "subnet_cidr" {
  description = "CIDR block for the subnet"
  type        = string
  default     = "10.0.0.0/24"
}

variable "pods_cidr" {
  description = "CIDR block for GKE pods"
  type        = string
  default     = "10.1.0.0/16"
}

variable "services_cidr" {
  description = "CIDR block for GKE services"
  type        = string
  default     = "10.2.0.0/20"
}

variable "master_cidr" {
  description = "CIDR block for GKE master"
  type        = string
  default     = "172.16.0.0/28"
}

# GKE Configuration
variable "release_channel" {
  description = "GKE release channel (RAPID, REGULAR, STABLE)"
  type        = string
  default     = "REGULAR"

  validation {
    condition     = contains(["RAPID", "REGULAR", "STABLE"], var.release_channel)
    error_message = "Release channel must be one of: RAPID, REGULAR, STABLE."
  }
}

variable "node_machine_type" {
  description = "Machine type for worker nodes"
  type        = string
  default     = "e2-medium"
}

variable "node_min_size" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 5
}

variable "node_desired_size" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 2
}

variable "use_preemptible" {
  description = "Use preemptible VMs for cost savings"
  type        = bool
  default     = false
}

variable "enable_gpu_nodes" {
  description = "Enable GPU node pool"
  type        = bool
  default     = false
}

variable "gpu_node_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 2
}

# Inference Service Configuration
variable "inference_replicas" {
  description = "Number of inference service replicas"
  type        = number
  default     = 2
}

variable "inference_min_replicas" {
  description = "Minimum replicas for autoscaling"
  type        = number
  default     = 1
}

variable "inference_max_replicas" {
  description = "Maximum replicas for autoscaling"
  type        = number
  default     = 10
}

variable "inference_cpu_request" {
  description = "CPU request for inference pods"
  type        = string
  default     = "100m"
}

variable "inference_cpu_limit" {
  description = "CPU limit for inference pods"
  type        = string
  default     = "500m"
}

variable "inference_memory_request" {
  description = "Memory request for inference pods"
  type        = string
  default     = "256Mi"
}

variable "inference_memory_limit" {
  description = "Memory limit for inference pods"
  type        = string
  default     = "512Mi"
}

variable "model_storage_size" {
  description = "Size of persistent storage for models"
  type        = string
  default     = "10Gi"
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}
