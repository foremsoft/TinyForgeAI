# TinyForgeAI Azure Terraform Variables

variable "location" {
  description = "Azure region for deployment"
  type        = string
  default     = "eastus"
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
variable "vnet_cidr" {
  description = "CIDR block for the VNet"
  type        = string
  default     = "10.0.0.0/16"
}

variable "aks_subnet_cidr" {
  description = "CIDR block for the AKS subnet"
  type        = string
  default     = "10.0.0.0/22"
}

variable "service_cidr" {
  description = "CIDR block for Kubernetes services"
  type        = string
  default     = "10.1.0.0/16"
}

variable "dns_service_ip" {
  description = "IP address for Kubernetes DNS service"
  type        = string
  default     = "10.1.0.10"
}

# AKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for AKS"
  type        = string
  default     = "1.28"
}

variable "node_vm_size" {
  description = "VM size for worker nodes"
  type        = string
  default     = "Standard_D2s_v3"
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

# ACR Configuration
variable "acr_sku" {
  description = "SKU for Azure Container Registry"
  type        = string
  default     = "Basic"

  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.acr_sku)
    error_message = "ACR SKU must be one of: Basic, Standard, Premium."
  }
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

# Logging
variable "log_retention_days" {
  description = "Log Analytics retention in days"
  type        = number
  default     = 30
}

# Auth
variable "admin_group_object_ids" {
  description = "Azure AD group object IDs for cluster admin access"
  type        = list(string)
  default     = []
}
