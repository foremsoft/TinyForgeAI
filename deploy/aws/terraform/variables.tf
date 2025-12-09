# TinyForgeAI AWS Terraform Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
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

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24"]
}

# EKS Configuration
variable "eks_cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "node_instance_types" {
  description = "EC2 instance types for worker nodes"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
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

variable "gpu_node_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 2
}

variable "use_spot_instances" {
  description = "Use spot instances for cost savings"
  type        = bool
  default     = false
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
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# Auth
variable "additional_aws_auth_roles" {
  description = "Additional IAM roles to add to aws-auth configmap"
  type = list(object({
    rolearn  = string
    username = string
    groups   = list(string)
  }))
  default = []
}
