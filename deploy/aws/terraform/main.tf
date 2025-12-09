# TinyForgeAI AWS Infrastructure with Terraform
# This creates an EKS cluster with all necessary components

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
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
  # backend "s3" {
  #   bucket         = "tinyforge-terraform-state"
  #   key            = "eks/terraform.tfstate"
  #   region         = "us-west-2"
  #   dynamodb_table = "terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "TinyForgeAI"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Locals
locals {
  cluster_name = "${var.environment}-tinyforge-eks"
  azs          = slice(data.aws_availability_zones.available.names, 0, 2)

  tags = {
    Project     = "TinyForgeAI"
    Environment = var.environment
  }
}

# VPC Module
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.environment}-tinyforge-vpc"
  cidr = var.vpc_cidr

  azs             = local.azs
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "production"
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Tags required for EKS
  public_subnet_tags = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                      = 1
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"             = 1
  }

  tags = local.tags
}

# EKS Module
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = local.cluster_name
  cluster_version = var.eks_cluster_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  # Cluster add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # Node groups
  eks_managed_node_groups = {
    # General purpose nodes
    general = {
      name            = "general"
      instance_types  = var.node_instance_types
      capacity_type   = var.use_spot_instances ? "SPOT" : "ON_DEMAND"
      min_size        = var.node_min_size
      max_size        = var.node_max_size
      desired_size    = var.node_desired_size

      labels = {
        role = "general"
      }

      tags = {
        "k8s.io/cluster-autoscaler/enabled"             = "true"
        "k8s.io/cluster-autoscaler/${local.cluster_name}" = "owned"
      }
    }

    # GPU nodes (optional)
    gpu = {
      name            = "gpu"
      instance_types  = ["g4dn.xlarge", "g4dn.2xlarge"]
      capacity_type   = "ON_DEMAND"
      min_size        = 0
      max_size        = var.gpu_node_max_size
      desired_size    = 0
      ami_type        = "AL2_x86_64_GPU"

      labels = {
        role                              = "gpu"
        "nvidia.com/gpu.present"          = "true"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]

      tags = {
        "k8s.io/cluster-autoscaler/enabled"             = "true"
        "k8s.io/cluster-autoscaler/${local.cluster_name}" = "owned"
      }
    }
  }

  # Access configuration
  manage_aws_auth_configmap = true

  aws_auth_roles = var.additional_aws_auth_roles

  tags = local.tags
}

# ECR Repository
resource "aws_ecr_repository" "inference" {
  name                 = "${var.environment}-tinyforge-inference"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = local.tags
}

resource "aws_ecr_lifecycle_policy" "inference" {
  repository = aws_ecr_repository.inference.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 20 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 20
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# S3 Bucket for Model Storage
resource "aws_s3_bucket" "models" {
  bucket = "${var.environment}-tinyforge-models-${data.aws_caller_identity.current.account_id}"

  tags = local.tags
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM Policy for S3 Access
resource "aws_iam_policy" "s3_model_access" {
  name        = "${var.environment}-tinyforge-s3-model-access"
  description = "Policy for TinyForgeAI to access model storage"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*"
        ]
      }
    ]
  })
}

# Kubernetes provider configuration
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", local.cluster_name]
  }
}

# Helm provider configuration
provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", local.cluster_name]
    }
  }
}

# Namespace for TinyForgeAI
resource "kubernetes_namespace" "tinyforge" {
  depends_on = [module.eks]

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
        repository = aws_ecr_repository.inference.repository_url
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
        storageClass = "gp3"
        size         = var.model_storage_size
      }

      env = {
        AWS_REGION              = var.aws_region
        MODEL_STORAGE_BUCKET    = aws_s3_bucket.models.id
        INFERENCE_PORT          = "8000"
        LOG_LEVEL               = var.environment == "production" ? "INFO" : "DEBUG"
      }

      serviceAccount = {
        create = true
        annotations = {
          "eks.amazonaws.com/role-arn" = aws_iam_role.inference_service.arn
        }
      }
    })
  ]
}

# IAM Role for Service Account (IRSA)
resource "aws_iam_role" "inference_service" {
  name = "${var.environment}-tinyforge-inference-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${module.eks.oidc_provider}:sub" = "system:serviceaccount:tinyforge:tinyforge"
            "${module.eks.oidc_provider}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "inference_s3" {
  role       = aws_iam_role.inference_service.name
  policy_arn = aws_iam_policy.s3_model_access.arn
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "eks" {
  name              = "/aws/eks/${local.cluster_name}/cluster"
  retention_in_days = var.log_retention_days

  tags = local.tags
}
