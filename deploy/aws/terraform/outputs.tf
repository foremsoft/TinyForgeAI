# TinyForgeAI AWS Terraform Outputs

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnets
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID for the cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_oidc_issuer_url" {
  description = "OIDC issuer URL for the cluster"
  value       = module.eks.cluster_oidc_issuer_url
}

output "ecr_repository_url" {
  description = "ECR repository URL for inference images"
  value       = aws_ecr_repository.inference.repository_url
}

output "model_storage_bucket" {
  description = "S3 bucket name for model storage"
  value       = aws_s3_bucket.models.id
}

output "model_storage_bucket_arn" {
  description = "S3 bucket ARN for model storage"
  value       = aws_s3_bucket.models.arn
}

output "inference_role_arn" {
  description = "IAM role ARN for inference service"
  value       = aws_iam_role.inference_service.arn
}

output "kubeconfig_command" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.aws_region}"
}

output "ecr_login_command" {
  description = "Command to login to ECR"
  value       = "aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.inference.repository_url}"
}

output "push_image_commands" {
  description = "Commands to build and push Docker image"
  value       = <<-EOT
    # Build and tag the image
    docker build -t tinyforge-inference:latest .
    docker tag tinyforge-inference:latest ${aws_ecr_repository.inference.repository_url}:latest

    # Login to ECR
    aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.inference.repository_url}

    # Push the image
    docker push ${aws_ecr_repository.inference.repository_url}:latest
  EOT
}
