# TinyForgeAI AWS Deployment

Deploy TinyForgeAI to Amazon Web Services using CloudFormation or Terraform.

## Prerequisites

- AWS CLI configured with appropriate credentials
- AWS account with permissions to create EKS, VPC, ECR, S3, and IAM resources
- Docker installed locally
- kubectl installed
- Terraform >= 1.0 (for Terraform deployment)

## Deployment Options

### Option 1: CloudFormation (Quick Start)

Deploy using the CloudFormation template:

```bash
# Deploy the stack
aws cloudformation create-stack \
  --stack-name tinyforge-eks \
  --template-body file://cloudformation-eks.yaml \
  --parameters \
    ParameterKey=EnvironmentName,ParameterValue=tinyforge \
    ParameterKey=NodeInstanceType,ParameterValue=t3.medium \
    ParameterKey=NodeGroupDesiredSize,ParameterValue=2 \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for stack creation
aws cloudformation wait stack-create-complete --stack-name tinyforge-eks

# Get outputs
aws cloudformation describe-stacks --stack-name tinyforge-eks --query 'Stacks[0].Outputs'
```

### Option 2: Terraform (Production)

For production deployments with more flexibility:

```bash
cd terraform

# Initialize Terraform
terraform init

# Copy and customize variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings

# Preview changes
terraform plan

# Apply infrastructure
terraform apply

# Get outputs
terraform output
```

## Post-Deployment Steps

### 1. Configure kubectl

```bash
# Using CloudFormation output
aws eks update-kubeconfig --name tinyforge-cluster --region us-west-2

# Or using Terraform output
$(terraform output -raw kubeconfig_command)
```

### 2. Build and Push Docker Image

```bash
# Login to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <ECR_REPOSITORY_URI>

# Build and push
docker build -t tinyforge-inference:latest .
docker tag tinyforge-inference:latest <ECR_REPOSITORY_URI>:latest
docker push <ECR_REPOSITORY_URI>:latest
```

### 3. Deploy TinyForgeAI

If using CloudFormation (without Helm integration):

```bash
# Install using Helm
helm install tinyforge ../helm-chart/ \
  -n tinyforge \
  --create-namespace \
  --set image.repository=<ECR_REPOSITORY_URI> \
  --set image.tag=latest
```

Terraform automatically deploys TinyForgeAI using Helm.

### 4. Verify Deployment

```bash
# Check pods
kubectl get pods -n tinyforge

# Check services
kubectl get svc -n tinyforge

# Port forward to test
kubectl port-forward svc/tinyforge-inference-svc 8000:80 -n tinyforge

# Test health endpoint
curl http://localhost:8000/health
```

## Resource Summary

### CloudFormation Creates

| Resource | Description |
|----------|-------------|
| VPC | Virtual Private Cloud with public/private subnets |
| EKS Cluster | Managed Kubernetes cluster |
| Node Group | EC2 worker nodes |
| ECR Repository | Container registry for inference image |
| S3 Bucket | Model storage |
| IAM Roles | Service roles for EKS and nodes |
| NAT Gateway | Outbound internet for private subnets |

### Terraform Creates (Additional)

| Resource | Description |
|----------|-------------|
| GPU Node Group | Optional GPU nodes for inference |
| IRSA | IAM Roles for Service Accounts |
| Helm Release | Automated TinyForgeAI deployment |
| CloudWatch Logs | Cluster logging |

## Cost Considerations

### Estimated Monthly Costs (us-west-2)

| Component | Dev/Test | Production |
|-----------|----------|------------|
| EKS Control Plane | $73 | $73 |
| NAT Gateway | $32 | $64 (2 AZs) |
| EC2 (2x t3.medium) | $60 | $120 (4x) |
| EBS Storage | $10 | $30 |
| S3 Storage | $1 | $5 |
| **Total** | **~$176** | **~$292** |

### Cost Optimization Tips

1. **Use Spot Instances**: Set `use_spot_instances = true` for 60-90% savings on compute
2. **Single NAT Gateway**: Use in non-production for 50% NAT savings
3. **Reserved Instances**: For production, consider 1-year reservations
4. **Cluster Autoscaler**: Scale down nodes during low traffic
5. **GPU on Demand**: Keep GPU node group at 0 and scale up only when needed

## Security Best Practices

1. **Network Isolation**: Worker nodes are in private subnets
2. **IRSA**: Use IAM Roles for Service Accounts instead of node-level permissions
3. **ECR Scanning**: Image scanning enabled on push
4. **Encryption**: S3 and EBS encrypted at rest
5. **Secrets Management**: Use AWS Secrets Manager or Parameter Store for secrets

## Cleanup

### CloudFormation

```bash
aws cloudformation delete-stack --stack-name tinyforge-eks
aws cloudformation wait stack-delete-complete --stack-name tinyforge-eks
```

### Terraform

```bash
cd terraform
terraform destroy
```

## Troubleshooting

### EKS Cluster Not Accessible

```bash
# Check cluster status
aws eks describe-cluster --name tinyforge-cluster --query 'cluster.status'

# Verify IAM permissions
aws sts get-caller-identity
```

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod -l app=tinyforge -n tinyforge

# Check node status
kubectl get nodes

# Check node group events
aws eks describe-nodegroup --cluster-name tinyforge-cluster --nodegroup-name tinyforge-nodegroup
```

### ECR Push Failed

```bash
# Refresh ECR login (expires after 12 hours)
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <ECR_URI>

# Check repository exists
aws ecr describe-repositories --repository-names tinyforge-inference
```

## See Also

- [Main Deployment Guide](../README.md)
- [GCP Deployment](../gcp/README.md)
- [Azure Deployment](../azure/README.md)
- [Helm Chart Documentation](../helm-chart/README.md)
