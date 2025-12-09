# TinyForgeAI Azure Deployment

Deploy TinyForgeAI to Microsoft Azure using Terraform with AKS.

## Prerequisites

- Azure CLI (`az`) installed and configured
- Azure subscription with sufficient quota
- Terraform >= 1.0
- Docker installed locally
- kubectl installed

## Quick Start

### 1. Set Up Azure Authentication

```bash
# Login to Azure
az login

# Set default subscription
az account set --subscription "Your Subscription Name"

# Verify
az account show
```

### 2. Deploy Infrastructure

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
```

### 3. Build and Push Docker Image

```bash
# Login to ACR
az acr login --name devtinyforgeacr

# Build the image
docker build -t tinyforge-inference:latest .

# Tag and push
docker tag tinyforge-inference:latest devtinyforgeacr.azurecr.io/tinyforge-inference:latest
docker push devtinyforgeacr.azurecr.io/tinyforge-inference:latest
```

### 4. Verify Deployment

```bash
# Configure kubectl
az aks get-credentials --resource-group dev-tinyforge-rg --name dev-tinyforge-aks

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

| Resource | Description |
|----------|-------------|
| Resource Group | Container for all resources |
| VNet | Virtual network with AKS subnet |
| AKS Cluster | Managed Kubernetes cluster |
| Node Pool | Auto-scaling VM node pool |
| GPU Pool | Optional GPU nodes |
| ACR | Azure Container Registry |
| Storage Account | Model storage with blob container |
| Log Analytics | Cluster monitoring and logs |
| Managed Identities | Workload identity for pods |

## Cost Considerations

### Estimated Monthly Costs (East US)

| Component | Dev/Test | Production |
|-----------|----------|------------|
| AKS (no control plane cost) | $0 | $0 |
| VMs (2x D2s_v3) | $140 | $280 |
| ACR (Basic) | $5 | $20 (Standard) |
| Storage Account | $5 | $15 |
| Log Analytics | $10 | $30 |
| **Total** | **~$160** | **~$345** |

### Cost Optimization

1. **Spot VMs**: Use spot instances for 60-90% savings
2. **Reserved Instances**: 1-3 year reservations for production
3. **Dev/Test Pricing**: Use Azure Dev/Test subscription
4. **Autoscaling**: Scale to zero during off-hours

## Security Features

- **Azure AD Integration**: RBAC with Azure AD
- **Managed Identities**: No credentials in pods
- **Network Policies**: Calico for pod-level security
- **Private Endpoints**: Optional private ACR/Storage access
- **Encryption**: Storage and disk encryption by default

## Cleanup

```bash
cd terraform
terraform destroy
```

## See Also

- [Main Deployment Guide](../README.md)
- [AWS Deployment](../aws/README.md)
- [GCP Deployment](../gcp/README.md)
