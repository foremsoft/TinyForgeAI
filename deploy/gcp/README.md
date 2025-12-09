# TinyForgeAI GCP Deployment

Deploy TinyForgeAI to Google Cloud Platform using Terraform with GKE.

## Prerequisites

- Google Cloud SDK (`gcloud`) installed and configured
- GCP project with billing enabled
- Terraform >= 1.0
- Docker installed locally
- kubectl installed

## Quick Start

### 1. Set Up GCP Authentication

```bash
# Login to GCP
gcloud auth login

# Set default project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable container.googleapis.com artifactregistry.googleapis.com
```

### 2. Deploy Infrastructure

```bash
cd terraform

# Initialize Terraform
terraform init

# Copy and customize variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project ID and settings

# Preview changes
terraform plan

# Apply infrastructure
terraform apply
```

### 3. Build and Push Docker Image

```bash
# Configure Docker authentication
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build the image
docker build -t tinyforge-inference:latest .

# Tag and push
docker tag tinyforge-inference:latest \
  us-central1-docker.pkg.dev/YOUR_PROJECT/dev-tinyforge-inference/inference:latest
docker push us-central1-docker.pkg.dev/YOUR_PROJECT/dev-tinyforge-inference/inference:latest
```

### 4. Verify Deployment

```bash
# Configure kubectl
gcloud container clusters get-credentials dev-tinyforge-gke --region us-central1

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
| VPC Network | Custom VPC with subnets for GKE |
| Cloud NAT | Outbound internet for private nodes |
| GKE Cluster | Regional Autopilot-style cluster |
| Node Pool | Auto-scaling node pool |
| GPU Pool | Optional T4 GPU nodes |
| Artifact Registry | Container registry |
| Cloud Storage | Model storage bucket |
| Service Accounts | Workload identity enabled |

## Cost Considerations

### Estimated Monthly Costs (us-central1)

| Component | Dev/Test | Production |
|-----------|----------|------------|
| GKE Management | $73 | $73 |
| Cloud NAT | $32 | $64 |
| Compute (2x e2-medium) | $50 | $100 |
| Persistent Disk | $10 | $30 |
| Cloud Storage | $1 | $5 |
| **Total** | **~$166** | **~$272** |

### Cost Optimization

1. **Preemptible VMs**: Set `use_preemptible = true` for 60-80% savings
2. **Committed Use Discounts**: 1-3 year commitments for production
3. **E2 Machine Types**: Cost-effective for most workloads
4. **Autopilot**: Consider GKE Autopilot for automatic optimization

## Security Features

- **Private Cluster**: Worker nodes have no public IPs
- **Workload Identity**: Pod-level IAM without service account keys
- **VPC-native**: IP aliases for better network security
- **Artifact Registry**: Vulnerability scanning for images
- **Encrypted Storage**: GCS and PD encrypted by default

## Cleanup

```bash
cd terraform
terraform destroy
```

## See Also

- [Main Deployment Guide](../README.md)
- [AWS Deployment](../aws/README.md)
- [Azure Deployment](../azure/README.md)
