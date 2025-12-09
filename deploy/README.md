# TinyForgeAI Deployment Guide

This directory contains deployment configurations for TinyForgeAI inference services.

## Directory Structure

```
deploy/
├── k8s/                    # Raw Kubernetes manifests
│   ├── deployment.yaml     # Deployment configuration
│   ├── service.yaml        # Service (ClusterIP + NodePort)
│   ├── hpa.yaml           # Horizontal Pod Autoscaler
│   ├── pvc.yaml           # Persistent Volume Claim
│   └── configmap.yaml     # Configuration
├── helm-chart/            # Helm chart for production
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
├── aws/                   # AWS deployment (CloudFormation + Terraform)
│   ├── cloudformation-eks.yaml
│   ├── terraform/
│   └── README.md
├── gcp/                   # GCP deployment (Terraform)
│   ├── terraform/
│   └── README.md
└── azure/                 # Azure deployment (Terraform)
    ├── terraform/
    └── README.md
```

## Quick Start

### Prerequisites

- Docker installed and running
- Kubernetes cluster (minikube, kind, or cloud provider)
- kubectl configured
- Helm 3.x (for Helm deployments)

### Build Docker Image

```bash
# From project root
docker build -f docker/Dockerfile.inference -t tinyforge-inference:latest .
```

### Option 1: Raw Kubernetes Manifests

```bash
# Create namespace (optional)
kubectl create namespace tinyforge

# Apply all manifests
kubectl apply -f deploy/k8s/ -n tinyforge

# Check deployment status
kubectl get pods -n tinyforge
kubectl get svc -n tinyforge
```

### Option 2: Helm Chart

```bash
# Install with default values
helm install tinyforge deploy/helm-chart/ -n tinyforge --create-namespace

# Install with custom values
helm install tinyforge deploy/helm-chart/ \
  -n tinyforge \
  --create-namespace \
  --set replicaCount=3 \
  --set autoscaling.enabled=true

# Upgrade existing release
helm upgrade tinyforge deploy/helm-chart/ -n tinyforge

# Uninstall
helm uninstall tinyforge -n tinyforge
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_REGISTRY_PATH` | Path to model files | `/models` |
| `INFERENCE_PORT` | Port for inference server | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Helm Values

Key configuration options in `values.yaml`:

```yaml
# Replica count
replicaCount: 1

# Image settings
image:
  repository: tinyforge-inference
  tag: latest

# Resource limits
resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 100m
    memory: 256Mi

# Autoscaling
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

# Persistence
persistence:
  enabled: true
  size: 1Gi
```

## Accessing the Service

### Port Forward (Development)

```bash
kubectl port-forward svc/tinyforge-inference-svc 8000:80 -n tinyforge
curl http://localhost:8000/health
```

### NodePort (Testing)

```bash
# Get node IP and port
NODE_PORT=$(kubectl get svc tinyforge-inference-nodeport -n tinyforge -o jsonpath='{.spec.ports[0].nodePort}')
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}')
curl http://$NODE_IP:$NODE_PORT/health
```

### Ingress (Production)

Enable ingress in Helm values:

```yaml
ingress:
  enabled: true
  className: nginx
  hosts:
    - host: tinyforge.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: tinyforge-tls
      hosts:
        - tinyforge.example.com
```

## Health Checks

The inference server provides health endpoints:

- `GET /health` - Liveness probe (server is running)
- `GET /readyz` - Readiness probe (server can accept requests)
- `GET /metrics` - Basic metrics (request count)

## Scaling

### Manual Scaling

```bash
kubectl scale deployment tinyforge-inference --replicas=3 -n tinyforge
```

### Auto Scaling

HPA is configured to scale based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)

```bash
# Check HPA status
kubectl get hpa -n tinyforge

# Describe HPA for details
kubectl describe hpa tinyforge-hpa -n tinyforge
```

## Troubleshooting

### Check Logs

```bash
kubectl logs -l app=tinyforge -n tinyforge --tail=100
```

### Describe Pod

```bash
kubectl describe pod -l app=tinyforge -n tinyforge
```

### Shell into Pod

```bash
kubectl exec -it $(kubectl get pod -l app=tinyforge -n tinyforge -o jsonpath='{.items[0].metadata.name}') -n tinyforge -- /bin/sh
```

### Check Events

```bash
kubectl get events -n tinyforge --sort-by='.lastTimestamp'
```

## Production Considerations

1. **Image Registry**: Push images to a container registry (Docker Hub, ECR, GCR, etc.)
2. **Secrets**: Use Kubernetes secrets for sensitive configuration
3. **Resource Limits**: Adjust based on actual workload
4. **Persistence**: Use cloud-native storage classes
5. **Monitoring**: Add Prometheus/Grafana for metrics
6. **Logging**: Configure centralized logging (ELK, Loki)
7. **Network Policies**: Restrict pod communication
8. **RBAC**: Configure appropriate service accounts

## Cloud Deployment Guides

Full infrastructure-as-code templates are available for each major cloud provider:

| Cloud Provider | IaC Tools | Documentation |
|----------------|-----------|---------------|
| **AWS** | CloudFormation, Terraform | [AWS Guide](./aws/README.md) |
| **GCP** | Terraform | [GCP Guide](./gcp/README.md) |
| **Azure** | Terraform | [Azure Guide](./azure/README.md) |

### What's Included

Each cloud deployment includes:
- **Managed Kubernetes**: EKS (AWS), GKE (GCP), AKS (Azure)
- **Container Registry**: ECR, Artifact Registry, ACR
- **Object Storage**: S3, GCS, Azure Blob
- **Networking**: VPC/VNet with private subnets
- **IAM/Workload Identity**: Pod-level permissions
- **Autoscaling**: HPA + Cluster Autoscaler
- **Monitoring**: Native cloud logging and metrics
- **GPU Support**: Optional GPU node pools

### Quick Start by Cloud

#### AWS EKS

```bash
cd deploy/aws/terraform
terraform init
terraform apply

# Or use CloudFormation
aws cloudformation create-stack \
  --stack-name tinyforge-eks \
  --template-body file://cloudformation-eks.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

#### GCP GKE

```bash
cd deploy/gcp/terraform
terraform init
terraform apply
```

#### Azure AKS

```bash
cd deploy/azure/terraform
terraform init
terraform apply
```

### Cost Comparison

| Cloud | Dev/Test | Production |
|-------|----------|------------|
| AWS | ~$176/mo | ~$292/mo |
| GCP | ~$166/mo | ~$272/mo |
| Azure | ~$160/mo | ~$345/mo |

*Estimates based on 2 medium nodes, single NAT, basic storage*

## See Also

- [Docker Guide](../docker/README.md)
- [Architecture](../docs/architecture.md)
- [API Reference](../docs/api_reference.md)
