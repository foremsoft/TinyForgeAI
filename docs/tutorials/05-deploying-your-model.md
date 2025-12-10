# Deploying Your Model

Learn how to deploy your trained model as a production-ready API that others can use.

## Table of Contents

1. [Deployment Options](#deployment-options)
2. [Local Deployment](#local-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [ONNX Export for Production](#onnx-export-for-production)
6. [Monitoring and Scaling](#monitoring-and-scaling)

---

## Deployment Options

| Option | Best For | Difficulty | Cost |
|--------|----------|------------|------|
| Local | Development, testing | Easy | Free |
| Docker | Consistent environments | Medium | Free |
| Cloud (VM) | Small-scale production | Medium | $$ |
| Cloud (Managed) | Large-scale production | Hard | $$$ |
| ONNX | Edge, mobile, low-latency | Medium | Varies |

---

## Local Deployment

### Quick Start with Inference Server

```bash
# Start the inference server with your model
python -m inference_server.main \
    --model-path ./trained_models/my_model \
    --port 8080
```

### Using the Inference API

```python
# examples/deployment/local_client.py

import requests

# Configuration
API_URL = "http://localhost:8080"

def ask_model(question: str) -> str:
    """Send a question to the model and get a response."""
    response = requests.post(
        f"{API_URL}/predict",
        json={"input": question}
    )
    return response.json()["output"]

# Example usage
if __name__ == "__main__":
    questions = [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
    ]

    for q in questions:
        print(f"Q: {q}")
        print(f"A: {ask_model(q)}")
        print()
```

### Running as a Service

**On Linux/Mac:**
```bash
# Create a systemd service
sudo nano /etc/systemd/system/tinyforge.service
```

```ini
[Unit]
Description=TinyForgeAI Inference Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/tinyforgeai
ExecStart=/opt/tinyforgeai/venv/bin/python -m inference_server.main --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable tinyforge
sudo systemctl start tinyforge
```

**On Windows:**
```python
# Use NSSM or create a Windows service
# Or run with Task Scheduler
```

---

## Docker Deployment

### Building the Docker Image

```bash
# Build the image
docker build -t tinyforgeai:latest .

# With your trained model included
docker build -t my-trained-model:latest \
    --build-arg MODEL_PATH=./trained_models/my_model .
```

### Running the Container

```bash
# Basic run
docker run -p 8080:8080 tinyforgeai:latest

# With GPU support
docker run --gpus all -p 8080:8080 tinyforgeai:latest

# With model volume
docker run -p 8080:8080 \
    -v $(pwd)/trained_models:/app/models \
    -e MODEL_PATH=/app/models/my_model \
    tinyforgeai:latest
```

### Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  inference:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./trained_models:/app/models
    environment:
      - MODEL_PATH=/app/models/my_model
      - LOG_LEVEL=info
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "3000:3000"
    depends_on:
      - inference
    environment:
      - API_URL=http://inference:8080

  # Optional: Add Prometheus for monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f inference

# Scale inference servers
docker-compose up -d --scale inference=3
```

---

## Cloud Deployment

### AWS Deployment

#### Option 1: EC2 (Virtual Machine)

```bash
# 1. Launch EC2 instance (g4dn.xlarge for GPU, or t3.large for CPU)

# 2. SSH into instance
ssh -i your-key.pem ec2-user@your-instance-ip

# 3. Install dependencies
sudo yum update -y
sudo yum install -y docker git

# 4. Clone and run
git clone https://github.com/anthropics/TinyForgeAI.git
cd TinyForgeAI
docker build -t tinyforgeai .
docker run -d -p 80:8080 tinyforgeai
```

#### Option 2: AWS Lambda (Serverless)

For ONNX-exported models:

```python
# lambda_handler.py
import json
import onnxruntime as ort
import numpy as np

# Load model once (cold start)
session = ort.InferenceSession("model.onnx")

def handler(event, context):
    body = json.loads(event['body'])
    input_text = body['input']

    # Tokenize and prepare input
    inputs = prepare_input(input_text)

    # Run inference
    outputs = session.run(None, inputs)

    return {
        'statusCode': 200,
        'body': json.dumps({'output': decode_output(outputs)})
    }
```

### Google Cloud Deployment

```bash
# Using Cloud Run
gcloud run deploy tinyforgeai \
    --image gcr.io/your-project/tinyforgeai \
    --platform managed \
    --region us-central1 \
    --memory 8Gi \
    --cpu 4
```

### Azure Deployment

```bash
# Using Azure Container Instances
az container create \
    --resource-group myResourceGroup \
    --name tinyforgeai \
    --image yourregistry.azurecr.io/tinyforgeai:latest \
    --ports 8080 \
    --cpu 4 \
    --memory 8
```

---

## ONNX Export for Production

### Why ONNX?

```
PyTorch Model          →    ONNX Model
- Python required           - Runs anywhere
- Heavy dependencies        - Lightweight
- ~100ms latency            - ~20ms latency
- Complex deployment        - Simple deployment
```

### Exporting to ONNX

```python
# examples/deployment/export_to_onnx.py

from backend.model_exporter.onnx_exporter import ONNXExporter

# Export your trained model
exporter = ONNXExporter(
    model_path="./trained_models/my_model",
    output_path="./deployed_models/my_model.onnx"
)

# Export with optimization
exporter.export(
    optimize=True,           # Apply ONNX optimizations
    quantize=True,           # Reduce size with quantization
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={           # Support variable batch sizes
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits": {0: "batch_size"},
    }
)

print("Model exported to ONNX!")
print(f"Original size: {exporter.original_size_mb:.1f} MB")
print(f"ONNX size: {exporter.onnx_size_mb:.1f} MB")
```

### Using ONNX Model

```python
# examples/deployment/onnx_inference.py

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

class ONNXInference:
    def __init__(self, model_path: str, tokenizer_path: str):
        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def predict(self, text: str) -> str:
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Run inference
        outputs = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
        )

        # Decode output
        logits = outputs[0]
        predicted_ids = np.argmax(logits, axis=-1)
        return self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)


# Usage
model = ONNXInference("model.onnx", "./tokenizer")
result = model.predict("What is machine learning?")
print(result)
```

### ONNX Performance Comparison

```python
# Benchmark PyTorch vs ONNX
import time

# PyTorch inference
start = time.time()
for _ in range(100):
    pytorch_model.predict("test input")
pytorch_time = time.time() - start

# ONNX inference
start = time.time()
for _ in range(100):
    onnx_model.predict("test input")
onnx_time = time.time() - start

print(f"PyTorch: {pytorch_time/100*1000:.1f}ms per request")
print(f"ONNX: {onnx_time/100*1000:.1f}ms per request")
print(f"Speedup: {pytorch_time/onnx_time:.1f}x")

# Typical results:
# PyTorch: 100ms per request
# ONNX: 20ms per request
# Speedup: 5x
```

---

## Monitoring and Scaling

### Built-in Monitoring

TinyForgeAI includes built-in Prometheus metrics integration:

```python
# Using TinyForgeAI's monitoring module
from backend.monitoring import MetricsRegistry, get_metrics_registry

# Get the shared metrics registry
registry = get_metrics_registry()

# Create custom metrics
request_counter = registry.counter(
    "requests_total",
    "Total requests",
    labels=["endpoint", "method"]
)
latency_histogram = registry.histogram(
    "request_duration_seconds",
    "Request latency in seconds"
)

# Use in your endpoints
@app.post("/predict")
async def predict(request: dict):
    request_counter.labels(endpoint="predict", method="POST").inc()

    with latency_histogram.time():
        result = model.predict(request["input"])
        return {"output": result}
```

### Adding Custom Metrics

```python
# examples/deployment/monitored_server.py

from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
import time

app = FastAPI()

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['endpoint'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')
ERRORS = Counter('errors_total', 'Total errors')

@app.post("/predict")
async def predict(request: dict):
    REQUEST_COUNT.labels(endpoint="predict").inc()

    start = time.time()
    try:
        result = model.predict(request["input"])
        return {"output": result}
    except Exception as e:
        ERRORS.inc()
        raise
    finally:
        REQUEST_LATENCY.observe(time.time() - start)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Rate Limiting

TinyForgeAI includes built-in rate limiting for production deployments:

```python
from services.dashboard_api.rate_limit import rate_limit, RateLimiter

# Decorator-based rate limiting
@app.post("/predict")
@rate_limit(requests=60, window=60)  # 60 requests per minute
async def predict(request: Request, data: dict):
    return {"output": model.predict(data["input"])}

# For distributed deployments, use Redis
import os
os.environ["TINYFORGE_REDIS_URL"] = "redis://localhost:6379/0"
```

Rate limit responses include standard headers:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining in window
- `X-RateLimit-Reset`: Unix timestamp when window resets
- `Retry-After`: Seconds to wait (when rate limited)

### Load Balancing with NGINX

```nginx
# nginx.conf
upstream tinyforge {
    server inference1:8080;
    server inference2:8080;
    server inference3:8080;
}

server {
    listen 80;

    location / {
        proxy_pass http://tinyforge;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /health {
        proxy_pass http://tinyforge/health;
    }
}
```

### Auto-Scaling on Kubernetes

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tinyforgeai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tinyforgeai
  template:
    metadata:
      labels:
        app: tinyforgeai
    spec:
      containers:
      - name: inference
        image: tinyforgeai:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tinyforgeai-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tinyforgeai
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Production Checklist

```
PRE-DEPLOYMENT:
□ Model tested locally
□ Unit tests passing
□ Load tests completed
□ ONNX export verified (if using)
□ Security audit done
□ API documentation ready

INFRASTRUCTURE:
□ Server/container provisioned
□ GPU drivers installed (if needed)
□ SSL/TLS configured
□ Domain/DNS configured
□ Load balancer set up
□ Health checks configured

RATE LIMITING & SECURITY:
□ Rate limiting configured (TINYFORGE_RATE_LIMIT_ENABLED=true)
□ Redis configured for distributed rate limiting
□ Auth endpoints have stricter limits
□ CORS settings configured
□ Input validation enabled

MONITORING:
□ Logging configured
□ Prometheus metrics enabled (backend.monitoring)
□ Grafana dashboards created
□ Alerts configured
□ Error tracking enabled (backend.exceptions)
□ Webhook notifications set up (backend.webhooks)

POST-DEPLOYMENT:
□ Smoke tests passed
□ Performance baseline recorded
□ Backup/rollback plan tested
□ Documentation updated
```

---

## Next Steps

Congratulations! You've completed the TinyForgeAI tutorial series. Here's what to do next:

1. **Contribute** - See the [Contribution Guide](../CONTRIBUTING.md)
2. **Advanced Topics** - Explore RAG, multi-model serving
3. **Community** - Join discussions on GitHub

---

## Quick Reference

```bash
# Local deployment
python -m inference_server.main --model-path ./model --port 8080

# Docker deployment
docker build -t mymodel .
docker run -p 8080:8080 mymodel

# ONNX export
python -m backend.model_exporter.onnx_exporter \
    --model-path ./model \
    --output ./model.onnx

# Test the API
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"input": "What is AI?"}'
```
