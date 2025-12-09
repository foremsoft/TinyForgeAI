# TinyForgeAI Docker Guide

This directory contains Docker configuration for running TinyForgeAI services.

## Quick Start

### Build and Run with Docker Compose

```bash
# Navigate to docker directory
cd docker

# Create model registry directory
mkdir -p model_registry

# Build and start the service
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

The inference server will be available at `http://localhost:8000`.

### Build Image Manually

```bash
# From repository root
docker build -f docker/Dockerfile.inference -t tinyforge-inference:local .
```

### Run Container Manually

```bash
# Basic run
docker run -p 8000:8000 tinyforge-inference:local

# With model registry volume
docker run -p 8000:8000 \
  -v $(pwd)/model_registry:/app/model_registry \
  tinyforge-inference:local

# With environment variables
docker run -p 8000:8000 \
  -e LOG_LEVEL=DEBUG \
  -e MODEL_REGISTRY_PATH=/app/model_registry \
  -v $(pwd)/model_registry:/app/model_registry \
  tinyforge-inference:local
```

## Model Registry

The inference server expects model files in the `/app/model_registry` directory inside the container.

### Expected Files

| File | Description |
|------|-------------|
| `model_stub.json` | Model metadata and configuration |
| `model.onnx` | ONNX model file (optional) |
| `quantized.onnx` | Quantized ONNX model (optional) |

### Providing Models

1. Create a `model_registry` directory:
   ```bash
   mkdir -p docker/model_registry
   ```

2. Copy your model files:
   ```bash
   cp /path/to/model_stub.json docker/model_registry/
   cp /path/to/model.onnx docker/model_registry/
   ```

3. Start the container with the volume mount (docker-compose does this automatically).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PORT` | `8000` | Port the server listens on |
| `MODEL_REGISTRY_PATH` | `/app/model_registry` | Path to model files |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering |

## Docker Compose Commands

```bash
# Start services
docker-compose up

# Start in detached mode
docker-compose up -d

# Rebuild and start
docker-compose up --build

# View logs
docker-compose logs -f inference

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Health Check

The container includes a health check that queries the `/health` endpoint:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' tinyforge-inference

# Manual health check
curl http://localhost:8000/health
```

## API Endpoints

Once running, the inference server exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Run inference |
| `/docs` | GET | OpenAPI documentation |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

## Development Mode

For development with live code reloading, uncomment the inference_server volume in `docker-compose.yml`:

```yaml
volumes:
  - ./model_registry:/app/model_registry:rw
  - ../inference_server:/app/inference_server:ro  # Uncomment this
```

Then restart with:
```bash
docker-compose up --build
```

## Testing Docker Configuration

Tests for Docker files run without requiring Docker to be installed. To run the optional Docker build test locally:

```bash
# Run all tests (Docker build test skipped by default)
pytest tests/test_docker_files.py -v

# Run with Docker build test enabled (requires Docker)
DOCKER_AVAILABLE=true pytest tests/test_docker_files.py -v
```

The `DOCKER_AVAILABLE` environment variable enables the actual Docker build test. This is skipped in CI environments where Docker may not be available.

## Troubleshooting

### Permission Denied on Mounted Volumes

If you encounter permission errors with mounted volumes:

```bash
# Option 1: Run as your user
docker run --user $(id -u):$(id -g) -p 8000:8000 \
  -v $(pwd)/model_registry:/app/model_registry \
  tinyforge-inference:local

# Option 2: Fix permissions on host
chmod -R 755 docker/model_registry
```

### Port Already in Use

If port 8000 is already in use:

```bash
# Option 1: Use a different port
docker run -p 8080:8000 tinyforge-inference:local

# Option 2: Find and stop the process using port 8000
lsof -i :8000
kill <PID>
```

### Container Exits Immediately

Check container logs for errors:

```bash
docker-compose logs inference
# or
docker logs tinyforge-inference
```

### Model Not Found

Ensure your model files are in the correct location:

```bash
ls -la docker/model_registry/
```

The container expects files at `/app/model_registry/`.
