# Configuration Guide

This document provides comprehensive documentation for all TinyForgeAI environment variables and configuration options.

## Quick Start

```bash
# Copy the example configuration
cp .env.example .env

# Edit with your settings
nano .env
```

## Environment Variables Reference

### Application Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TINYFORGE_LOG_LEVEL` | `INFO` | Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `TINYFORGE_ENV` | `development` | Environment mode. Options: `development`, `staging`, `production` |

---

### Model Training Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TINYFORGE_DEFAULT_MODEL` | `distilbert-base-uncased` | Default HuggingFace model for training |
| `TINYFORGE_OUTPUT_DIR` | `./output` | Directory for trained model outputs |
| `TINYFORGE_DEVICE` | `auto` | Compute device. Options: `auto`, `cpu`, `cuda`, `cuda:0`, `cuda:1`, `mps` |
| `TINYFORGE_USE_FP16` | `true` | Enable mixed precision training (reduces memory usage) |

---

### Training Hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `TINYFORGE_NUM_EPOCHS` | `3` | Number of training epochs |
| `TINYFORGE_BATCH_SIZE` | `8` | Training batch size (reduce if OOM) |
| `TINYFORGE_LEARNING_RATE` | `2e-5` | Learning rate for optimizer |
| `TINYFORGE_MAX_LENGTH` | `512` | Maximum sequence length for tokenization |

---

### LoRA Settings

LoRA (Low-Rank Adaptation) enables efficient fine-tuning with reduced memory usage.

| Variable | Default | Description |
|----------|---------|-------------|
| `TINYFORGE_USE_LORA` | `false` | Enable LoRA fine-tuning |
| `TINYFORGE_LORA_R` | `8` | LoRA rank (lower = smaller adapter) |
| `TINYFORGE_LORA_ALPHA` | `16` | LoRA scaling factor (typically 2x rank) |
| `TINYFORGE_LORA_DROPOUT` | `0.1` | Dropout rate for LoRA layers |

---

### RAG (Retrieval-Augmented Generation) Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TINYFORGE_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model for document embeddings |
| `TINYFORGE_CHUNK_SIZE` | `512` | Document chunk size in characters |
| `TINYFORGE_CHUNK_OVERLAP` | `50` | Overlap between chunks in characters |
| `TINYFORGE_INDEX_PATH` | `./data/index` | Path for storing vector index |

---

### Inference Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TINYFORGE_HOST` | `0.0.0.0` | Server bind address |
| `TINYFORGE_PORT` | `8000` | Server port |
| `TINYFORGE_WORKERS` | `1` | Number of worker processes |
| `TINYFORGE_CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated or `*`) |

---

### Dashboard API Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TINYFORGE_DASHBOARD_PORT` | `8001` | Dashboard API port |

---

### Authentication Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TINYFORGE_AUTH_ENABLED` | `false` | Enable API authentication |
| `TINYFORGE_API_KEY` | *(empty)* | API key for programmatic access |
| `TINYFORGE_USERNAME` | `admin` | Default admin username |
| `TINYFORGE_PASSWORD` | `tinyforge` | Default admin password (change in production!) |
| `TINYFORGE_SECRET_KEY` | `your-secret-key-here` | JWT secret key |

**Security Note**: Always change default credentials and generate secure keys in production:
```bash
# Generate a secure API key
openssl rand -hex 32

# Generate a secure secret key
openssl rand -hex 32
```

---

### Database Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TINYFORGE_DATABASE_URL` | `sqlite:///./data/tinyforge.db` | Database connection URL |
| `TINYFORGE_USE_DATABASE` | `true` | Enable database for job tracking |

Supported databases:
- SQLite: `sqlite:///./data/tinyforge.db`
- PostgreSQL: `postgresql://user:pass@host:5432/dbname`
- MySQL: `mysql://user:pass@host:3306/dbname`

---

### External Integrations

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | *(empty)* | HuggingFace Hub token for private models |
| `WANDB_API_KEY` | *(empty)* | Weights & Biases API key for experiment tracking |
| `MLFLOW_TRACKING_URI` | *(empty)* | MLflow tracking server URI |

Get your tokens:
- HuggingFace: https://huggingface.co/settings/tokens
- W&B: https://wandb.ai/settings

---

### Docker/Deployment Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCKER_TAG` | `latest` | Docker image tag |
| `DOCKER_MEMORY_LIMIT` | `4g` | Container memory limit |
| `DOCKER_GPU` | `none` | GPU support. Options: `nvidia`, `none` |

---

## Example Configurations

### Development Setup

Minimal configuration for local development:

```env
TINYFORGE_ENV=development
TINYFORGE_LOG_LEVEL=DEBUG
TINYFORGE_DEVICE=cpu
TINYFORGE_AUTH_ENABLED=false
```

### Production Setup

Secure configuration for production deployment:

```env
TINYFORGE_ENV=production
TINYFORGE_LOG_LEVEL=WARNING
TINYFORGE_DEVICE=auto
TINYFORGE_USE_FP16=true

# Authentication (generate secure keys!)
TINYFORGE_AUTH_ENABLED=true
TINYFORGE_API_KEY=<generate-with-openssl-rand-hex-32>
TINYFORGE_SECRET_KEY=<generate-with-openssl-rand-hex-32>
TINYFORGE_USERNAME=admin
TINYFORGE_PASSWORD=<strong-password>

# Database
TINYFORGE_DATABASE_URL=postgresql://user:pass@localhost:5432/tinyforge

# Server
TINYFORGE_WORKERS=4
TINYFORGE_CORS_ORIGINS=https://yourdomain.com
```

### GPU Training Setup

Configuration for GPU-accelerated training:

```env
TINYFORGE_DEVICE=cuda
TINYFORGE_USE_FP16=true
TINYFORGE_BATCH_SIZE=16
TINYFORGE_USE_LORA=true
TINYFORGE_LORA_R=16
TINYFORGE_LORA_ALPHA=32

# Memory optimization
TINYFORGE_MAX_LENGTH=256
```

### Apple Silicon (M1/M2/M3) Setup

Configuration for Apple Silicon Macs:

```env
TINYFORGE_DEVICE=mps
TINYFORGE_USE_FP16=false
TINYFORGE_BATCH_SIZE=4
```

### Docker Deployment

Configuration for Docker containers:

```env
TINYFORGE_HOST=0.0.0.0
TINYFORGE_PORT=8000
DOCKER_MEMORY_LIMIT=8g
DOCKER_GPU=nvidia
DOCKER_TAG=v0.3.0
```

---

## Configuration Priority

Configuration values are loaded in the following order (later overrides earlier):

1. Default values in code
2. `.env` file in project root
3. Environment variables
4. Command-line arguments (where supported)

---

## Validation

To validate your configuration:

```bash
# Check if .env file exists
ls -la .env

# View current environment
env | grep TINYFORGE

# Test configuration loading
python -c "from backend.config import settings; print(settings)"
```

---

## Troubleshooting

### Common Issues

**Out of Memory (OOM) Errors**
```env
# Reduce batch size
TINYFORGE_BATCH_SIZE=4

# Enable memory optimization
TINYFORGE_USE_FP16=true
TINYFORGE_USE_LORA=true
```

**CUDA Not Available**
```env
# Fall back to CPU
TINYFORGE_DEVICE=cpu
```

**Database Connection Failed**
```env
# Use in-memory storage for testing
TINYFORGE_USE_DATABASE=false

# Or use SQLite
TINYFORGE_DATABASE_URL=sqlite:///./data/tinyforge.db
```

**Authentication Issues**
```env
# Disable for development
TINYFORGE_AUTH_ENABLED=false

# Check API key format (should be 64 hex characters)
TINYFORGE_API_KEY=your-64-char-hex-key
```

---

## See Also

- [README.md](../README.md) - Getting started guide
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development setup
- [Docker Deployment](deployment.md) - Container configuration
