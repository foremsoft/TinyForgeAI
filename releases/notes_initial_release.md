# TinyForgeAI v0.1.0 — First Public Release

**Train tiny language models from your own data and deploy them instantly as REST microservices.**

This release includes the full end-to-end TinyForgeAI pipeline.

## Highlights

- **Dry-run training pipeline** — Instantly simulate fine-tuning for rapid prototyping
- **PEFT/LoRA adapter stub** — Ready for real training integration
- **DB + Google Docs (mock) + File ingestion connectors** — Multiple data sources supported
- **Microservice exporter** — FastAPI-based with Docker support
- **ONNX + Quantization stubs** — Export formats ready for expansion
- **Docker + Compose** — Production-ready deployment
- **Full CLI (`foremforge`)** — Simple command-line interface
- **E2E demo** — Complete workflow demonstration
- **Complete documentation + CI** — Ready for contributions

## Quick Demo

```bash
bash examples/e2e_demo.sh
```

Or using Python:

```bash
python examples/e2e_demo.py
```

## Installation

```bash
pip install tinyforgeai
```

Or from source:

```bash
git clone https://github.com/anthropics/TinyForgeAI.git
cd TinyForgeAI
pip install -e ".[dev]"
```

## Quick Start

```bash
# Train (dry-run)
foremforge train --data examples/data/demo_dataset.jsonl --out ./my_model --dry-run

# Export as microservice
foremforge export --model ./my_model/model_stub.json --out ./my_service --export-onnx

# Test the service
python -c "
from fastapi.testclient import TestClient
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('app', pathlib.Path('./my_service/app.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
client = TestClient(mod.app)
print(client.post('/predict', json={'input': 'hello'}).json())
"
```

## Features

### Training
- JSONL dataset format with flexible schema
- Stub trainer for quick prototyping
- Configurable training parameters
- Model artifact generation
- LoRA adapter support (stub)

### Export & Deployment
- FastAPI-based inference server generation
- Health check endpoints (`/health`, `/readyz`)
- Request counting and metrics (`/metrics`)
- Docker support (Dockerfile + docker-compose.yml)
- ONNX export capability (stub)

### Data Connectors
| Source | Status | Notes |
|--------|--------|-------|
| SQLite / Postgres | ✔ | Streaming reader, batch mode |
| Google Docs | ✔ (mock) | No OAuth needed in test mode |
| File Ingesters | ✔ | PDF/DOCX/TXT/MD/JSON/JSONL |

### CLI Commands
- `foremforge init` — Initialize project structure
- `foremforge train` — Train models from JSONL datasets
- `foremforge export` — Export models to inference services
- `foremforge serve` — Run inference services locally
- `foremforge ingest` — Convert files to training format

## Files Included

- `model_stub` example in MODEL_ZOO
- Microservice generator
- Connectors + examples
- Full docs + Model Zoo starter
- 215+ tests

## Documentation

- [README](https://github.com/anthropics/TinyForgeAI#readme)
- [Architecture Guide](https://github.com/anthropics/TinyForgeAI/blob/main/docs/architecture.md)
- [Training Guide](https://github.com/anthropics/TinyForgeAI/blob/main/docs/training.md)
- [Connectors Guide](https://github.com/anthropics/TinyForgeAI/blob/main/docs/connectors.md)
- [Developer Onboarding](https://github.com/anthropics/TinyForgeAI/blob/main/docs/developer_onboarding.md)

## Requirements

- Python 3.10+
- FastAPI, Pydantic, Click, uvicorn

## Roadmap

- Real training engine (HuggingFace Transformers)
- Model Zoo expansion
- Hosted SaaS dashboard
- Multi-tenant inference service
- RAG-enabled inference

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/anthropics/TinyForgeAI/blob/main/CONTRIBUTING.md).

## License

Apache License 2.0

---

**Full Changelog**: https://github.com/anthropics/TinyForgeAI/commits/v0.1.0

Thank you for trying TinyForgeAI!
