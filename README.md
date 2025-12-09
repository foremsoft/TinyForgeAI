# TinyForgeAI

[![CI](https://github.com/anthropics/TinyForgeAI/actions/workflows/ci.yml/badge.svg)](https://github.com/anthropics/TinyForgeAI/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v0.1.0-green.svg)](https://github.com/anthropics/TinyForgeAI/releases)

**Tiny language models — trained from your data — deployed as microservices in seconds.**

A developer-focused open-source framework by FOREM.

## Why TinyForgeAI?

Large LLMs are powerful — but heavy, expensive, and overkill for most enterprise use-cases.

TinyForgeAI gives you **your own small, focused language model** trained only on your data, packaged into a portable REST microservice you can deploy anywhere.

**Built for:**
- Companies wanting private, local, secure AI
- Developers who don't want to learn ML pipelines
- Teams needing simple logic-trained models, not 70B giants
- On-premise or edge inference
- Full automation: **Train → Export → Serve**

## Features

### Train Tiny Language Models From Your Data
- Works with DBs, Google Docs (mock mode), PDFs, DOCX, TXT/MD, APIs
- Dry-run mode instantly simulates fine-tuning (for rapid prototyping)
- PEFT/LoRA adapter stub included for upgrading to real training later

### Auto-Generate Microservice
- Export to ONNX (stub)
- Quantization stub
- Generates complete FastAPI service + Dockerfile + docker-compose

### FOREMForge CLI

```bash
foremforge init
foremforge train --data file.jsonl --out ./model --dry-run
foremforge export --model ./model/model_stub.json --out ./service --export-onnx
foremforge serve --dir ./service --dry-run
```

### Connectors System
| Source | Status | Notes |
|--------|--------|-------|
| SQLite / Postgres | ✔ (SQLite stub) | Streaming reader, batch mode |
| Google Docs | ✔ (mock mode) | No OAuth needed in test environments |
| File Ingesters | ✔ | PDF/DOCX/TXT/MD |
| APIs | Coming soon | Extend via Connector Interface |

Each connector outputs:
```json
{"input": "...", "output": "...", "metadata": {...}}
```

### E2E Demo Included

```bash
bash examples/e2e_demo.sh
```

Runs: Dataset → Training → Export → Smoke-test inference.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              TinyForgeAI                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────┐     ┌────────────────┐     ┌──────────────────┐          │
│   │ CLI/API │────▶│ Data Connectors│────▶│ Trainer          │          │
│   └─────────┘     │                │     │ (Dry-Run / LoRA) │          │
│                   │ • Files        │     └────────┬─────────┘          │
│                   │ • Database     │              │                    │
│                   │ • Google Docs  │              ▼                    │
│                   └────────────────┘     ┌──────────────────┐          │
│                                          │ Artifacts /      │          │
│                                          │ Model Registry   │          │
│                                          └────────┬─────────┘          │
│                                                   │                    │
│                                                   ▼                    │
│   ┌─────────────┐     ┌──────────────────────────────────────┐        │
│   │ Client Apps │◀────│ Inference Server (FastAPI)           │        │
│   └─────────────┘     │                                      │        │
│                       │ ┌──────────────┐  ┌───────────────┐  │        │
│                       │ │ ONNX Export  │  │ Quantization  │  │        │
│                       │ │ (stub)       │  │ (stub)        │  │        │
│                       │ └──────────────┘  └───────────────┘  │        │
│                       └──────────────────────────────────────┘        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train (Dry-Run)

```bash
foremforge train --data examples/data/demo_dataset.jsonl --out tmp/model --dry-run
```

### 3. Export as microservice

```bash
foremforge export --model tmp/model/model_stub.json --out tmp/service --overwrite --export-onnx
```

### 4. Test inference

```python
from fastapi.testclient import TestClient
import importlib.util, sys, pathlib

service_path = pathlib.Path("tmp/service/app.py")
spec = importlib.util.spec_from_file_location("service_app", service_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

client = TestClient(mod.app)
print(client.post("/predict", json={"input": "hello"}).json())
# Output: {"output": "olleh", "confidence": 0.75}
```

Or run the full demo:

```bash
# Using bash (Linux/macOS/Git Bash on Windows)
bash examples/e2e_demo.sh

# Using Python (cross-platform)
python examples/e2e_demo.py
```

## Project Structure

```
TinyForgeAI/
├── backend/
│   ├── dataset/          # JSONL schema validation
│   ├── trainer/          # Training pipeline (stub)
│   ├── exporter/         # Model export and packaging
│   └── api/              # FastAPI application
├── connectors/           # Data source connectors
│   ├── db_connector.py   # SQLite/Postgres
│   ├── google_docs_connector.py
│   └── file_ingest.py    # PDF/DOCX/TXT/MD
├── inference_server/     # Inference service template
├── cli/                  # CLI tool (foremforge)
├── examples/             # Sample data and demo scripts
├── docs/                 # Documentation
├── tests/                # Test suite (215+ tests)
├── docker/               # Docker configurations
├── MODEL_ZOO/            # Pre-built model examples
└── releases/             # Release scripts
```

## Model Zoo (Starter)

```
MODEL_ZOO/example_tiny_model/
├── model_card.md         # Model documentation
└── model_stub.json       # Example model artifact
```

## Run All Tests

```bash
pytest -q
```

## Docker & Deployment

**Build:**

```bash
docker build -f docker/Dockerfile.inference -t tinyforge-inference:local .
```

**Run:**

```bash
docker-compose -f docker/docker-compose.yml up --build
```

The inference server will be available at `http://localhost:8000`.

## Documentation

- [Architecture Guide](docs/architecture.md)
- [Training Guide](docs/training.md)
- [Connectors Guide](docs/connectors.md)
- [Release Guide](docs/release.md)
- [Launch Checklist](docs/launch.md)
- [Developer Onboarding](docs/developer_onboarding.md)
- [Docker Guide](docker/README.md)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Roadmap

- [ ] Real ONNX export
- [ ] True LoRA fine-tuning
- [ ] SaaS dashboard
- [ ] Model Zoo expansion
- [ ] Multi-tenant inference service
- [ ] Indexed & RAG-enabled inference

---

**GitHub:** [https://github.com/anthropics/TinyForgeAI](https://github.com/anthropics/TinyForgeAI)
