# TinyForgeAI

[![CI](https://github.com/foremsoft/TinyForgeAI/actions/workflows/ci.yml/badge.svg)](https://github.com/foremsoft/TinyForgeAI/actions/workflows/ci.yml)
[![Docker](https://github.com/foremsoft/TinyForgeAI/actions/workflows/docker_build.yml/badge.svg)](https://github.com/foremsoft/TinyForgeAI/actions/workflows/docker_build.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v0.1.0-green.svg)](https://github.com/foremsoft/TinyForgeAI/releases)

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
- **Real training** with HuggingFace Transformers + PEFT/LoRA support
- Dry-run mode for rapid prototyping without GPU
- Support for DistilBERT, GPT-2, Llama, and other HuggingFace models

### RAG (Retrieval-Augmented Generation)
- Document indexing with sentence-transformers embeddings
- Semantic search across your documents
- Easy integration with training pipeline

### Auto-Generate Microservice
- Export to ONNX for optimized inference
- Quantization support for smaller models
- Generates complete FastAPI service + Dockerfile + docker-compose
- Dashboard API for monitoring and management

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

### 1. Install

```bash
# Basic installation
pip install -e .

# With training support (PyTorch, Transformers, PEFT)
pip install -e ".[training]"

# With RAG support (sentence-transformers)
pip install -e ".[rag]"

# Everything
pip install -e ".[all]"
```

### 2. Train a Model

**Quick Start (Real Training):**
```bash
python examples/training/quick_start.py
```

**Or use the CLI (Dry-Run):**
```bash
foremforge train --data examples/data/demo_dataset.jsonl --out tmp/model --dry-run
```

**Python API:**
```python
from backend.training.real_trainer import RealTrainer, TrainingConfig

config = TrainingConfig(
    model_name="distilbert-base-uncased",
    output_dir="./my_model",
    num_epochs=3,
    batch_size=8,
    use_lora=True,  # Efficient fine-tuning
)

trainer = RealTrainer(config)
trainer.train("data.jsonl")
```

### 3. RAG Document Indexing

```python
from connectors.indexer import DocumentIndexer, IndexerConfig

indexer = DocumentIndexer(IndexerConfig())
indexer.index_file("documents/manual.pdf")

results = indexer.search("How do I reset my password?", top_k=3)
for r in results:
    print(f"Score: {r.score:.3f} - {r.document.content[:100]}...")
```

**Or run the RAG example:**
```bash
python examples/rag/quick_start_rag.py
```

### 4. Export & Deploy

```bash
foremforge export --model tmp/model/model_stub.json --out tmp/service --export-onnx
```

### 5. Run Demo

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
│   ├── config/           # Configuration management (Pydantic)
│   ├── data_processing/  # Dataset loading and file ingestion
│   ├── training/         # Training pipeline
│   │   ├── trainer.py    # Abstract trainer interface
│   │   └── real_trainer.py # HuggingFace/PEFT implementation
│   ├── model_exporter/   # ONNX export
│   └── api/              # FastAPI application
├── connectors/           # Data source connectors
│   ├── db_connector.py   # SQLite/Postgres
│   ├── google_docs.py    # Google Docs integration
│   └── indexer.py        # RAG document indexer
├── services/             # Microservices
│   └── dashboard_api/    # Dashboard REST API
├── inference_server/     # Inference service template
├── cli/                  # CLI tool (foremforge)
├── examples/             # Sample code
│   ├── training/         # Training examples
│   └── rag/              # RAG examples
├── docs/                 # Documentation
│   └── tutorials/        # Step-by-step tutorials
├── tests/                # Test suite (260+ tests)
├── docker/               # Docker configurations
├── MODEL_ZOO/            # Pre-built model examples
└── .github/workflows/    # CI/CD pipelines
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

### Tutorials (Start Here!)

| Tutorial | Description |
|----------|-------------|
| [01 - Introduction to AI Training](docs/tutorials/01-introduction-to-ai-training.md) | Beginner-friendly intro to AI concepts |
| [02 - Preparing Training Data](docs/tutorials/02-preparing-training-data.md) | How to format your data for training |
| [03 - Training Your First Model](docs/tutorials/03-training-your-first-model.md) | Hands-on training walkthrough |
| [04 - Understanding LoRA](docs/tutorials/04-understanding-lora.md) | Efficient fine-tuning explained |
| [05 - Deploying Your Model](docs/tutorials/05-deploying-your-model.md) | Local, Docker, and cloud deployment |

### Reference Guides

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and components
- [Contributing Guide](docs/CONTRIBUTING.md) - How to contribute
- [Training Guide](docs/training.md) - Training details
- [Connectors Guide](docs/connectors.md) - Data source integrations
- [Docker Guide](docker/README.md) - Container deployment

### Examples

- [Training Quick Start](examples/training/quick_start.py) - Train a model in minutes
- [RAG Quick Start](examples/rag/quick_start_rag.py) - Index and search documents

## Contributing

We welcome contributions! Please see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

**Quick start for contributors:**
```bash
git clone https://github.com/foremsoft/TinyForgeAI.git
cd TinyForgeAI
pip install -e ".[all]"
pytest  # Run tests
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Roadmap

- [x] Real training with HuggingFace Transformers
- [x] LoRA/PEFT fine-tuning support
- [x] RAG document indexing
- [x] Comprehensive tutorials
- [ ] Web dashboard UI
- [ ] Model Zoo expansion
- [ ] Multi-tenant inference service
- [ ] Cloud deployment templates (AWS, GCP, Azure)
- [ ] Model evaluation benchmarks

---

**GitHub:** [https://github.com/foremsoft/TinyForgeAI](https://github.com/foremsoft/TinyForgeAI)
