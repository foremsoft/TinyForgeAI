# TinyForgeAI

[![CI](https://github.com/foremsoft/TinyForgeAI/actions/workflows/ci.yml/badge.svg)](https://github.com/foremsoft/TinyForgeAI/actions/workflows/ci.yml)
[![Docker](https://github.com/foremsoft/TinyForgeAI/actions/workflows/docker_build.yml/badge.svg)](https://github.com/foremsoft/TinyForgeAI/actions/workflows/docker_build.yml)
[![codecov](https://codecov.io/gh/foremsoft/TinyForgeAI/graph/badge.svg)](https://codecov.io/gh/foremsoft/TinyForgeAI)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v0.1.0-green.svg)](https://github.com/foremsoft/TinyForgeAI/releases)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Good First Issues](https://img.shields.io/github/issues/foremsoft/TinyForgeAI/good%20first%20issue?color=7057ff&label=good%20first%20issues)](https://github.com/foremsoft/TinyForgeAI/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

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

### Model Zoo with Pre-configured Models
- **13 pre-configured models** across 9 task types
- Task types: Q&A, Summarization, Classification, Sentiment, Code Generation, Conversation, NER, Translation, Text Generation
- CLI for browsing, selecting, and training models
- Sample datasets included for quick experimentation

### Model Evaluation & Benchmarks
- Comprehensive evaluation metrics: BLEU, ROUGE, F1, Exact Match, Perplexity
- Benchmark runner for comparing models on datasets
- CLI for evaluating trained models

### RAG (Retrieval-Augmented Generation)
- Document indexing with sentence-transformers embeddings
- Semantic search across your documents
- Easy integration with training pipeline

### Web Dashboard
- React-based dashboard UI for training management
- **Easy Mode**: Step-by-step wizard for non-technical users
- **Advanced Mode**: Full control for power users
- Real-time job monitoring and logs
- Model browsing and playground for inference testing
- Service health monitoring

### Training UIs for Everyone
Three training interfaces designed for different users:

| Interface | Best For | Features |
|-----------|----------|----------|
| **Gradio** | Demos, beginners | Instant shareable links, drag-and-drop |
| **Streamlit** | Data scientists | Rich visualizations, batch testing |
| **React Dashboard** | Production | Easy/Advanced modes, job management |

```bash
# Gradio (simplest)
cd ui/gradio && pip install -r requirements.txt && python training_app.py

# Streamlit (data exploration)
cd ui/streamlit && pip install -r requirements.txt && streamlit run training_app.py

# React Dashboard (production)
cd dashboard && npm install && npm run dev
```

### Auto-Generate Microservice
- Export to ONNX for optimized inference
- Quantization support for smaller models
- Generates complete FastAPI service + Dockerfile + docker-compose
- Dashboard API with Prometheus metrics for monitoring

### Model Registry
- **Semantic Versioning**: Full SemVer 2.0.0 support with major, minor, patch bumping
- **Lifecycle Management**: Draft, Active, Staged, Deprecated, Archived states
- **Model Cards**: HuggingFace-style documentation with markdown export
- **Training Provenance**: Track metrics, configuration, data sources, and lineage
- **CLI Interface**: Full command-line management (`python -m backend.model_registry.cli`)

### Production-Ready Backend
- **Rate Limiting**: Configurable rate limiting with sliding window (in-memory or Redis-based for distributed deployments)
- **Webhooks**: Event-driven notifications for job status changes, model training, and system events
- **Monitoring**: Prometheus metrics integration with custom counters, gauges, and histograms
- **Exception Handling**: Comprehensive error hierarchy with structured error responses

### Killer Features (NEW!)

#### One-Click URL Training
Train models directly from URLs - no data preparation needed!
```bash
# Train from Notion page
foremforge train-url https://notion.so/my-faq-page --out ./model

# Train from Google Docs
foremforge train-url https://docs.google.com/document/d/xxx

# Preview what will be extracted
foremforge train-url https://example.com/faq --preview
```

#### Training Data Generator
Generate 500+ training samples from just 5 examples:
```bash
# Augment your data
foremforge augment -i examples.jsonl -o augmented.jsonl -n 500

# Use LLM for higher quality (requires API key)
foremforge augment -i data.jsonl -o more_data.jsonl --use-llm
```

#### Shareable Playground
Create instant, shareable demos of your models:
```bash
# Local playground
foremforge playground --model ./my_model

# Get a public shareable link
foremforge playground --model ./my_model --share

# Export as standalone HTML (works offline!)
foremforge playground --model ./my_model --export demo.html
```

### FOREMForge CLI

```bash
foremforge init
foremforge train --data file.jsonl --out ./model --dry-run
foremforge train-url https://notion.so/my-faq  # NEW: Train from URL
foremforge augment -i data.jsonl -o more.jsonl  # NEW: Generate data
foremforge export --model ./model/model_stub.json --out ./service --export-onnx
foremforge serve --dir ./service --dry-run
foremforge playground --model ./model --share  # NEW: Shareable demo
```

### Connectors System
| Source | Status | Notes |
|--------|--------|-------|
| SQLite / Postgres | ✔ (SQLite stub) | Streaming reader, batch mode |
| Google Docs | ✔ (mock mode) | No OAuth needed in test environments |
| Google Drive | ✔ | Files, folders, workspace exports |
| Notion | ✔ | Pages, databases, properties |
| Slack | ✔ | Channels, messages, threads |
| Confluence | ✔ | Spaces, pages, search |
| File Ingesters | ✔ | PDF/DOCX/TXT/MD |
| REST APIs | ✔ | Pagination, auth, rate limiting |

Each connector outputs:
```json
{"input": "...", "output": "...", "metadata": {...}}
```

### MCP Integration (Model Context Protocol)

TinyForgeAI includes an MCP server that enables AI assistants (Claude, ChatGPT, Gemini) to interact with your training pipeline directly.

```bash
# Install MCP support
pip install mcp

# Run MCP server
python -m mcp.server
```

**With Claude Desktop**, add to config:
```json
{
    "mcpServers": {
        "tinyforge": {
            "command": "python",
            "args": ["-m", "mcp.server"],
            "cwd": "/path/to/TinyForgeAI"
        }
    }
}
```

Then ask Claude:
- "List available models in TinyForgeAI"
- "Train a Q&A model on my FAQ data"
- "Search my indexed documents for authentication"

See [MCP Documentation](mcp/README.md) for full details.

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
│   ├── evaluation/       # Model evaluation & benchmarks
│   │   ├── evaluator.py  # Evaluation runner
│   │   ├── metrics.py    # BLEU, ROUGE, F1, etc.
│   │   └── benchmark.py  # Benchmark comparison
│   ├── model_exporter/   # ONNX export
│   ├── exceptions.py     # Custom exception hierarchy
│   ├── webhooks.py       # Webhook event system
│   ├── monitoring.py     # Prometheus metrics
│   └── api/              # FastAPI application
├── connectors/           # Data source connectors
│   ├── db_connector.py   # SQLite/Postgres
│   ├── api_connector.py  # REST API connector
│   ├── google_docs.py    # Google Docs integration
│   ├── google_drive_connector.py  # Google Drive files
│   ├── notion_connector.py  # Notion pages/databases
│   └── indexer.py        # RAG document indexer
├── model_zoo/            # Pre-configured model registry
│   ├── registry.py       # Model configurations (13 models)
│   ├── cli.py            # Model Zoo CLI
│   └── datasets/         # Sample training datasets
├── services/             # Microservices
│   ├── dashboard_api/    # Dashboard REST API
│   │   └── rate_limit.py # Rate limiting middleware
│   └── training_worker/  # Async training job worker
├── ui/                   # Training User Interfaces
│   ├── gradio/           # Gradio interface (demos/beginners)
│   └── streamlit/        # Streamlit interface (data scientists)
├── dashboard/            # React web dashboard
│   └── src/
│       ├── components/   # TrainingWizard (Easy Mode)
│       └── pages/        # Train, Models, Logs, Playground
├── inference_server/     # Inference service template
├── cli/                  # CLI tool (foremforge)
├── mcp/                  # MCP server for AI assistant integration
│   ├── server.py         # Main MCP server
│   ├── tools/            # Training, inference, connector tools
│   └── resources/        # Model registry, job status resources
├── deploy/               # Deployment configurations
│   ├── k8s/              # Kubernetes manifests
│   ├── helm-chart/       # Helm chart for production
│   ├── aws/              # AWS CloudFormation + Terraform
│   ├── gcp/              # GCP Terraform
│   └── azure/            # Azure Terraform
├── examples/             # Sample code
│   ├── training/         # Training examples
│   ├── rag/              # RAG examples
│   └── tutorial_data/    # Sample datasets for tutorials
├── docs/                 # Documentation
│   └── tutorials/
│       ├── hands-on/     # Practical tutorials (90 min)
│       └── beginners-course/  # Complete AI course (4 hours)
├── tests/                # Test suite (456+ tests)
├── docker/               # Docker configurations
└── .github/workflows/    # CI/CD pipelines
```

## Model Zoo

Browse and train pre-configured models for various NLP tasks:

```bash
# List all available models
python -m model_zoo.cli list

# Get detailed info about a model
python -m model_zoo.cli info qa_flan_t5_small

# Train with a pre-configured model (dry run)
python -m model_zoo.cli train qa_flan_t5_small --dry-run

# Train with your own data
python -m model_zoo.cli train summarization_t5_small --data your_data.jsonl
```

**Available Task Types:**
| Task | Models | Description |
|------|--------|-------------|
| Question Answering | qa_flan_t5_small, qa_flan_t5_base | Answer questions from context |
| Summarization | summarization_t5_small, summarization_bart | Summarize long text |
| Classification | classification_distilbert | Text classification |
| Sentiment | sentiment_roberta | Sentiment analysis |
| Code Generation | code_gen_small | Generate code from prompts |
| Conversation | chat_gpt2_small, chat_dialogpt | Chatbot responses |
| NER | ner_bert | Named entity recognition |
| Translation | translation_en_es, translation_en_fr | Language translation |
| Text Generation | text_gen_gpt2_medium | Open-ended text generation |

## Model Evaluation

Evaluate your trained models with comprehensive metrics:

```bash
# Evaluate a model on a test dataset
python -m backend.evaluation.cli evaluate \
    --model ./my_model \
    --data test_data.jsonl \
    --metrics bleu rouge f1

# Run benchmarks comparing multiple models
python -m backend.evaluation.cli benchmark \
    --models model1 model2 \
    --dataset benchmark_data.jsonl
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

### Cloud Deployment

Deploy to AWS, GCP, or Azure with Infrastructure-as-Code templates:

| Cloud | IaC Tools | Guide |
|-------|-----------|-------|
| **AWS** | CloudFormation, Terraform | [AWS Guide](deploy/aws/README.md) |
| **GCP** | Terraform | [GCP Guide](deploy/gcp/README.md) |
| **Azure** | Terraform | [Azure Guide](deploy/azure/README.md) |

```bash
# AWS EKS
cd deploy/aws/terraform && terraform init && terraform apply

# GCP GKE
cd deploy/gcp/terraform && terraform init && terraform apply

# Azure AKS
cd deploy/azure/terraform && terraform init && terraform apply
```

Each deployment includes managed Kubernetes, container registry, object storage, autoscaling, and optional GPU support.

## Documentation

### Tutorials (Start Here!)

**Beginner's Course (4 hours)** - Learn AI from scratch:
| Module | Topic | Description |
|--------|-------|-------------|
| [00](docs/tutorials/beginners-course/00-what-is-ai.md) | What is AI? | AI concepts explained (no code) |
| [01-02](docs/tutorials/beginners-course/01-setup-your-computer.md) | Setup & First Script | Environment setup, first AI code |
| [03-04](docs/tutorials/beginners-course/03-understanding-data.md) | Data & Simple Bot | Data formats, build a chatbot |
| [05-07](docs/tutorials/beginners-course/05-what-is-a-model.md) | Models & Training | Train your own AI model |
| [08-10](docs/tutorials/beginners-course/08-test-and-improve.md) | Test, Deploy, Next Steps | Improve, deploy, advanced topics |

**Hands-On Tutorials (130 min)** - Build real projects:
| Tutorial | Time | What You Build |
|----------|------|----------------|
| [Quick Start](docs/tutorials/hands-on/00-quickstart.md) | 5 min | Install & first script |
| [FAQ Bot](docs/tutorials/hands-on/01-faq-bot.md) | 15 min | Working chatbot |
| [Document Search](docs/tutorials/hands-on/02-document-search.md) | 20 min | Search PDFs & docs |
| [Train Model](docs/tutorials/hands-on/03-train-your-model.md) | 30 min | Custom AI model |
| [Deploy](docs/tutorials/hands-on/04-deploy-your-project.md) | 20 min | Put it online |
| [MCP Integration](docs/tutorials/hands-on/05-mcp-integration.md) | 15 min | AI assistant control |
| [Killer Features](docs/tutorials/hands-on/06-killer-features.md) | 25 min | URL training, augmentation, playground |

**Concept Tutorials:**
| Tutorial | Description |
|----------|-------------|
| [01 - Introduction to AI Training](docs/tutorials/01-introduction-to-ai-training.md) | Beginner-friendly intro to AI concepts |
| [02 - Preparing Training Data](docs/tutorials/02-preparing-training-data.md) | How to format your data for training |
| [03 - Training Your First Model](docs/tutorials/03-training-your-first-model.md) | Hands-on training walkthrough |
| [04 - Understanding LoRA](docs/tutorials/04-understanding-lora.md) | Efficient fine-tuning explained |
| [05 - Deploying Your Model](docs/tutorials/05-deploying-your-model.md) | Local, Docker, and cloud deployment |

### Reference Guides

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and components
- [MCP Integration Guide](docs/mcp.md) - AI assistant integration via Model Context Protocol
- [Contributing Guide](docs/CONTRIBUTING.md) - How to contribute
- [Training Guide](docs/training.md) - Training details
- [Connectors Guide](docs/connectors.md) - Data source integrations
- [Use Cases Guide](docs/use_cases.md) - Practical use cases with examples
- [Training UIs Guide](ui/README.md) - Gradio, Streamlit, React interfaces
- [Docker Guide](docker/README.md) - Container deployment

### Examples

- [Training Quick Start](examples/training/quick_start.py) - Train a model in minutes
- [RAG Quick Start](examples/rag/quick_start_rag.py) - Index and search documents
- [Tutorial Data](examples/tutorial_data/) - Sample datasets for learning

## Community & Support

- [GitHub Discussions](https://github.com/foremsoft/TinyForgeAI/discussions) - Questions, ideas, and community support
- [GitHub Issues](https://github.com/foremsoft/TinyForgeAI/issues) - Bug reports and feature requests
- [Wiki](https://github.com/foremsoft/TinyForgeAI/wiki) - Tutorials and guides

## Contributing

We welcome contributions from developers of all skill levels! Whether you're fixing a typo, adding a feature, or improving documentation, your help is appreciated.

[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Good First Issues](https://img.shields.io/github/issues/foremsoft/TinyForgeAI/good%20first%20issue?color=7057ff&label=good%20first%20issues)](https://github.com/foremsoft/TinyForgeAI/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

**Ways to contribute:**
- Report bugs and suggest features via [Issues](https://github.com/foremsoft/TinyForgeAI/issues)
- Pick up a [good first issue](https://github.com/foremsoft/TinyForgeAI/labels/good%20first%20issue) to get started
- Improve documentation or add tutorials
- Add new connectors or enhance existing ones
- Help review pull requests

**Quick start for contributors:**
```bash
git clone https://github.com/foremsoft/TinyForgeAI.git
cd TinyForgeAI
pip install -e ".[all]"
pytest  # Run tests
```

See our [Contributing Guide](docs/CONTRIBUTING.md) for detailed instructions and our [Contributors](CONTRIBUTORS.md) to see who's helping build TinyForgeAI.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Roadmap

- [x] Real training with HuggingFace Transformers
- [x] LoRA/PEFT fine-tuning support
- [x] RAG document indexing
- [x] Comprehensive tutorials
- [x] Model evaluation benchmarks (BLEU, ROUGE, F1, Perplexity)
- [x] Web dashboard UI (React-based)
- [x] Model Zoo expansion (13 pre-configured models, 9 task types)
- [x] Multi-tenant inference service
- [x] Cloud deployment templates (AWS, GCP, Azure)
- [x] A/B testing for model comparisons
- [x] Production-ready backend (rate limiting, webhooks, monitoring, exceptions)
- [x] REST API connector with pagination and auth support
- [x] Google Drive connector with workspace file exports
- [x] Notion connector for pages and databases
- [x] Model versioning and registry
- [x] **Training UIs** for non-technical users (Gradio, Streamlit, React Easy Mode)
- [x] **Beginner's AI Course** (11 modules, learn AI from scratch)
- [x] **Hands-On Tutorials** (build real projects in 90 minutes)
- [x] **MCP Integration** (13 tools, 4 resources for AI assistant control)
- [x] **One-Click URL Training** (train from Notion, Google Docs, websites)
- [x] **Training Data Generator** (augment 5 examples to 500+ samples)
- [x] **Shareable Playground** (instant demos with public links or offline HTML)

---

**GitHub:** [https://github.com/foremsoft/TinyForgeAI](https://github.com/foremsoft/TinyForgeAI)
