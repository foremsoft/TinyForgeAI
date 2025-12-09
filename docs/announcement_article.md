# TinyForgeAI — Tiny Models, Big Impact

*Train private, fast, and deployable language models from your own data*

---

## The Problem with Large Language Models

Large language models are powerful, but they're not always the right tool for the job.

**The reality for most businesses:**
- You don't need 70 billion parameters to answer questions about your company's refund policy
- Cloud API costs add up quickly for high-volume use cases
- Sensitive data can't always be sent to external APIs
- Latency requirements often exceed what cloud services can deliver
- Compliance regulations may mandate on-premise solutions

**What most companies actually need:**
- A small, focused model trained on *their* specific data
- Fast inference times (milliseconds, not seconds)
- Complete data privacy and control
- Predictable costs with no per-request fees
- Easy deployment to existing infrastructure

## Introducing TinyForgeAI

TinyForgeAI is an open-source framework that lets you train tiny language models from your own data and deploy them as REST microservices — all from the command line.

No ML expertise required. No cloud dependencies. No massive GPU clusters.

### How It Works

```
Your Data → Train → Export → Deploy → Serve
```

1. **Ingest your data** from databases, files, or Google Docs
2. **Train a tiny model** on your specific use case
3. **Export as a microservice** with one command
4. **Deploy anywhere** — Docker, Kubernetes, or bare metal

### Three Commands to Production

```bash
# Train a model from your data
foremforge train --data your_data.jsonl --out ./model --dry-run

# Export as a microservice
foremforge export --model ./model/model_stub.json --out ./service

# Deploy and serve
foremforge serve --dir ./service --port 8000
```

That's it. You now have a REST API answering questions based on your data.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        TinyForgeAI                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐     ┌──────────────┐     ┌──────────────┐    │
│   │  CLI    │────▶│  Connectors  │────▶│   Trainer    │    │
│   └─────────┘     │              │     │              │    │
│                   │ • Files      │     │ • Dry-run    │    │
│                   │ • Database   │     │ • LoRA stub  │    │
│                   │ • Google Docs│     └──────┬───────┘    │
│                   └──────────────┘            │            │
│                                               ▼            │
│                                      ┌──────────────┐      │
│                                      │   Exporter   │      │
│                                      │              │      │
│                                      │ • FastAPI    │      │
│                                      │ • Docker     │      │
│                                      │ • ONNX       │      │
│                                      └──────┬───────┘      │
│                                              │             │
│                                              ▼             │
│   ┌─────────────┐                   ┌──────────────┐      │
│   │ Your Apps   │◀──────────────────│  Inference   │      │
│   └─────────────┘                   │   Server     │      │
│                                      └──────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### Multiple Data Connectors

TinyForgeAI ingests data from the sources you already use:

| Source | Status | Notes |
|--------|--------|-------|
| Local Files | ✅ | PDF, DOCX, TXT, JSON, JSONL |
| SQLite/PostgreSQL | ✅ | Streaming reader, batch mode |
| Google Docs | ✅ | Mock mode for testing |
| REST APIs | 🔜 | Coming soon |

### Production-Ready Export

Every exported service includes:
- FastAPI application with `/predict`, `/health`, `/readyz` endpoints
- Dockerfile optimized for production
- docker-compose.yml for easy deployment
- Kubernetes manifests and Helm charts

### Dry-Run Mode

Test your entire pipeline without training real models:

```bash
foremforge train --data data.jsonl --out ./model --dry-run
```

Dry-run mode:
- Validates your data format
- Creates stub model artifacts
- Tests the full export pipeline
- Verifies your deployment configuration

Perfect for CI/CD pipelines and rapid iteration.

## Real-World Use Cases

### Internal Support Bot

Train on your knowledge base, deploy behind your firewall:

```bash
foremforge train --data support_articles.jsonl --out ./support_model
foremforge export --model ./support_model/model_stub.json --out ./support_service
docker-compose up -d
```

### Document Q&A System

Turn your PDFs and Word documents into a queryable API:

```bash
foremforge ingest --input ./documents/ --output training_data.jsonl
foremforge train --data training_data.jsonl --out ./doc_model
```

### Workflow Automation

Embed inference directly in your automation pipelines:

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"input": "What's the status of order #12345?"}
)
print(response.json()["output"])
```

## Why Not Just Use GPT-4/Claude?

| Factor | Cloud LLMs | TinyForgeAI |
|--------|------------|-------------|
| Data Privacy | ❌ Data leaves your network | ✅ 100% on-premise |
| Cost | 💰 Per-request pricing | 💰 Fixed infrastructure |
| Latency | 🐌 100-2000ms | ⚡ <50ms |
| Customization | ❌ Generic responses | ✅ Your data, your model |
| Offline | ❌ Requires internet | ✅ Works anywhere |
| Compliance | ⚠️ May violate policies | ✅ Full control |

## Getting Started

### Installation

```bash
pip install tinyforgeai
```

Or from source:

```bash
git clone https://github.com/anthropics/TinyForgeAI.git
cd TinyForgeAI
pip install -e ".[dev]"
```

### Run the Demo

```bash
python examples/e2e_demo.py
```

This will:
1. Train a stub model on sample data
2. Export it as a FastAPI service
3. Test the `/predict` endpoint

### Next Steps

1. [Read the documentation](https://github.com/anthropics/TinyForgeAI/tree/main/docs)
2. [Try the connectors](https://github.com/anthropics/TinyForgeAI/blob/main/docs/connectors.md)
3. [Deploy to Kubernetes](https://github.com/anthropics/TinyForgeAI/tree/main/deploy)

## What's Next

TinyForgeAI v0.1.0 is the foundation. Here's what's coming:

- **Real Training Engine**: HuggingFace Transformers + LoRA fine-tuning
- **Model Zoo Expansion**: Pre-trained models for common tasks
- **Web Dashboard**: Visual training and deployment management
- **RAG Integration**: Retrieval-augmented generation for better accuracy
- **Multi-tenant Inference**: Serve multiple models from one service

## Contributing

TinyForgeAI is open source under Apache 2.0. We welcome contributions!

- [GitHub Repository](https://github.com/anthropics/TinyForgeAI)
- [Contributing Guide](https://github.com/anthropics/TinyForgeAI/blob/main/CONTRIBUTING.md)
- [Issue Tracker](https://github.com/anthropics/TinyForgeAI/issues)

---

**TinyForgeAI** — Because sometimes, smaller is smarter.

*Train tiny models. Deploy anywhere. Own your AI.*

---

## About

TinyForgeAI is developed by FOREM as an open-source project. We believe AI should be accessible, private, and deployable anywhere.

**Links:**
- GitHub: https://github.com/anthropics/TinyForgeAI
- Documentation: https://github.com/anthropics/TinyForgeAI/tree/main/docs
- License: Apache 2.0
