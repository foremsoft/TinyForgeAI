# TinyForgeAI Wiki

Welcome to the TinyForgeAI Wiki! This is your central resource for learning how to train, deploy, and manage tiny language models.

---

## Quick Links

| Resource | Description |
|----------|-------------|
| [Getting Started](#getting-started) | Installation and first steps |
| [Training UIs](Training-UIs) | Graphical interfaces for training |
| [Beginner's Course](Beginners-Course) | Learn AI from scratch (4 hours) |
| [Hands-On Tutorials](Hands-On-Tutorials) | Build real projects (90 min) |
| [Connectors](Connectors) | Connect to data sources |
| [Deployment](Deployment) | Deploy to production |

---

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/foremsoft/TinyForgeAI.git
cd TinyForgeAI

# Install with all features
pip install -e ".[all]"
```

### Your First Model (5 minutes)

```bash
# Train a model (dry run - no GPU needed)
foremforge train --data examples/data/demo_dataset.jsonl --out ./my_model --dry-run

# Export for deployment
foremforge export --model ./my_model/model_stub.json --out ./my_service

# Test the service
foremforge serve --dir ./my_service --dry-run
```

### Using the Training UIs

Prefer a graphical interface? Choose one:

| Interface | Command | Best For |
|-----------|---------|----------|
| **Gradio** | `cd ui/gradio && python training_app.py` | Demos, beginners |
| **Streamlit** | `cd ui/streamlit && streamlit run training_app.py` | Data scientists |
| **React Dashboard** | `cd dashboard && npm run dev` | Production |

---

## What's New (December 2025)

### Training UIs for Everyone
- **Gradio Interface**: Simple drag-and-drop training with shareable links
- **Streamlit Interface**: Rich data exploration and visualization
- **React Dashboard Easy Mode**: Step-by-step wizard for non-technical users

### Beginner's AI Course
- 11 modules covering AI fundamentals to deployment
- No prior experience required
- ~4 hours to complete

### Enhanced Tutorials
- Hands-on tutorials: Build real projects in 90 minutes
- Sample datasets included for practice

### New Connectors
- Google Docs integration
- Slack connector
- Confluence connector

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              TinyForgeAI                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────┐     ┌────────────────┐     ┌──────────────────┐          │
│   │ CLI/UI  │────▶│ Data Connectors│────▶│ Trainer          │          │
│   └─────────┘     │                │     │ (LoRA / Full)    │          │
│                   │ • Files        │     └────────┬─────────┘          │
│                   │ • Database     │              │                    │
│                   │ • Google Docs  │              ▼                    │
│                   │ • Slack        │     ┌──────────────────┐          │
│                   │ • Notion       │     │ Model Registry   │          │
│                   └────────────────┘     └────────┬─────────┘          │
│                                                   │                    │
│                                                   ▼                    │
│   ┌─────────────┐     ┌──────────────────────────────────────┐        │
│   │ Client Apps │◀────│ Inference Server (FastAPI)           │        │
│   └─────────────┘     └──────────────────────────────────────┘        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Popular Topics

- [Training Your First Model](Training-Your-First-Model)
- [Data Formats (CSV, JSONL)](Data-Formats)
- [Using LoRA for Efficient Training](Using-LoRA)
- [Deploying with Docker](Docker-Deployment)
- [Cloud Deployment (AWS/GCP/Azure)](Cloud-Deployment)

---

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/foremsoft/TinyForgeAI/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/foremsoft/TinyForgeAI/discussions)
- **Documentation**: [Full docs](https://github.com/foremsoft/TinyForgeAI/tree/main/docs)

---

## Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/foremsoft/TinyForgeAI/blob/main/docs/CONTRIBUTING.md).
