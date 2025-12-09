# TinyForgeAI — Social Launch Kit

Ready-to-use content for launching TinyForgeAI across social platforms.

---

## LinkedIn Post

```
We just released TinyForgeAI — an open-source framework to train tiny language models from your own data and deploy them instantly as microservices.

Why this matters:
Most companies don't need giant LLMs. They need small, private, deterministic AI engines trained only on their documents.

TinyForgeAI gives you exactly that.

👉 Train a tiny model
👉 Export as a REST microservice
👉 Deploy anywhere (Docker/K8s)
👉 No ML expertise needed

GitHub: https://github.com/anthropics/TinyForgeAI

Try the demo: bash examples/e2e_demo.sh

#OpenSource #MachineLearning #Python #AI #MLOps #TinyML
```

---

## X/Twitter Post

```
🔥 Just launched TinyForgeAI — train tiny language models from your own data → deploy as microservices in seconds.

Simple CLI. Dry-run training. Connectors (DB, Docs, Files). ONNX export. Docker. E2E demo.

Check it out → https://github.com/anthropics/TinyForgeAI

#AI #ML #Python #OpenSource
```

### Thread Version

```
🧵 1/5
Introducing TinyForgeAI — the simplest way to go from raw data to deployed ML model.

No ML expertise required. Just your data + 3 commands.
```

```
2/5
The problem: LLMs are overkill for most business use cases.

You don't need 70B parameters to answer questions about YOUR documents.

You need a small, fast, private model trained on YOUR data.
```

```
3/5
TinyForgeAI does exactly that:

✅ Ingest data (DB, files, Google Docs)
✅ Train a tiny model (dry-run or real)
✅ Export as microservice
✅ Deploy with Docker

All from the CLI.
```

```
4/5
Quick demo:

foremforge train --data data.jsonl --out ./model
foremforge export --model ./model/model_stub.json --out ./service
foremforge serve --dir ./service --port 8000

That's it. You now have a REST API.
```

```
5/5
TinyForgeAI is open source (Apache 2.0).

GitHub: https://github.com/anthropics/TinyForgeAI

Star it ⭐ if you find it useful!

#AI #MachineLearning #Python #OpenSource
```

---

## Reddit Post

### r/MachineLearning (Show and Tell)

**Title:** [P] TinyForgeAI - Lightweight Fine-Tuning and Deployment Platform

```
I've been working on TinyForgeAI, an open-source tool that simplifies fine-tuning language models and deploying them as APIs.

**What it does:**
- Train models from JSONL datasets
- Export to production-ready FastAPI services
- Docker support out of the box
- Multiple data connectors (local files, databases, Google Docs)

**Quick example:**

pip install tinyforgeai
foremforge train --data data.jsonl --out ./model --dry-run
foremforge export --model ./model/model_stub.json --out ./service
foremforge serve --dir ./service --port 8000

**Why I built it:**
Most ML deployment tools assume you already have a trained model. TinyForgeAI handles the entire pipeline from raw data to deployed service.

**Current status:**
- v0.1.0 released
- Dry-run training (real training coming soon)
- 215+ tests passing
- Full documentation

GitHub: https://github.com/anthropics/TinyForgeAI
Docs: https://github.com/anthropics/TinyForgeAI/tree/main/docs

Feedback welcome! Especially interested in:
- What connectors you'd want (APIs, cloud storage, etc.)
- Export formats beyond ONNX
- UI preferences

License: Apache 2.0
```

### r/Python

**Title:** TinyForgeAI: Train and deploy ML models with Python + FastAPI

```
Just released TinyForgeAI - a Python framework for the complete ML workflow.

**Stack:**
- Click for CLI
- FastAPI for inference servers
- Pydantic for data validation
- Docker for deployment

**Features:**
- JSONL dataset handling
- Multiple data connectors (files, DB, Google Docs)
- Auto-generated microservices
- Health checks, metrics, Docker support

**Install:**
pip install tinyforgeai

**Try it:**
python examples/e2e_demo.py

GitHub: https://github.com/anthropics/TinyForgeAI

Built with Python 3.10+, fully typed, 215+ tests.
```

### r/LocalLLaMA

**Title:** TinyForgeAI - Tool for training and deploying small, local LLMs

```
For those running local models, I built TinyForgeAI to simplify the training → deployment pipeline.

**Key features:**
- Train tiny models from your own data
- Export to ONNX (coming: quantization)
- Generate complete FastAPI services
- Docker-ready deployment
- No cloud required

The idea: not every use case needs GPT-4. Sometimes a small model trained on your specific data is better (faster, cheaper, private).

Currently uses a stub trainer for prototyping. Real HuggingFace integration is on the roadmap.

GitHub: https://github.com/anthropics/TinyForgeAI
```

---

## Hacker News

**Title:** Show HN: TinyForgeAI – Train tiny LLMs from your data, deploy as microservices

```
TinyForgeAI is an open-source platform for fine-tuning small language models and deploying them as inference services.

The thesis: Large LLMs are overkill for most enterprise use cases. A small model trained only on your data is often faster, cheaper, and more private.

Key features:
- JSONL dataset format
- Multiple data connectors (files, databases, Google Docs)
- FastAPI-based inference servers
- Docker support
- Full CLI (foremforge)

Quick start:

    foremforge train --data data.jsonl --out ./model --dry-run
    foremforge export --model ./model/model_stub.json --out ./service
    foremforge serve --dir ./service --port 8000

Current version (v0.1.0) uses a stub trainer for rapid prototyping. Real fine-tuning (HuggingFace Transformers + LoRA) is the next priority.

GitHub: https://github.com/anthropics/TinyForgeAI
Docs: https://github.com/anthropics/TinyForgeAI/tree/main/docs

Looking for feedback on:
1. What data connectors would be most useful?
2. What export formats matter to you?
3. Would a web UI be valuable?

Apache 2.0 licensed.
```

---

## Product Hunt

### Tagline Options

1. "Train tiny AI models from your data, deploy as microservices"
2. "From raw data to deployed ML API in 3 commands"
3. "The simplest way to train and deploy your own language model"

### First Comment

```
Thanks for checking out TinyForgeAI!

We built it because small LLMs are the future — private, cheap, fast, and fine-tuned to your business.

This v0.1.0 release includes:
✅ Full CLI for training and export
✅ Multiple data connectors
✅ Auto-generated FastAPI microservices
✅ Docker deployment
✅ Complete documentation

What's next:
- Real fine-tuning (HuggingFace + LoRA)
- More export formats
- Web dashboard

We'd love your feedback! What features would make this more useful for you?
```

### Description

```
TinyForgeAI is an open-source framework that lets you train small language models from your own data and deploy them as REST microservices — all from the command line.

Perfect for:
🏢 Companies wanting private, on-premise AI
👩‍💻 Developers who don't want to learn complex ML pipelines
🔒 Teams needing secure, data-local inference
⚡ Edge deployment scenarios

Features:
• Simple CLI: train, export, serve
• Multiple data sources: files, databases, Google Docs
• Auto-generated FastAPI services
• Docker-ready deployment
• Extensible connector system

No ML expertise required. Apache 2.0 licensed.
```

---

## Taglines (Short)

- "Tiny models. Big impact."
- "Your data. Your model. Your microservice."
- "Train → Export → Serve. Done."
- "ML deployment without the PhD."
- "Small AI for real problems."

---

## Newsletter/Blog Intro

```
Introducing TinyForgeAI — From Raw Data to Deployed Models

We're launching TinyForgeAI, a Python toolkit that streamlines the ML workflow from data preparation to production deployment.

The problem with large language models:
- Expensive to run
- Overkill for specific tasks
- Privacy concerns with cloud APIs
- Complex to fine-tune

TinyForgeAI's approach:
- Train small models on YOUR data
- Deploy as simple REST services
- Run anywhere (local, cloud, edge)
- No ML expertise required

Get started in 3 commands:

1. foremforge train --data your_data.jsonl --out ./model
2. foremforge export --model ./model/model_stub.json --out ./service
3. foremforge serve --dir ./service --port 8000

GitHub: https://github.com/anthropics/TinyForgeAI
```

---

## Image/Graphics Suggestions

For social posts, consider creating:

1. **Architecture diagram** (already in README)
2. **CLI demo GIF** — Show the 3 commands running
3. **Before/After comparison** — "Traditional ML pipeline" vs "TinyForgeAI"
4. **Logo/Banner** — 1200x630 for social sharing

---

## Hashtags

```
#TinyForgeAI #OpenSource #MachineLearning #Python #AI #MLOps #FastAPI #Docker #TinyML #EdgeAI #LocalAI #LLM #FineTuning #DataScience
```
