# Hands-On Tutorials

Build real AI-powered applications in about 90 minutes. Project-based learning with working code.

---

## Tutorial Overview

| Tutorial | Time | What You Build |
|----------|------|----------------|
| [Quick Start](../tutorials/hands-on/00-quickstart.md) | 5 min | Install & first script |
| [FAQ Bot](../tutorials/hands-on/01-faq-bot.md) | 15 min | Working chatbot |
| [Document Search](../tutorials/hands-on/02-document-search.md) | 20 min | Search PDFs & docs |
| [Train Model](../tutorials/hands-on/03-train-your-model.md) | 30 min | Custom AI model |
| [Deploy](../tutorials/hands-on/04-deploy-your-project.md) | 20 min | Put it online |

**Total Time:** ~90 minutes

---

## Prerequisites

- Python 3.10+
- Basic command line knowledge
- A text editor

**No AI experience needed!**

---

## Tutorial 0: Quick Start (5 min)

Get TinyForgeAI running on your computer.

```bash
# Clone and install
git clone https://github.com/foremsoft/TinyForgeAI.git
cd TinyForgeAI
pip install -e .

# Test it works
python -c "print('Ready!')"
```

---

## Tutorial 1: FAQ Bot (15 min)

Build a chatbot that answers questions from your data.

### What You Learn
- Load and process text data
- Build a working chatbot
- Create a REST API

### Key Code

```python
from difflib import SequenceMatcher

def find_answer(question, faq_data):
    best_match = max(
        faq_data,
        key=lambda x: SequenceMatcher(None, question.lower(), x['question'].lower()).ratio()
    )
    return best_match['answer']
```

---

## Tutorial 2: Document Search (20 min)

Search through PDFs, Word docs, and text files.

### What You Learn
- Ingest different file types
- Create a searchable index
- Build keyword and AI-powered search

### Key Concept

```python
from connectors.file_ingest import ingest_file

# Ingest any supported file
content = ingest_file("manual.pdf")

# Search by meaning (semantic search)
from connectors.indexer import DocumentIndexer
indexer = DocumentIndexer()
indexer.add_document(content)
results = indexer.search("How do I reset my password?")
```

---

## Tutorial 3: Train Model (30 min)

Train your own AI model on custom data.

### What You Learn
- Prepare training data
- Train an AI model
- Test and evaluate

### Key Code

```python
from backend.training.real_trainer import RealTrainer, TrainingConfig

config = TrainingConfig(
    model_name="distilbert-base-uncased",
    output_dir="./my_model",
    num_epochs=3,
    use_lora=True  # Efficient fine-tuning
)

trainer = RealTrainer(config)
trainer.train("my_data.jsonl")
```

### Or Use the GUI

```bash
# Gradio (easiest)
cd ui/gradio && python training_app.py

# Streamlit (data exploration)
cd ui/streamlit && streamlit run training_app.py

# React Dashboard (production)
cd dashboard && npm run dev
```

---

## Tutorial 4: Deploy (20 min)

Put your AI online for others to use.

### What You Learn
- Package your application
- Deploy to the cloud
- Create a web interface

### Deployment Options

| Platform | Command |
|----------|---------|
| **Docker** | `docker build -t my-ai . && docker run -p 8000:8000 my-ai` |
| **Railway** | Connect GitHub repo, auto-deploy |
| **Render** | Similar to Railway |
| **AWS/GCP/Azure** | Use IaC templates in `deploy/` |

---

## Training UIs Available

Prefer buttons over commands? Use our graphical interfaces:

| Interface | Best For | Command |
|-----------|----------|---------|
| **Gradio** | Demos, beginners | `cd ui/gradio && python training_app.py` |
| **Streamlit** | Data exploration | `cd ui/streamlit && streamlit run training_app.py` |
| **React Dashboard** | Production | `cd dashboard && npm run dev` |

---

## Sample Data

Practice with included datasets:

```
examples/tutorial_data/
├── sample_faqs.csv           # 15 FAQ pairs
├── sample_training_data.jsonl # 25 training examples
└── README.md
```

---

## Tips for Success

1. **Type code yourself** - Helps retention
2. **Run each example** - Verify it works
3. **Experiment** - Change values, see what happens
4. **Take breaks** - Fresh eyes help

---

## What's Next?

After completing these tutorials:

1. Try the [Beginner's Course](Beginners-Course) for deeper understanding
2. Explore the [Model Zoo](../README.md#model-zoo) for pre-configured models
3. Read the [Use Cases Guide](../use_cases.md) for enterprise examples
4. Deploy to production with [Cloud Deployment](Cloud-Deployment)

---

## Getting Help

- [GitHub Issues](https://github.com/foremsoft/TinyForgeAI/issues) - Report bugs
- [GitHub Discussions](https://github.com/foremsoft/TinyForgeAI/discussions) - Ask questions
