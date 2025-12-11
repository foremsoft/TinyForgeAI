# TinyForgeAI Hands-On Tutorials

Welcome! These tutorials will teach you how to build AI-powered applications from scratch, even if you've never done it before.

---

## Who Are These Tutorials For?

- **Complete beginners** who want to learn AI development
- **Developers** who want to add AI features to their applications
- **Business users** who want to create custom chatbots and search systems

**No AI experience required!** We explain everything step by step.

---

## Tutorial Path

Follow these tutorials in order for the best learning experience:

| # | Tutorial | Time | What You'll Build |
|---|----------|------|-------------------|
| 0 | [Quick Start](00-quickstart.md) | 5 min | Install TinyForgeAI & run your first script |
| 1 | [FAQ Bot](01-faq-bot.md) | 15 min | A chatbot that answers questions |
| 2 | [Document Search](02-document-search.md) | 20 min | Search through PDFs, Word docs & more |
| 3 | [Train Your Model](03-train-your-model.md) | 30 min | Train a custom AI model |
| 4 | [Deploy Your Project](04-deploy-your-project.md) | 20 min | Put your AI online for others to use |
| 5 | [MCP Integration](05-mcp-integration.md) | 15 min | Use AI assistants to control TinyForgeAI |
| 6 | [Killer Features](06-killer-features.md) | 25 min | URL Training, Data Augmentation, Playground |

**Total time:** About 130 minutes to complete all tutorials.

---

## What You'll Learn

### By the End of Tutorial 1 (FAQ Bot):
- Load and process text data
- Build a working chatbot
- Create a REST API

### By the End of Tutorial 2 (Document Search):
- Ingest PDFs, Word docs, and text files
- Create a searchable index
- Build keyword and AI-powered search

### By the End of Tutorial 3 (Training):
- Prepare training data
- Train an AI model
- Test and evaluate your model

### By the End of Tutorial 4 (Deployment):
- Package your application
- Deploy to the cloud
- Create a web interface

### By the End of Tutorial 5 (MCP Integration):
- Connect TinyForgeAI to Claude Desktop
- Train models using natural language
- Search documents with AI assistance

### By the End of Tutorial 6 (Killer Features):
- Train models directly from URLs (no data prep)
- Generate 500+ training samples from 5 examples
- Create shareable model playgrounds

---

## Prerequisites

- **Python 3.10+** installed ([download here](https://python.org))
- Basic command line knowledge (how to open a terminal and run commands)
- A text editor (VS Code, Notepad++, or any editor you like)

**That's it!** No prior AI or machine learning experience needed.

---

## Quick Start

```bash
# 1. Get TinyForgeAI
git clone https://github.com/foremsoft/TinyForgeAI.git
cd TinyForgeAI

# 2. Install it
pip install -e .

# 3. Test it works
python -c "print('Ready to learn AI!')"

# 4. Start the first tutorial
# Open docs/tutorials/hands-on/00-quickstart.md
```

---

## Sample Projects Included

Each tutorial includes complete, working code that you can copy and run:

```
docs/tutorials/hands-on/
├── 00-quickstart.md        # Installation & first steps
├── 01-faq-bot.md           # Build a FAQ chatbot
├── 02-document-search.md   # Search your documents
├── 03-train-your-model.md  # Train custom AI
├── 04-deploy-your-project.md # Go to production
├── 05-mcp-integration.md   # AI assistant integration
└── 06-killer-features.md   # URL training, augmentation, playground
```

---

## Training UIs Available

Don't want to use the command line? We have graphical interfaces for training:

| Interface | Best For | How to Run |
|-----------|----------|------------|
| **Gradio** | Demos, beginners | `cd ui/gradio && python training_app.py` |
| **Streamlit** | Data exploration | `cd ui/streamlit && streamlit run training_app.py` |
| **React Dashboard** | Production | `cd dashboard && npm run dev` |

See the [Training UIs Guide](../../../ui/README.md) for details.

---

## Getting Help

Stuck? Here's how to get help:

1. **Check the Troubleshooting section** at the end of each tutorial
2. **Search existing issues** on [GitHub Issues](https://github.com/foremsoft/TinyForgeAI/issues)
3. **Open a new issue** if you can't find a solution

When asking for help, include:
- Which tutorial you're on
- The exact error message
- Your Python version (`python --version`)
- Your operating system (Windows, Mac, Linux)

---

## Tips for Success

1. **Type the code yourself** instead of copy-pasting (helps you learn)
2. **Run each example** before moving to the next step
3. **Experiment!** Change values and see what happens
4. **Take breaks** if you get stuck - fresh eyes help

---

## Ready?

Start with [00-quickstart.md](00-quickstart.md) and you'll be building AI applications in no time!

Happy learning!
