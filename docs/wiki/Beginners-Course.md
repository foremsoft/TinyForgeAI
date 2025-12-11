# Beginner's AI Course

Learn AI from absolute zero to deploying your own model. No prior experience needed!

---

## Course Overview

| Stat | Value |
|------|-------|
| **Duration** | ~4 hours |
| **Modules** | 11 |
| **Prerequisites** | Basic computer skills |
| **Outcome** | Deploy your own AI model |

---

## Who Is This For?

- Complete beginners who have never touched AI
- Developers from other fields (web, mobile, backend)
- Students curious about how AI works
- Business professionals who want to build AI solutions
- Hobbyists and makers

---

## What You'll Build

1. A working FAQ chatbot
2. A document search system
3. Your own trained AI model
4. A deployed web API

---

## Course Modules

| Module | Title | Time | Description |
|--------|-------|------|-------------|
| [00](../tutorials/beginners-course/00-what-is-ai.md) | What is AI? | 15 min | AI concepts (no code!) |
| [01](../tutorials/beginners-course/01-setup-your-computer.md) | Setup Your Computer | 20 min | Install Python, VS Code |
| [02](../tutorials/beginners-course/02-your-first-ai-script.md) | Your First AI Script | 15 min | 10 lines of AI code |
| [03](../tutorials/beginners-course/03-understanding-data.md) | Understanding Data | 20 min | CSV, JSON, JSONL |
| [04](../tutorials/beginners-course/04-build-a-simple-bot.md) | Build a Simple Bot | 25 min | FAQ chatbot + REST API |
| [05](../tutorials/beginners-course/05-what-is-a-model.md) | What is a Model? | 20 min | Weights, training basics |
| [06](../tutorials/beginners-course/06-prepare-training-data.md) | Prepare Training Data | 25 min | Load from various sources |
| [07](../tutorials/beginners-course/07-train-your-model.md) | Train Your First Model | 45 min | Train with DistilBERT |
| [08](../tutorials/beginners-course/08-test-and-improve.md) | Test & Improve | 20 min | Evaluate and improve |
| [09](../tutorials/beginners-course/09-deploy-and-share.md) | Deploy & Share | 25 min | Put it online |
| [10](../tutorials/beginners-course/10-next-steps.md) | Next Steps | 15 min | Advanced topics |

---

## Learning Path

```
Start Here!
    │
    ▼
┌──────────────────────────────────────┐
│  Module 0: What is AI?               │
│  └── Understand concepts (no code)   │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Module 1-2: Setup & First Script    │
│  └── Environment ready, first code   │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Module 3-4: Data & Simple Bot       │
│  └── Learn formats, build chatbot    │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Module 5-7: Models & Training       │
│  └── Understand AI, train your own   │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Module 8-9: Test & Deploy           │
│  └── Improve accuracy, go live       │
└──────────────────────────────────────┘
    │
    ▼
🎓 You're an AI Builder!
```

---

## Quick Start

```bash
# 1. Get TinyForgeAI
git clone https://github.com/foremsoft/TinyForgeAI.git
cd TinyForgeAI

# 2. Install
pip install -e .

# 3. Start learning
# Open docs/tutorials/beginners-course/00-what-is-ai.md
```

---

## Prefer a GUI?

Use our [Training UIs](Training-UIs) if you prefer clicking buttons:

| Interface | Command |
|-----------|---------|
| **Gradio** | `cd ui/gradio && python training_app.py` |
| **Streamlit** | `cd ui/streamlit && streamlit run training_app.py` |
| **Dashboard** | `cd dashboard && npm run dev` |

The React Dashboard has an **Easy Mode** wizard!

---

## Sample Data

Practice with included sample data:

```
examples/tutorial_data/
├── sample_faqs.csv           # 15 FAQ pairs
├── sample_training_data.jsonl # 25 training examples
└── README.md
```

---

## Tips for Success

1. **Type the code yourself** - Don't just copy-paste
2. **Run every example** - See code work before moving on
3. **Experiment** - Change values, break things
4. **Take notes** - Write concepts in your own words
5. **Don't rush** - Understanding > speed

---

## Getting Help

1. Check the **Troubleshooting** section in each module
2. Search [GitHub Issues](https://github.com/foremsoft/TinyForgeAI/issues)
3. Ask in [GitHub Discussions](https://github.com/foremsoft/TinyForgeAI/discussions)

---

## See Also

- [Training UIs](Training-UIs) - Graphical training interfaces
- [Hands-On Tutorials](Hands-On-Tutorials) - Shorter project-based tutorials
- [Data Formats](Data-Formats) - CSV and JSONL specifications
