# TinyForgeAI Quick Start Guide

**Time needed:** 5 minutes
**Skill level:** Complete beginner
**What you'll do:** Install TinyForgeAI and run your first AI example

---

## What is TinyForgeAI?

TinyForgeAI helps you create your own AI assistant that knows YOUR data. Think of it like this:

```
Your Company Documents  →  TinyForgeAI  →  AI That Answers Questions About YOUR Business
     (PDFs, FAQs, etc.)      (Training)       (Custom ChatGPT-like assistant)
```

---

## Step 1: Install TinyForgeAI

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

```bash
# Download TinyForgeAI
git clone https://github.com/foremsoft/TinyForgeAI.git

# Go into the folder
cd TinyForgeAI

# Install it
pip install -e .
```

**Don't have Python?** Download it from [python.org](https://python.org) first (version 3.10 or newer).

---

## Step 2: Test the Installation

Run this command to check everything works:

```bash
python -c "import connectors; print('TinyForgeAI installed successfully!')"
```

You should see: `TinyForgeAI installed successfully!`

---

## Step 3: Load Some Data

Let's load a sample document. Create a file called `test.py`:

```python
# test.py - Your first TinyForgeAI script

from connectors.file_ingest import ingest_file

# Create a sample text file to work with
sample_text = """
TinyForgeAI FAQ

Q: What is TinyForgeAI?
A: TinyForgeAI is a tool for creating custom AI assistants.

Q: Do I need a powerful computer?
A: No! TinyForgeAI is designed to work on regular laptops.

Q: What data formats are supported?
A: TXT, PDF, DOCX, Markdown, and more.
"""

# Save the sample text
with open("sample_faq.txt", "w") as f:
    f.write(sample_text)

# Now load it with TinyForgeAI
text = ingest_file("sample_faq.txt")
print("Loaded document!")
print("-" * 40)
print(text)
```

Run it:

```bash
python test.py
```

**Expected output:**
```
Loaded document!
----------------------------------------
TinyForgeAI FAQ

Q: What is TinyForgeAI?
A: TinyForgeAI is a tool for creating custom AI assistants.
...
```

---

## Step 4: Create Training Data

AI learns from examples. Let's create some question-answer pairs:

```python
# create_training_data.py

import json

# Training data = examples of questions and answers
training_data = [
    {
        "input": "What is TinyForgeAI?",
        "output": "TinyForgeAI is a tool for creating custom AI assistants trained on your own data."
    },
    {
        "input": "Do I need an expensive computer?",
        "output": "No, TinyForgeAI works on regular laptops and computers."
    },
    {
        "input": "What file types can I use?",
        "output": "TinyForgeAI supports TXT, PDF, DOCX, Markdown, CSV, and data from APIs."
    },
    {
        "input": "How long does training take?",
        "output": "A simple model can be trained in minutes. Larger models may take hours."
    },
    {
        "input": "Is TinyForgeAI free?",
        "output": "Yes, TinyForgeAI is open source and free to use."
    }
]

# Save as JSONL (one JSON object per line)
with open("my_training_data.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")

print(f"Created training data with {len(training_data)} examples!")
print("File saved: my_training_data.jsonl")
```

Run it:

```bash
python create_training_data.py
```

---

## Step 5: View Your Training Data

```python
# view_data.py

import json

print("Your Training Data:")
print("=" * 60)

with open("my_training_data.jsonl", "r") as f:
    for i, line in enumerate(f, 1):
        item = json.loads(line)
        print(f"\nExample {i}:")
        print(f"  Question: {item['input']}")
        print(f"  Answer:   {item['output'][:50]}...")

print("\n" + "=" * 60)
print("This data will teach your AI how to answer questions!")
```

---

## What's Next?

You've completed the basics! Continue with:

| Tutorial | What You'll Build | Time |
|----------|------------------|------|
| [01-faq-bot.md](01-faq-bot.md) | A FAQ chatbot for your business | 15 min |
| [02-document-search.md](02-document-search.md) | Search through your documents | 20 min |
| [03-train-your-model.md](03-train-your-model.md) | Train a real AI model | 30 min |

---

## Troubleshooting

### "pip not found"
```bash
# Try this instead:
python -m pip install -e .
```

### "git not found"
Download Git from [git-scm.com](https://git-scm.com) or download TinyForgeAI as a ZIP file from GitHub.

### "Permission denied"
```bash
# On Mac/Linux, try:
sudo pip install -e .

# On Windows, run Command Prompt as Administrator
```

### Still stuck?
- Open an issue: [github.com/foremsoft/TinyForgeAI/issues](https://github.com/foremsoft/TinyForgeAI/issues)
- Include the error message you see

---

**Congratulations!** You've taken your first step into AI development!
