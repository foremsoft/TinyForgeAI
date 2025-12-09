# Introduction to AI Model Training

Welcome to TinyForgeAI! This tutorial series will teach you everything you need to know about training AI models, even if you're a complete beginner.

## Table of Contents

1. [What is AI Model Training?](#what-is-ai-model-training)
2. [Key Concepts Explained Simply](#key-concepts-explained-simply)
3. [How TinyForgeAI Makes It Easy](#how-tinyforgeai-makes-it-easy)
4. [Your First Steps](#your-first-steps)

---

## What is AI Model Training?

Imagine teaching a child to recognize animals. You show them pictures: "This is a cat," "This is a dog," and over time, they learn to identify animals on their own.

**AI model training works similarly:**

1. **You provide examples** (called "training data")
2. **The model learns patterns** from those examples
3. **The model can then make predictions** on new data it hasn't seen before

### Real-World Analogy

```
Human Learning:
  See 1000 cat photos → Brain learns "cat patterns" → Can recognize new cats

AI Learning:
  Process 1000 cat photos → Model learns "cat patterns" → Can recognize new cats
```

### What is "Fine-Tuning"?

Instead of training a model from scratch (which requires millions of examples), we can **fine-tune** an existing model. This is like teaching someone who already knows English to learn medical terminology - they don't need to relearn the entire language!

```
Pre-trained Model (knows general language)
        ↓
   Fine-tuning (your specific data)
        ↓
Specialized Model (knows your domain)
```

---

## Key Concepts Explained Simply

### 1. Language Models (LLMs)

**What they are:** Programs that understand and generate human-like text.

**Examples:** GPT, BERT, Llama, Mistral

**How they work:**
```
Input:  "The weather today is"
Output: "sunny and warm" (model predicts what comes next)
```

### 2. Datasets

**What they are:** Collections of examples used to train your model.

**Format for TinyForgeAI:**
```json
{"input": "What is Python?", "output": "Python is a programming language."}
{"input": "Explain AI simply", "output": "AI is software that can learn and make decisions."}
```

### 3. Training Parameters

| Parameter | Simple Explanation | Analogy |
|-----------|-------------------|---------|
| **Epochs** | How many times to review all data | Reading a textbook multiple times |
| **Batch Size** | How many examples to learn at once | Studying 10 flashcards at a time |
| **Learning Rate** | How big steps to take when learning | Walking vs running while learning |

### 4. LoRA (Low-Rank Adaptation)

**The Problem:** Training large models requires huge amounts of computer memory.

**The Solution:** LoRA trains only a small "adapter" instead of the whole model.

```
Traditional Training:        LoRA Training:
Update 7 billion parameters  Update only 1 million parameters
Needs 80GB GPU memory        Needs only 8GB GPU memory
Takes days                   Takes hours
```

### 5. ONNX Export

**What it is:** Converting your trained model to a universal format.

**Why it matters:** Like saving a document as PDF - it works everywhere!

```
PyTorch Model → ONNX → Works on: Windows, Linux, Mac, Mobile, Web
```

---

## How TinyForgeAI Makes It Easy

### Traditional Way (Complex)
```python
# You'd need to write 200+ lines of code for:
# - Loading datasets
# - Configuring tokenizers
# - Setting up training loops
# - Managing GPU memory
# - Saving checkpoints
# - Handling errors
```

### TinyForgeAI Way (Simple)
```python
from backend.training.real_trainer import RealTrainer, TrainingConfig

# Just 5 lines to train a model!
config = TrainingConfig(
    model_name="distilbert-base-uncased",
    output_dir="./my-model",
    num_epochs=3
)
trainer = RealTrainer(config)
trainer.train("my_data.jsonl")
```

---

## Your First Steps

### Step 1: Install TinyForgeAI

```bash
# Clone the repository
git clone https://github.com/anthropics/TinyForgeAI.git
cd TinyForgeAI

# Install basic dependencies
pip install -e .

# Install training dependencies (optional, for GPU training)
pip install -e ".[training]"
```

### Step 2: Prepare Your Data

Create a file called `my_data.jsonl`:
```json
{"input": "What is TinyForgeAI?", "output": "TinyForgeAI is a platform for fine-tuning AI models."}
{"input": "How do I train a model?", "output": "Use the RealTrainer class with your dataset."}
{"input": "What is fine-tuning?", "output": "Fine-tuning adapts a pre-trained model to your specific task."}
```

### Step 3: Train Your First Model

```python
from backend.training.real_trainer import RealTrainer, TrainingConfig

# Configure training
config = TrainingConfig(
    model_name="distilbert-base-uncased",  # Small, fast model for learning
    output_dir="./my-first-model",
    num_epochs=3,
    batch_size=8,
    learning_rate=2e-5
)

# Create trainer and start training
trainer = RealTrainer(config)
trainer.train("my_data.jsonl")

print("Congratulations! You've trained your first AI model!")
```

### Step 4: Test Your Model

```python
# Your model is saved in ./my-first-model
# Use it for predictions through the inference server
```

---

## What's Next?

Continue to the next tutorials:

1. **[02-preparing-training-data.md](02-preparing-training-data.md)** - Learn how to create effective training datasets
2. **[03-training-your-first-model.md](03-training-your-first-model.md)** - Detailed walkthrough of the training process
3. **[04-understanding-lora.md](04-understanding-lora.md)** - Deep dive into efficient training with LoRA
4. **[05-deploying-your-model.md](05-deploying-your-model.md)** - Serve your model as an API

---

## Glossary

| Term | Definition |
|------|------------|
| **Epoch** | One complete pass through all training data |
| **Batch** | A subset of data processed together |
| **Fine-tuning** | Adapting a pre-trained model to a specific task |
| **Inference** | Using a trained model to make predictions |
| **LoRA** | Efficient training technique that updates fewer parameters |
| **ONNX** | Universal model format for deployment |
| **Tokenizer** | Converts text to numbers the model understands |
| **Checkpoint** | A saved snapshot of the model during training |

---

## Need Help?

- **GitHub Issues:** [Report bugs or ask questions](https://github.com/anthropics/TinyForgeAI/issues)
- **Discussions:** Join our community discussions
- **Examples:** Check the `examples/` folder for working code

Happy Learning! 🚀
