# Training Your First Model

This hands-on tutorial walks you through training an AI model from start to finish.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Understanding the Training Process](#understanding-the-training-process)
3. [Step-by-Step Training Guide](#step-by-step-training-guide)
4. [Monitoring Training Progress](#monitoring-training-progress)
5. [Troubleshooting Common Issues](#troubleshooting-common-issues)

---

## Prerequisites

Before starting, ensure you have:

### Hardware Requirements

| Training Type | Minimum | Recommended |
|--------------|---------|-------------|
| CPU Training | 8GB RAM | 16GB RAM |
| GPU Training | 4GB VRAM | 8GB+ VRAM |
| LoRA Training | 4GB VRAM | 6GB+ VRAM |

### Software Requirements

```bash
# Install TinyForgeAI with training dependencies
pip install -e ".[training]"

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Check GPU Availability

```python
import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("No GPU found. Training will use CPU (slower).")
```

---

## Understanding the Training Process

### What Happens During Training?

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. LOAD DATA                                                   │
│     ┌──────────┐                                                │
│     │ dataset  │ → Tokenize → Convert to tensors                │
│     └──────────┘                                                │
│           ↓                                                     │
│  2. LOAD MODEL                                                  │
│     ┌──────────┐                                                │
│     │ pretrained│ → Load weights → Prepare for training         │
│     │  model    │                                               │
│     └──────────┘                                                │
│           ↓                                                     │
│  3. TRAINING LOOP (for each epoch)                              │
│     ┌────────────────────────────────────────────┐              │
│     │ For each batch:                            │              │
│     │   a. Forward pass (make predictions)       │              │
│     │   b. Calculate loss (how wrong it was)     │              │
│     │   c. Backward pass (calculate gradients)   │              │
│     │   d. Update weights (improve model)        │              │
│     └────────────────────────────────────────────┘              │
│           ↓                                                     │
│  4. SAVE MODEL                                                  │
│     ┌──────────┐                                                │
│     │ trained  │ → Save weights → Ready for use!                │
│     │  model   │                                                │
│     └──────────┘                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Training Metrics

| Metric | What It Means | Good Values |
|--------|--------------|-------------|
| **Loss** | How wrong the model is | Lower is better (aim for < 1.0) |
| **Accuracy** | % of correct predictions | Higher is better (aim for > 80%) |
| **Learning Rate** | Speed of learning | Usually 1e-5 to 5e-5 |
| **Epoch** | Full pass through data | 1-10 typically |

---

## Step-by-Step Training Guide

### Step 1: Prepare Your Environment

```python
# training_example.py

import os
import sys

# Add project to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.training.real_trainer import RealTrainer, TrainingConfig

print("Environment ready!")
```

### Step 2: Create Your Dataset

Save this as `examples/training/sample_data.jsonl`:

```jsonl
{"input": "What is machine learning?", "output": "Machine learning is a type of artificial intelligence that allows computers to learn from data without being explicitly programmed."}
{"input": "How does deep learning work?", "output": "Deep learning uses neural networks with many layers to learn complex patterns in data, similar to how the human brain processes information."}
{"input": "What is a neural network?", "output": "A neural network is a computing system inspired by the human brain, consisting of interconnected nodes that process information in layers."}
{"input": "Explain supervised learning", "output": "Supervised learning is a type of machine learning where the model learns from labeled examples, like learning to recognize cats from photos labeled 'cat'."}
{"input": "What is unsupervised learning?", "output": "Unsupervised learning finds patterns in data without labeled examples, like grouping similar customers together without knowing the groups beforehand."}
{"input": "Define reinforcement learning", "output": "Reinforcement learning trains agents to make decisions by rewarding good actions and penalizing bad ones, like teaching a robot to walk through trial and error."}
{"input": "What is natural language processing?", "output": "Natural language processing (NLP) is the field of AI focused on enabling computers to understand, interpret, and generate human language."}
{"input": "Explain transfer learning", "output": "Transfer learning uses knowledge from a model trained on one task to improve performance on a different but related task, like using image recognition skills to detect medical conditions."}
{"input": "What is overfitting?", "output": "Overfitting occurs when a model learns the training data too well, including its noise and errors, causing poor performance on new, unseen data."}
{"input": "How do you prevent overfitting?", "output": "Prevent overfitting by using more training data, applying regularization, using dropout layers, or stopping training early when validation performance decreases."}
```

### Step 3: Configure Training

```python
# examples/training/train_qa_model.py

from backend.training.real_trainer import RealTrainer, TrainingConfig
from pathlib import Path

# Configuration with explanations
config = TrainingConfig(
    # Model Selection
    model_name="distilbert-base-uncased",  # Small, fast model for learning
    # Other options:
    # - "bert-base-uncased": Standard BERT, more accurate but slower
    # - "gpt2": For text generation tasks
    # - "facebook/opt-125m": Small GPT-like model

    # Output Settings
    output_dir="./trained_models/qa_model",  # Where to save the model

    # Training Parameters
    num_epochs=3,           # How many times to go through data
    batch_size=8,           # Examples processed together
    learning_rate=2e-5,     # How fast to learn (2e-5 = 0.00002)

    # Hardware Settings
    device="auto",          # Auto-detect GPU or CPU

    # Checkpointing
    save_steps=100,         # Save progress every 100 steps
    eval_steps=50,          # Evaluate every 50 steps

    # Memory Optimization
    gradient_accumulation_steps=4,  # Effective batch = 8 * 4 = 32

    # Logging
    logging_steps=10,       # Log every 10 steps
)

print("Configuration created!")
print(f"  Model: {config.model_name}")
print(f"  Output: {config.output_dir}")
print(f"  Epochs: {config.num_epochs}")
```

### Step 4: Initialize the Trainer

```python
# Continue from Step 3

# Create the trainer
trainer = RealTrainer(config)

print(f"Trainer initialized!")
print(f"  Device: {trainer.device}")
print(f"  Training dependencies available: {trainer.training_available}")
```

### Step 5: Start Training

```python
# Continue from Step 4

# Path to your training data
data_path = Path("examples/training/sample_data.jsonl")

# Start training!
print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50 + "\n")

try:
    trainer.train(str(data_path))
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"\nModel saved to: {config.output_dir}")
except Exception as e:
    print(f"Training failed: {e}")
    print("\nTroubleshooting tips:")
    print("  - Check if you have enough memory")
    print("  - Try reducing batch_size")
    print("  - Ensure data file exists and is valid")
```

### Step 6: Complete Training Script

Here's the full script:

```python
#!/usr/bin/env python3
"""
TinyForgeAI Training Example

This script demonstrates how to train a model using TinyForgeAI.
Run with: python examples/training/train_qa_model.py
"""

import os
import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.training.real_trainer import RealTrainer, TrainingConfig


def main():
    print("="*60)
    print("TinyForgeAI Training Example")
    print("="*60)

    # Step 1: Configuration
    print("\n[1/4] Creating configuration...")
    config = TrainingConfig(
        model_name="distilbert-base-uncased",
        output_dir="./trained_models/qa_model",
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        save_steps=100,
        logging_steps=10,
    )
    print(f"      Model: {config.model_name}")
    print(f"      Epochs: {config.num_epochs}")

    # Step 2: Initialize Trainer
    print("\n[2/4] Initializing trainer...")
    trainer = RealTrainer(config)

    if not trainer.training_available:
        print("\n⚠️  Training dependencies not installed!")
        print("   Install with: pip install -e '.[training]'")
        return

    print(f"      Device: {trainer.device}")

    # Step 3: Prepare Data
    print("\n[3/4] Preparing data...")
    data_path = project_root / "examples" / "training" / "sample_data.jsonl"

    if not data_path.exists():
        print(f"      Creating sample data at {data_path}")
        create_sample_data(data_path)

    print(f"      Data: {data_path}")

    # Step 4: Train
    print("\n[4/4] Training model...")
    print("-"*60)

    trainer.train(str(data_path))

    print("-"*60)
    print("\n✓ Training complete!")
    print(f"  Model saved to: {config.output_dir}")
    print("\nNext steps:")
    print("  1. Test your model with the inference server")
    print("  2. Export to ONNX for deployment")


def create_sample_data(path: Path):
    """Create sample training data if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = [
        {"input": "What is machine learning?", "output": "Machine learning is AI that learns from data."},
        {"input": "Explain neural networks", "output": "Neural networks are brain-inspired computing systems."},
        {"input": "What is deep learning?", "output": "Deep learning uses multi-layer neural networks."},
        {"input": "Define AI", "output": "AI is artificial intelligence, machines that can think."},
        {"input": "What is NLP?", "output": "NLP is natural language processing for text understanding."},
    ]

    import json
    with open(path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')


if __name__ == "__main__":
    main()
```

---

## Monitoring Training Progress

### Understanding Training Output

```
Epoch 1/3
[████████████████████] 100%
  Step 10/100 | Loss: 2.543 | LR: 2.00e-5
  Step 20/100 | Loss: 1.876 | LR: 1.90e-5
  Step 30/100 | Loss: 1.234 | LR: 1.80e-5
  ...
  Epoch 1 Complete | Avg Loss: 1.543 | Val Loss: 1.432

Epoch 2/3
[████████████████████] 100%
  Step 10/100 | Loss: 1.123 | LR: 1.50e-5  ← Loss decreasing = good!
  ...
```

### What to Look For

| Observation | Meaning | Action |
|-------------|---------|--------|
| Loss decreasing | Model is learning | Keep training |
| Loss not changing | Learning rate too low | Increase LR |
| Loss increasing | Learning rate too high | Decrease LR |
| Val loss increasing | Overfitting | Stop early, add regularization |

### Using TensorBoard

```python
# Enable TensorBoard logging
config = TrainingConfig(
    # ... other settings ...
    logging_dir="./logs",
    report_to="tensorboard",
)

# Then run TensorBoard
# tensorboard --logdir ./logs
```

---

## Troubleshooting Common Issues

### Issue 1: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce batch size
config = TrainingConfig(
    batch_size=4,  # Reduce from 8 to 4
    gradient_accumulation_steps=8,  # Increase to compensate
)

# Or enable gradient checkpointing
config = TrainingConfig(
    gradient_checkpointing=True,
)

# Or use a smaller model
config = TrainingConfig(
    model_name="distilbert-base-uncased",  # Smaller than BERT
)
```

### Issue 2: Loss Not Decreasing

**Symptoms:**
```
Loss stays around the same value for many steps
```

**Solutions:**
```python
# Try different learning rates
for lr in [1e-5, 2e-5, 5e-5, 1e-4]:
    config = TrainingConfig(learning_rate=lr)
    # Test and compare

# Increase epochs
config = TrainingConfig(num_epochs=10)

# Check your data for issues
# - Duplicates?
# - Inconsistent formatting?
# - Not enough examples?
```

### Issue 3: Training Too Slow

**Solutions:**
```python
# Use mixed precision (requires GPU)
config = TrainingConfig(
    fp16=True,  # Half precision
)

# Increase batch size if memory allows
config = TrainingConfig(
    batch_size=16,  # Larger batches = faster
)

# Use multiple GPUs if available
config = TrainingConfig(
    device="cuda",
    multi_gpu=True,
)
```

### Issue 4: Model Not Saving

**Check:**
```python
# Ensure output directory is writable
import os
os.makedirs(config.output_dir, exist_ok=True)

# Check disk space
import shutil
total, used, free = shutil.disk_usage("/")
print(f"Free space: {free // (2**30)} GB")
```

---

## Next Steps

Congratulations on training your first model! Continue learning:

→ **[04-understanding-lora.md](04-understanding-lora.md)** - Train efficiently with LoRA
→ **[05-deploying-your-model.md](05-deploying-your-model.md)** - Serve your model as an API

---

## Quick Reference

```bash
# Quick training command
python examples/training/train_qa_model.py

# With custom data
python -m cli.foremforge train \
  --model distilbert-base-uncased \
  --data my_data.jsonl \
  --output ./my-model \
  --epochs 3

# Monitor with TensorBoard
tensorboard --logdir ./logs
```
