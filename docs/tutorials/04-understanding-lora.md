# Understanding LoRA: Efficient Fine-Tuning

LoRA (Low-Rank Adaptation) is a revolutionary technique that makes fine-tuning large models accessible to everyone. This tutorial explains what LoRA is and how to use it in TinyForgeAI.

## Table of Contents

1. [Why LoRA Matters](#why-lora-matters)
2. [How LoRA Works](#how-lora-works)
3. [Using LoRA in TinyForgeAI](#using-lora-in-tinyforgeai)
4. [LoRA Parameters Explained](#lora-parameters-explained)
5. [Best Practices](#best-practices)

---

## Why LoRA Matters

### The Problem with Traditional Fine-Tuning

```
Traditional Fine-Tuning:
┌─────────────────────────────────────────────┐
│  Large Language Model (7 billion parameters)│
│  ═══════════════════════════════════════    │
│  All 7B parameters need to be updated       │
│  Requires: 28 GB GPU memory                 │
│  Training time: Days                        │
│  Storage: 28 GB per fine-tuned model        │
└─────────────────────────────────────────────┘
```

**Problems:**
- Need expensive GPUs ($10,000+)
- Very slow training
- Each fine-tuned model takes 28GB+ storage
- Hard for individuals and small teams

### The LoRA Solution

```
LoRA Fine-Tuning:
┌─────────────────────────────────────────────┐
│  Original Model (frozen, 7B params)         │
│  ═══════════════════════════════════════    │
│  + Small Adapter (~1M params, 0.01%)        │
│    ┌───┐                                    │
│    │ A │ ← Only this is trained!            │
│    └───┘                                    │
│  Requires: 4-8 GB GPU memory                │
│  Training time: Hours                       │
│  Storage: 4 MB per adapter                  │
└─────────────────────────────────────────────┘
```

**Benefits:**
- Works on consumer GPUs
- 10-100x faster training
- Tiny adapter files (MBs instead of GBs)
- Can swap adapters for different tasks

---

## How LoRA Works

### The Math (Simplified)

Original weight update:
```
W_new = W_old + ΔW

Where ΔW is a huge matrix (same size as original)
```

LoRA insight:
```
ΔW = A × B

Where A and B are much smaller matrices!

Original: 4096 × 4096 = 16.7 million values
LoRA:     4096 × 8 + 8 × 4096 = 65,536 values (256× smaller!)
```

### Visual Explanation

```
TRADITIONAL FINE-TUNING:
┌──────────────────┐
│ Weight Matrix    │  ← Update ALL these values
│ (4096 × 4096)   │     16.7 million parameters
│ ████████████████│
│ ████████████████│
│ ████████████████│
│ ████████████████│
└──────────────────┘

LoRA FINE-TUNING:
┌──────────────────┐     ┌─┐   ┌─────────────────┐
│ Original Weights │  +  │A│ × │        B        │
│ (frozen)         │     │ │   │   (8 × 4096)    │
│ ░░░░░░░░░░░░░░░░│     │ │   └─────────────────┘
│ ░░░░░░░░░░░░░░░░│     │ │   Only 65K params!
│ ░░░░░░░░░░░░░░░░│     │ │
│ ░░░░░░░░░░░░░░░░│     └─┘
└──────────────────┘   (4096×8)
```

### What Gets Trained?

| Layer Type | Traditional | LoRA |
|------------|-------------|------|
| Embedding | Updated | Frozen |
| Attention Q | Updated | A,B adapters |
| Attention K | Updated | A,B adapters |
| Attention V | Updated | A,B adapters |
| Attention O | Updated | A,B adapters |
| Feed Forward | Updated | A,B adapters |
| Output Head | Updated | Frozen |

---

## Using LoRA in TinyForgeAI

### Basic LoRA Training

```python
from backend.training.real_trainer import RealTrainer, TrainingConfig

config = TrainingConfig(
    model_name="meta-llama/Llama-2-7b-hf",  # Large model
    output_dir="./my-lora-model",

    # Enable LoRA
    use_lora=True,

    # LoRA parameters
    lora_r=8,                    # Rank (smaller = less params)
    lora_alpha=32,               # Scaling factor
    lora_dropout=0.1,            # Regularization
    lora_target_modules=["q_proj", "v_proj"],  # Which layers

    # Training settings
    num_epochs=3,
    batch_size=4,               # Can use larger batches with LoRA!
    learning_rate=1e-4,         # LoRA uses higher learning rate
)

trainer = RealTrainer(config)
trainer.train("my_data.jsonl")
```

### Memory Comparison

```python
# Test without LoRA
config_full = TrainingConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    use_lora=False,  # Full fine-tuning
)
# Result: Requires ~28GB GPU memory (won't fit on most GPUs)

# Test with LoRA
config_lora = TrainingConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    use_lora=True,
    lora_r=8,
)
# Result: Requires ~8GB GPU memory (fits on RTX 3070, RTX 4060, etc.)
```

### Loading and Using a LoRA Model

```python
from backend.training.real_trainer import RealTrainer, TrainingConfig

# Load base model with LoRA adapter
config = TrainingConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    lora_path="./my-lora-model",  # Your trained adapter
)

trainer = RealTrainer(config)

# Use for inference
output = trainer.generate("What is machine learning?")
print(output)
```

### Swapping LoRA Adapters

One amazing LoRA feature: swap adapters without reloading the base model!

```python
# Load base model once
trainer = RealTrainer(TrainingConfig(
    model_name="meta-llama/Llama-2-7b-hf"
))

# Use customer support adapter
trainer.load_lora("./adapters/customer-support")
response = trainer.generate("How do I reset my password?")

# Switch to code helper adapter
trainer.load_lora("./adapters/code-helper")
response = trainer.generate("Write a Python function to sort a list")

# Switch to medical adapter
trainer.load_lora("./adapters/medical-qa")
response = trainer.generate("What are symptoms of flu?")
```

---

## LoRA Parameters Explained

### Key Parameters

| Parameter | What It Does | Typical Values | Impact |
|-----------|-------------|----------------|--------|
| `r` (rank) | Size of adapter matrices | 4, 8, 16, 32, 64 | Higher = more capacity, more memory |
| `alpha` | Scaling factor | 16, 32, 64 | Higher = stronger adaptation |
| `dropout` | Regularization | 0.0, 0.05, 0.1 | Helps prevent overfitting |
| `target_modules` | Which layers to adapt | Varies by model | More modules = more adaptation |

### Rank (r) Deep Dive

```
Rank 4:   Very efficient, less expressive
          Memory: Lowest
          Best for: Simple tasks, limited data

Rank 8:   Good balance (recommended default)
          Memory: Low
          Best for: Most use cases

Rank 16:  More expressive
          Memory: Moderate
          Best for: Complex tasks

Rank 32+: Maximum expressiveness
          Memory: Higher
          Best for: When you have lots of data
```

### Alpha Deep Dive

The `alpha` parameter scales the LoRA updates:

```
Effective update = (alpha / r) × LoRA_output

Example:
  r=8, alpha=32: scaling = 32/8 = 4x
  r=8, alpha=16: scaling = 16/8 = 2x
```

**Rule of thumb:** Start with `alpha = 2 × r`

### Target Modules by Model

```python
# For Llama/Mistral models
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# For GPT-2/GPT-Neo
lora_target_modules = ["c_attn", "c_proj"]

# For BERT/RoBERTa
lora_target_modules = ["query", "key", "value", "dense"]

# Maximum coverage (more memory, better results)
lora_target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]
```

---

## Best Practices

### 1. Start Small, Scale Up

```python
# Start with small rank
config = TrainingConfig(
    lora_r=4,        # Start here
    lora_alpha=8,
)
# If results aren't good enough, increase to r=8, r=16, etc.
```

### 2. Use Higher Learning Rates

```python
# LoRA can use higher learning rates than full fine-tuning
config = TrainingConfig(
    learning_rate=1e-4,   # LoRA: 1e-4 to 3e-4
    # vs full fine-tuning: 1e-5 to 5e-5
)
```

### 3. Target the Right Modules

```python
# Minimal (fastest, least memory)
lora_target_modules = ["q_proj", "v_proj"]

# Balanced (recommended)
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Maximum (best quality, more memory)
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
```

### 4. Combine with Other Techniques

```python
config = TrainingConfig(
    use_lora=True,
    lora_r=8,

    # Combine with gradient checkpointing for even less memory
    gradient_checkpointing=True,

    # Use 4-bit quantization for huge models
    load_in_4bit=True,  # QLoRA technique

    # Increase effective batch size
    gradient_accumulation_steps=4,
)
```

### 5. Save and Organize Adapters

```
trained_models/
├── base_model/              # Original model (shared)
└── lora_adapters/
    ├── customer_support/    # 4 MB
    │   └── adapter_model.bin
    ├── code_assistant/      # 4 MB
    │   └── adapter_model.bin
    └── medical_qa/          # 4 MB
        └── adapter_model.bin

Total: 1 base model + multiple tiny adapters
vs: 3 full models (28GB × 3 = 84GB)
```

---

## LoRA vs Full Fine-Tuning Comparison

| Aspect | Full Fine-Tuning | LoRA |
|--------|-----------------|------|
| Memory needed | 28GB+ | 4-8GB |
| Training speed | Slow (days) | Fast (hours) |
| Storage per model | 28GB | 4-50MB |
| Quality | Best possible | 95-99% of full |
| Flexibility | One model | Swap adapters |
| Hardware | A100, H100 | RTX 3070+ |

---

## Troubleshooting

### Issue: "PEFT not installed"

```bash
pip install peft
# or
pip install -e ".[training]"
```

### Issue: LoRA adapter not loading

```python
# Check the adapter path
import os
assert os.path.exists("./my-lora-model/adapter_config.json")
assert os.path.exists("./my-lora-model/adapter_model.bin")
```

### Issue: Results not as good as full fine-tuning

```python
# Try:
# 1. Increase rank
lora_r=16  # or 32

# 2. Target more modules
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# 3. Train longer
num_epochs=5  # or more

# 4. Use more data
```

---

## Next Steps

→ **[05-deploying-your-model.md](05-deploying-your-model.md)** - Deploy your LoRA model as an API

---

## Quick Reference

```python
# Minimal LoRA config
config = TrainingConfig(
    model_name="your-model",
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
)

# Recommended LoRA config
config = TrainingConfig(
    model_name="your-model",
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    learning_rate=1e-4,
)

# Maximum efficiency (QLoRA)
config = TrainingConfig(
    model_name="your-large-model",
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    load_in_4bit=True,
    gradient_checkpointing=True,
)
```
