# Module 7: Train Your First Model

**Time needed:** 30-45 minutes (including training time)
**Prerequisites:** Module 6 (prepared training data)
**Goal:** Train a real AI model using TinyForgeAI

---

## What We're Building

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Your Data        Pre-trained Model        Your Model      │
│   (JSONL)     +    (DistilBERT)       =    (Fine-tuned)    │
│                                                             │
│   "What time?"     Knows English           Knows YOUR      │
│   → "9 AM to 6 PM" + Grammar              + business       │
│                    + General knowledge    + Q&A pairs      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Prepare Your Training Data

First, let's create a training dataset if you haven't already:

```python
# create_training_data.py - Create training dataset

import json

# FAQ training data
training_examples = [
    # Greetings
    {"input": "Hello", "output": "Hello! How can I help you today?"},
    {"input": "Hi", "output": "Hi there! What can I assist you with?"},
    {"input": "Hey", "output": "Hey! How can I help?"},
    {"input": "Good morning", "output": "Good morning! How may I assist you?"},
    {"input": "Good afternoon", "output": "Good afternoon! What can I do for you?"},

    # Business Hours
    {"input": "What are your business hours?", "output": "We're open Monday through Friday, 9 AM to 6 PM EST."},
    {"input": "What are your hours?", "output": "Our hours are Monday-Friday, 9 AM to 6 PM EST."},
    {"input": "When do you open?", "output": "We open at 9 AM EST, Monday through Friday."},
    {"input": "When do you close?", "output": "We close at 6 PM EST on weekdays."},
    {"input": "Are you open on weekends?", "output": "No, we're closed on weekends. Our hours are Monday-Friday, 9 AM to 6 PM."},
    {"input": "What time do you open?", "output": "We open at 9 AM EST on weekdays."},
    {"input": "What time do you close?", "output": "We close at 6 PM EST Monday through Friday."},

    # Contact
    {"input": "How do I contact you?", "output": "Email support@example.com or call 1-800-555-1234."},
    {"input": "How do I contact support?", "output": "Reach us at support@example.com or 1-800-555-1234."},
    {"input": "What is your email?", "output": "Our email is support@example.com"},
    {"input": "What is your phone number?", "output": "Our phone number is 1-800-555-1234"},
    {"input": "How can I reach you?", "output": "Email support@example.com or call 1-800-555-1234."},

    # Products
    {"input": "What do you sell?", "output": "We provide TinyForgeAI - a platform for training custom AI models."},
    {"input": "What is TinyForgeAI?", "output": "TinyForgeAI is an open-source platform for fine-tuning language models."},
    {"input": "How much does it cost?", "output": "TinyForgeAI is free and open source under the Apache 2.0 license."},
    {"input": "Is it free?", "output": "Yes! TinyForgeAI is completely free to use."},
    {"input": "What models do you support?", "output": "We support DistilBERT, BERT, GPT-2, Llama, Mistral, and more."},

    # Account
    {"input": "How do I create an account?", "output": "Click 'Sign Up' on our website and follow the instructions."},
    {"input": "How do I reset my password?", "output": "Click 'Forgot Password' on the login page to reset."},
    {"input": "I forgot my password", "output": "No worries! Click 'Forgot Password' on the login page."},

    # Returns/Refunds
    {"input": "What is your return policy?", "output": "We offer a 30-day money-back guarantee on all products."},
    {"input": "Can I get a refund?", "output": "Yes! We have a 30-day money-back guarantee."},
    {"input": "How do I return something?", "output": "Email support@example.com with your order number for return instructions."},

    # Shipping
    {"input": "Do you ship internationally?", "output": "Yes! We ship to over 50 countries worldwide."},
    {"input": "How long does shipping take?", "output": "Standard shipping takes 5-7 business days."},
    {"input": "Is shipping free?", "output": "Free shipping on orders over $50!"},

    # Closing
    {"input": "Thank you", "output": "You're welcome! Is there anything else I can help with?"},
    {"input": "Thanks", "output": "Happy to help! Let me know if you have other questions."},
    {"input": "Goodbye", "output": "Goodbye! Have a wonderful day!"},
    {"input": "Bye", "output": "Bye! Thanks for chatting with us!"},
]

# Save training data
with open('train_data.jsonl', 'w', encoding='utf-8') as f:
    for example in training_examples:
        f.write(json.dumps(example) + '\n')

print(f"✅ Created train_data.jsonl with {len(training_examples)} examples")

# Create a small test set
test_examples = [
    {"input": "Hello there", "output": "Hello! How can I help you today?"},
    {"input": "What are your opening hours?", "output": "We're open Monday through Friday, 9 AM to 6 PM EST."},
    {"input": "How can I contact customer service?", "output": "Email support@example.com or call 1-800-555-1234."},
    {"input": "Thanks for your help", "output": "You're welcome! Is there anything else I can help with?"},
]

with open('test_data.jsonl', 'w', encoding='utf-8') as f:
    for example in test_examples:
        f.write(json.dumps(example) + '\n')

print(f"✅ Created test_data.jsonl with {len(test_examples)} examples")
```

---

## Step 2: Understanding Training Parameters

Before training, let's understand the key parameters:

```python
# training_config_explained.py - Understanding training parameters

"""
Training Configuration Parameters Explained
"""

training_config = {
    # MODEL SELECTION
    "model_name": "distilbert-base-uncased",
    # Options:
    # - distilbert-base-uncased: Small, fast, good for beginners (66M params)
    # - bert-base-uncased: Larger, more accurate (110M params)
    # - gpt2: Good for text generation (124M params)

    # TRAINING DATA
    "train_file": "train_data.jsonl",
    "test_file": "test_data.jsonl",

    # BATCH SIZE
    "batch_size": 8,
    # How many examples to process at once
    # - Smaller (4-8): Uses less memory, slower
    # - Larger (16-32): Uses more memory, faster
    # If you get "out of memory" errors, reduce this

    # EPOCHS
    "epochs": 3,
    # How many times to go through all training data
    # - 1-3: Quick training, might underfit
    # - 3-10: Good balance for most tasks
    # - 10+: Risk of overfitting

    # LEARNING RATE
    "learning_rate": 5e-5,  # 0.00005
    # How much to adjust weights each step
    # - Higher (1e-4): Faster learning, might overshoot
    # - Lower (1e-6): Slower, more precise
    # Default (5e-5) works well for fine-tuning

    # MAX LENGTH
    "max_length": 128,
    # Maximum tokens per example
    # - 128: Good for short Q&A
    # - 256-512: For longer text
    # Longer = more memory, slower training

    # OUTPUT
    "output_dir": "./my_trained_model",
    # Where to save the trained model
}

# Print configuration
print("Training Configuration:")
print("=" * 50)
for key, value in training_config.items():
    print(f"  {key}: {value}")
```

---

## Step 3: Train with TinyForgeAI (Easy Way)

### Using TinyForgeAI's Built-in Trainer

```python
# train_easy.py - Simple training with TinyForgeAI

"""
The easiest way to train a model with TinyForgeAI.
Just specify your data and let TinyForgeAI handle the rest!
"""

try:
    from trainer.simple_trainer import SimpleTrainer

    # Create trainer
    trainer = SimpleTrainer(
        model_name="distilbert-base-uncased",
        task="text-classification"
    )

    # Train!
    print("Starting training...")
    trainer.train(
        train_file="train_data.jsonl",
        output_dir="./my_faq_model",
        epochs=3,
        batch_size=8
    )

    print("✅ Training complete!")
    print(f"   Model saved to: ./my_faq_model")

except ImportError:
    print("TinyForgeAI trainer not found.")
    print("See manual training method below.")
```

---

## Step 4: Train Manually (Learn the Details)

For a deeper understanding, let's train step by step:

```python
# train_detailed.py - Detailed training script with explanations

"""
Manual Training Script
This shows exactly what happens during training.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import os


# ============================================================
# STEP 1: Load and Prepare Data
# ============================================================

class FAQDataset(Dataset):
    """Custom dataset for FAQ training."""

    def __init__(self, filepath: str, tokenizer, max_length: int = 128):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load JSONL file
        print(f"Loading data from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.examples.append(data)

        print(f"  Loaded {len(self.examples)} examples")

        # Create label mapping
        self.labels = list(set(ex['output'] for ex in self.examples))
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        print(f"  Found {len(self.labels)} unique answers (classes)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize input
        encoding = self.tokenizer(
            example['input'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Get label
        label = self.label2id[example['output']]

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }


# ============================================================
# STEP 2: Initialize Model and Tokenizer
# ============================================================

def setup_training(model_name: str, num_labels: int):
    """Set up model and tokenizer."""
    print(f"\nLoading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model for classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    print(f"  Model parameters: {model.num_parameters():,}")

    return tokenizer, model


# ============================================================
# STEP 3: Training Loop
# ============================================================

def train_model(
    model,
    train_dataloader,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    device: str = None
):
    """
    The main training loop.

    This is where the magic happens:
    1. Forward pass: Model makes predictions
    2. Calculate loss: How wrong was it?
    3. Backward pass: Calculate gradients
    4. Update weights: Adjust to reduce error
    """

    # Determine device (GPU if available, else CPU)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nTraining on: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  (Training will be slower on CPU)")

    model = model.to(device)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Total training steps
    total_steps = len(train_dataloader) * epochs

    # Learning rate scheduler (gradually decrease LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training history
    history = {
        'loss': [],
        'epoch_loss': []
    }

    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batches per epoch: {len(train_dataloader)}")
    print(f"  Total steps: {total_steps}")
    print()

    # Training loop
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 40)

        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Training")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Get loss
            loss = outputs.loss
            epoch_loss += loss.item()
            history['loss'].append(loss.item())

            # Backward pass (calculate gradients)
            loss.backward()

            # Update weights
            optimizer.step()
            scheduler.step()

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Epoch statistics
        avg_loss = epoch_loss / len(train_dataloader)
        history['epoch_loss'].append(avg_loss)
        print(f"  Average loss: {avg_loss:.4f}\n")

    print("✅ Training complete!")
    return history


# ============================================================
# STEP 4: Save the Model
# ============================================================

def save_model(model, tokenizer, output_dir: str, label_mapping: dict):
    """Save the trained model."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving model to {output_dir}...")

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mapping
    with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f, indent=2)

    print("  ✓ Model saved")
    print("  ✓ Tokenizer saved")
    print("  ✓ Label mapping saved")


# ============================================================
# MAIN: Put It All Together
# ============================================================

def main():
    # Configuration
    MODEL_NAME = "distilbert-base-uncased"
    TRAIN_FILE = "train_data.jsonl"
    OUTPUT_DIR = "./my_faq_model"
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 5e-5
    MAX_LENGTH = 128

    print("=" * 60)
    print("  TinyForgeAI Training Script")
    print("=" * 60)

    # Step 1: Load tokenizer first (needed for dataset)
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Step 2: Create dataset
    print("\n[2/4] Preparing dataset...")
    train_dataset = FAQDataset(TRAIN_FILE, tokenizer, MAX_LENGTH)

    # Create data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Step 3: Initialize model
    print("\n[3/4] Initializing model...")
    _, model = setup_training(MODEL_NAME, num_labels=len(train_dataset.labels))

    # Update model's label mapping
    model.config.label2id = train_dataset.label2id
    model.config.id2label = train_dataset.id2label

    # Step 4: Train!
    print("\n[4/4] Training...")
    history = train_model(
        model,
        train_dataloader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # Save the model
    save_model(
        model,
        tokenizer,
        OUTPUT_DIR,
        {"label2id": train_dataset.label2id, "id2label": train_dataset.id2label}
    )

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Model saved to: {OUTPUT_DIR}")
    print(f"  Final loss: {history['epoch_loss'][-1]:.4f}")
    print("\n  Next: Test your model with Module 8!")


if __name__ == "__main__":
    main()
```

---

## Step 5: Train with LoRA (Memory Efficient)

If you have limited memory, use LoRA (Low-Rank Adaptation):

```python
# train_lora.py - Memory-efficient training with LoRA

"""
LoRA Training - Use 80% less memory!

LoRA only trains a small part of the model (the adapters)
instead of all parameters. This means:
- Less memory needed
- Faster training
- Similar quality results
"""

try:
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    print("LoRA Training Setup")
    print("=" * 50)

    # Load base model
    model_name = "distilbert-base-uncased"
    print(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=10  # Adjust based on your data
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification
        r=8,                          # LoRA rank (lower = smaller adapter)
        lora_alpha=32,                # Scaling factor
        lora_dropout=0.1,             # Dropout for regularization
        target_modules=["q_lin", "v_lin"]  # Which layers to adapt
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Show parameter savings
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"\nParameter Comparison:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable (LoRA): {trainable:,}")
    print(f"  Reduction: {(1 - trainable/total)*100:.1f}%")

    print("\n✅ LoRA model ready for training!")
    print("   Use the same training loop as before.")

except ImportError:
    print("PEFT library not installed.")
    print("Install with: pip install peft")
```

---

## Step 6: Monitor Training

```python
# monitor_training.py - Visualize training progress

"""
Training Monitor - See how your model is learning
"""

import json
import matplotlib.pyplot as plt


def plot_training_history(history_file: str = "training_history.json"):
    """Plot training loss over time."""

    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Loss per step
    axes[0].plot(history['loss'], alpha=0.7)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss per Step')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Loss per epoch
    axes[1].plot(history['epoch_loss'], marker='o', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Loss')
    axes[1].set_title('Average Loss per Epoch')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150)
    plt.show()

    print("✅ Saved training_progress.png")


def training_summary(history: dict):
    """Print training summary."""
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)

    initial_loss = history['epoch_loss'][0]
    final_loss = history['epoch_loss'][-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100

    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Improvement:  {improvement:.1f}%")

    if final_loss < 0.5:
        print("\n  ✅ Training looks good!")
    elif final_loss < 1.0:
        print("\n  ⚠️  Consider training for more epochs")
    else:
        print("\n  ❌ Loss is still high - check your data or try more epochs")


# Example usage
if __name__ == "__main__":
    # Create sample history for demonstration
    sample_history = {
        'loss': [2.5, 2.3, 2.1, 1.8, 1.5, 1.2, 0.9, 0.7, 0.5, 0.4],
        'epoch_loss': [2.3, 1.2, 0.5]
    }

    # Save sample
    with open('training_history.json', 'w') as f:
        json.dump(sample_history, f)

    training_summary(sample_history)

    try:
        plot_training_history()
    except Exception as e:
        print(f"Could not plot (matplotlib may not be installed): {e}")
```

---

## What to Expect During Training

```
============================================================
  TinyForgeAI Training Script
============================================================

[1/4] Loading tokenizer...

[2/4] Preparing dataset...
Loading data from train_data.jsonl...
  Loaded 35 examples
  Found 28 unique answers (classes)

[3/4] Initializing model...
Loading model: distilbert-base-uncased
  Model parameters: 66,955,010

[4/4] Training...

Training on: cpu
  (Training will be slower on CPU)

Starting training...
  Epochs: 3
  Batches per epoch: 5
  Total steps: 15

Epoch 1/3
----------------------------------------
Training: 100%|██████████| 5/5 [00:12<00:00, 2.5s/it, loss=2.3456]
  Average loss: 2.1234

Epoch 2/3
----------------------------------------
Training: 100%|██████████| 5/5 [00:11<00:00, 2.3s/it, loss=0.8765]
  Average loss: 0.9876

Epoch 3/3
----------------------------------------
Training: 100%|██████████| 5/5 [00:11<00:00, 2.2s/it, loss=0.3456]
  Average loss: 0.4567

✅ Training complete!

Saving model to ./my_faq_model...
  ✓ Model saved
  ✓ Tokenizer saved
  ✓ Label mapping saved

============================================================
  Training Complete!
============================================================
  Model saved to: ./my_faq_model
  Final loss: 0.4567

  Next: Test your model with Module 8!
```

---

## Troubleshooting

### "CUDA out of memory"
```python
# Reduce batch size
BATCH_SIZE = 4  # or even 2

# Or use CPU
device = 'cpu'  # Slower but works
```

### "Model not learning (loss not decreasing)"
```python
# Try higher learning rate
LEARNING_RATE = 1e-4  # 10x higher

# Or train longer
EPOCHS = 10
```

### "Overfitting (training loss low but test performance bad)"
```python
# Train fewer epochs
EPOCHS = 2

# Or add more training data
```

---

## Checkpoint Quiz

**1. What is "loss" in training?**
<details>
<summary>Click for answer</summary>

Loss measures how wrong the model's predictions are. Lower loss = better predictions. Training tries to minimize loss by adjusting the model's weights.

</details>

**2. Why does loss usually decrease during training?**
<details>
<summary>Click for answer</summary>

Each training step adjusts the model's weights to reduce error. Over many steps, the model learns patterns in the data and makes better predictions.

</details>

**3. What's the advantage of LoRA training?**
<details>
<summary>Click for answer</summary>

LoRA only trains a small "adapter" instead of the full model. This uses ~80% less memory and trains faster while achieving similar results.

</details>

---

## What's Next?

In **Module 8: Test & Improve Your Model**, you'll:
- Load and use your trained model
- Test it with new questions
- Measure accuracy
- Improve through iteration

**You've trained a model! Let's see how well it works.**

---

[← Back to Module 6](06-prepare-training-data.md) | [Continue to Module 8 →](08-test-and-improve.md)
