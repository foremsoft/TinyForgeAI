# Train Your Own AI Model

**Time needed:** 30 minutes
**Skill level:** Beginner to Intermediate
**What you'll build:** A custom AI model trained on your data

---

## What We're Building

By the end of this tutorial, you'll have:

1. A dataset prepared for training
2. A fine-tuned AI model
3. A working chatbot using your model

```
Your Data → Training → Your Custom AI Model → Chatbot API
```

---

## Prerequisites

- Completed previous tutorials
- At least 8GB RAM
- Optional: GPU for faster training (works without one too!)

Install training dependencies:

```bash
pip install -e ".[training]"
```

This installs: PyTorch, Transformers, Datasets, PEFT (for LoRA), Accelerate

---

## Part 1: Prepare Your Training Data

### What AI Training Data Looks Like

AI learns from examples. Each example has:
- **Input**: What the user might ask
- **Output**: What the AI should respond

```json
{"input": "What is Python?", "output": "Python is a programming language known for its simplicity."}
{"input": "How do I print hello world?", "output": "Use: print('Hello, World!')"}
```

### Step 1.1: Create Training Data

```python
# prepare_data.py - Create training data for your AI

import json

# Your training examples - ADD YOUR OWN!
training_examples = [
    # Basic greetings
    {
        "input": "Hello",
        "output": "Hello! How can I help you today?"
    },
    {
        "input": "Hi there",
        "output": "Hi! I'm your AI assistant. What would you like to know?"
    },
    {
        "input": "Good morning",
        "output": "Good morning! How can I assist you?"
    },

    # About the product/service (CUSTOMIZE THESE!)
    {
        "input": "What is TinyForgeAI?",
        "output": "TinyForgeAI is a platform for fine-tuning language models and deploying them as microservices. It makes AI training accessible to everyone."
    },
    {
        "input": "How do I get started with TinyForgeAI?",
        "output": "To get started: 1) Install with pip install -e . 2) Prepare your training data as JSONL 3) Run the trainer. Check our tutorials for step-by-step guides."
    },
    {
        "input": "What models does TinyForgeAI support?",
        "output": "TinyForgeAI supports various models including DistilBERT, BERT, GPT-2, Llama, Mistral, and more. Start with smaller models like DistilBERT for learning."
    },

    # Technical questions
    {
        "input": "What is fine-tuning?",
        "output": "Fine-tuning is adapting a pre-trained AI model to your specific use case. Instead of training from scratch, you teach an existing model your domain knowledge."
    },
    {
        "input": "What is LoRA?",
        "output": "LoRA (Low-Rank Adaptation) is an efficient training technique that updates only a small portion of the model. It uses less memory and trains faster while achieving similar results."
    },
    {
        "input": "How much data do I need for training?",
        "output": "For basic fine-tuning, start with 100-500 high-quality examples. More data generally improves results, but quality matters more than quantity."
    },

    # Troubleshooting
    {
        "input": "My training is running out of memory",
        "output": "Try these solutions: 1) Reduce batch_size to 4 or 2 2) Use a smaller model 3) Enable gradient checkpointing 4) Use LoRA for efficient training"
    },
    {
        "input": "Training is very slow",
        "output": "Training speed depends on hardware. Tips: 1) Use a GPU if available 2) Reduce epochs for testing 3) Use a smaller model first 4) Enable mixed precision training"
    },

    # General questions
    {
        "input": "Thank you",
        "output": "You're welcome! Let me know if you have any other questions."
    },
    {
        "input": "Goodbye",
        "output": "Goodbye! Feel free to come back if you need more help."
    }
]

# Data augmentation - create variations
augmented = []
for example in training_examples:
    augmented.append(example)

    # Add question mark variations
    inp = example["input"]
    if not inp.endswith("?") and not inp.endswith(".") and not inp.endswith("!"):
        augmented.append({
            "input": inp + "?",
            "output": example["output"]
        })

    # Add "please" variations
    if not "please" in inp.lower():
        augmented.append({
            "input": inp + " please",
            "output": example["output"]
        })

# Remove duplicates
seen = set()
unique_examples = []
for ex in augmented:
    key = ex["input"].lower()
    if key not in seen:
        seen.add(key)
        unique_examples.append(ex)

# Split into train and validation sets (90/10)
split_point = int(len(unique_examples) * 0.9)
train_data = unique_examples[:split_point]
val_data = unique_examples[split_point:]

# Save training data
with open("train_data.jsonl", "w", encoding="utf-8") as f:
    for example in train_data:
        f.write(json.dumps(example) + "\n")

# Save validation data
with open("val_data.jsonl", "w", encoding="utf-8") as f:
    for example in val_data:
        f.write(json.dumps(example) + "\n")

print(f"Created {len(train_data)} training examples")
print(f"Created {len(val_data)} validation examples")
print("\nFiles saved:")
print("  - train_data.jsonl")
print("  - val_data.jsonl")
```

Run it:

```bash
python prepare_data.py
```

### Step 1.2: Validate Your Data

```python
# validate_data.py - Check your training data for issues

import json

def validate_jsonl(filepath):
    """Validate a JSONL training file."""
    issues = []
    examples = []

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                examples.append(data)

                # Check required fields
                if "input" not in data:
                    issues.append(f"Line {i}: Missing 'input' field")
                if "output" not in data:
                    issues.append(f"Line {i}: Missing 'output' field")

                # Check for empty values
                if data.get("input", "").strip() == "":
                    issues.append(f"Line {i}: Empty input")
                if data.get("output", "").strip() == "":
                    issues.append(f"Line {i}: Empty output")

                # Check for very short outputs
                if len(data.get("output", "")) < 10:
                    issues.append(f"Line {i}: Very short output ({len(data.get('output', ''))} chars)")

            except json.JSONDecodeError as e:
                issues.append(f"Line {i}: Invalid JSON - {e}")

    return examples, issues

print("Validating training data...")
print("=" * 50)

for filepath in ["train_data.jsonl", "val_data.jsonl"]:
    print(f"\n{filepath}:")
    examples, issues = validate_jsonl(filepath)

    print(f"  Total examples: {len(examples)}")

    if issues:
        print(f"  Issues found: {len(issues)}")
        for issue in issues[:5]:  # Show first 5
            print(f"    - {issue}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")
    else:
        print("  No issues found!")

    # Statistics
    if examples:
        avg_input_len = sum(len(ex.get("input", "")) for ex in examples) / len(examples)
        avg_output_len = sum(len(ex.get("output", "")) for ex in examples) / len(examples)
        print(f"  Avg input length: {avg_input_len:.0f} chars")
        print(f"  Avg output length: {avg_output_len:.0f} chars")
```

---

## Part 2: Train Your Model

### Step 2.1: Basic Training (No GPU Required)

```python
# train_basic.py - Train a model (works on CPU!)

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 60)
    print("TinyForgeAI Model Training")
    print("=" * 60)

    # Check if training dependencies are available
    try:
        from backend.training.real_trainer import RealTrainer, TrainingConfig
        print("Training dependencies found!")
    except ImportError as e:
        print(f"\nError: Training dependencies not installed")
        print(f"Run: pip install -e '.[training]'")
        print(f"\nDetails: {e}")
        return

    # Configuration for CPU training (conservative settings)
    config = TrainingConfig(
        # Use a small, fast model
        model_name="distilbert-base-uncased",

        # Output directory
        output_dir="./my_trained_model",

        # Training parameters (conservative for CPU)
        num_epochs=3,           # Number of passes through data
        batch_size=4,           # Small batch for CPU
        learning_rate=2e-5,     # Standard learning rate

        # Logging
        logging_steps=10,       # Print progress every 10 steps

        # Memory optimization
        gradient_accumulation_steps=2,  # Simulate larger batches
    )

    print("\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Output: {config.output_dir}")

    # Check for training data
    train_file = "train_data.jsonl"
    if not Path(train_file).exists():
        print(f"\nError: Training data not found: {train_file}")
        print("Run prepare_data.py first!")
        return

    # Count examples
    with open(train_file) as f:
        num_examples = sum(1 for _ in f)
    print(f"  Training examples: {num_examples}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = RealTrainer(config)
    print(f"  Device: {trainer.device}")

    # Estimate training time
    steps_per_epoch = num_examples // config.batch_size
    total_steps = steps_per_epoch * config.num_epochs
    print(f"  Estimated steps: {total_steps}")

    # Start training
    print("\n" + "=" * 60)
    print("Starting training... (this may take a few minutes)")
    print("=" * 60 + "\n")

    try:
        trainer.train(train_file)
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Model saved to: {config.output_dir}")
        print("=" * 60)

    except Exception as e:
        print(f"\nTraining failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Reduce batch_size to 2")
        print("  2. Reduce num_epochs to 1")
        print("  3. Check your training data format")
        raise

if __name__ == "__main__":
    main()
```

Run training:

```bash
python train_basic.py
```

**Expected output:**
```
TinyForgeAI Model Training
============================================================
Training dependencies found!

Configuration:
  Model: distilbert-base-uncased
  Epochs: 3
  Batch size: 4
  ...

Starting training...
Step 10/45: loss=2.34
Step 20/45: loss=1.89
Step 30/45: loss=1.45
...

Training complete!
Model saved to: ./my_trained_model
```

### Step 2.2: Training with LoRA (Recommended)

LoRA uses much less memory and trains faster:

```python
# train_lora.py - Train with LoRA (efficient training)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 60)
    print("TinyForgeAI LoRA Training")
    print("=" * 60)

    try:
        from backend.training.real_trainer import RealTrainer, TrainingConfig
    except ImportError:
        print("Run: pip install -e '.[training]'")
        return

    config = TrainingConfig(
        model_name="distilbert-base-uncased",
        output_dir="./my_lora_model",

        # Training settings
        num_epochs=3,
        batch_size=4,
        learning_rate=1e-4,  # Higher LR for LoRA

        # Enable LoRA
        use_lora=True,
        lora_r=8,            # LoRA rank (higher = more capacity)
        lora_alpha=16,       # LoRA scaling
        lora_dropout=0.1,    # Prevent overfitting

        logging_steps=10,
    )

    print(f"\nLoRA Configuration:")
    print(f"  LoRA rank (r): {config.lora_r}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  Trainable params: ~{config.lora_r * 768 * 2:,} (vs ~66M full)")

    train_file = "train_data.jsonl"
    if not Path(train_file).exists():
        print(f"Error: {train_file} not found")
        return

    trainer = RealTrainer(config)
    print(f"Device: {trainer.device}")

    print("\nStarting LoRA training...")
    trainer.train(train_file)

    print(f"\nLoRA model saved to: {config.output_dir}")
    print("This model is much smaller than full fine-tuning!")

if __name__ == "__main__":
    main()
```

---

## Part 3: Test Your Trained Model

### Step 3.1: Simple Testing

```python
# test_model.py - Test your trained model

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_model(model_path):
    """Test a trained model with sample inputs."""
    print(f"Loading model from: {model_path}")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers import pipeline

        # For text generation models
        try:
            generator = pipeline("text-generation", model=model_path)

            test_inputs = [
                "What is TinyForgeAI?",
                "How do I get started?",
                "What is fine-tuning?",
                "Hello",
            ]

            print("\nTesting model responses:")
            print("=" * 50)

            for inp in test_inputs:
                result = generator(inp, max_length=100, num_return_sequences=1)
                print(f"\nInput: {inp}")
                print(f"Output: {result[0]['generated_text']}")

        except Exception as e:
            print(f"Note: Model may not be a text generation model")
            print(f"For classification models, use a different testing approach")
            print(f"Error: {e}")

    except ImportError:
        print("Install transformers: pip install transformers")

if __name__ == "__main__":
    model_path = "./my_trained_model"

    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Run training first!")
    else:
        test_model(model_path)
```

### Step 3.2: Interactive Chat with Your Model

```python
# chat_with_model.py - Chat with your trained model

import json
from difflib import SequenceMatcher

class TrainedChatBot:
    """
    A chatbot that uses your training data for responses.
    For production, you'd load the actual trained model.
    This version uses similarity matching as a demonstration.
    """

    def __init__(self, training_file="train_data.jsonl"):
        self.examples = []
        with open(training_file, "r") as f:
            for line in f:
                self.examples.append(json.loads(line))
        print(f"Loaded {len(self.examples)} trained examples")

    def get_response(self, user_input):
        """Find the best response for user input."""
        best_match = None
        best_score = 0

        for example in self.examples:
            score = SequenceMatcher(
                None,
                user_input.lower(),
                example["input"].lower()
            ).ratio()

            if score > best_score:
                best_score = score
                best_match = example

        if best_score > 0.5:
            return best_match["output"], best_score
        else:
            return "I'm not sure how to answer that. Can you rephrase?", 0

    def chat(self):
        """Interactive chat session."""
        print("\n" + "=" * 50)
        print("Chat with Your Trained AI")
        print("Type 'quit' to exit")
        print("=" * 50 + "\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("AI: Goodbye!")
                break

            if not user_input:
                continue

            response, confidence = self.get_response(user_input)

            if confidence > 0.7:
                print(f"AI: {response}")
            elif confidence > 0.5:
                print(f"AI: {response}")
                print(f"    (confidence: {confidence:.0%})")
            else:
                print(f"AI: {response}")

            print()


if __name__ == "__main__":
    bot = TrainedChatBot()
    bot.chat()
```

---

## Part 4: Deploy as an API

```python
# model_api.py - Serve your model as an API

from fastapi import FastAPI
from pydantic import BaseModel
import json
from difflib import SequenceMatcher

app = FastAPI(title="My AI Model API")

# Load training data for inference
EXAMPLES = []
with open("train_data.jsonl", "r") as f:
    for line in f:
        EXAMPLES.append(json.loads(line))

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    confidence: float

@app.get("/")
def home():
    return {
        "name": "My AI Model API",
        "examples_loaded": len(EXAMPLES),
        "usage": "POST /chat with {'message': 'your question'}"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    best_match = None
    best_score = 0

    for example in EXAMPLES:
        score = SequenceMatcher(
            None,
            request.message.lower(),
            example["input"].lower()
        ).ratio()
        if score > best_score:
            best_score = score
            best_match = example

    if best_score > 0.4:
        return ChatResponse(
            response=best_match["output"],
            confidence=round(best_score, 2)
        )
    else:
        return ChatResponse(
            response="I don't have a good answer for that.",
            confidence=0
        )

@app.get("/health")
def health():
    return {"status": "ok"}

# Run with: uvicorn model_api:app --reload
```

Test it:

```bash
uvicorn model_api:app --reload

# In another terminal:
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is TinyForgeAI?"}'
```

---

## Summary: What You've Learned

| Step | What You Did |
|------|--------------|
| 1 | Prepared training data in JSONL format |
| 2 | Validated your data for quality |
| 3 | Trained a model with TinyForgeAI |
| 4 | Tested model responses |
| 5 | Created a chat interface |
| 6 | Deployed as a REST API |

---

## Tips for Better Results

### More Data = Better Model
```python
# Aim for at least 100-500 quality examples
# Quality > Quantity - remove duplicates and errors
```

### Diverse Examples
```python
# Include variations of the same question:
{"input": "What is X?", "output": "..."}
{"input": "Can you explain X?", "output": "..."}
{"input": "Tell me about X", "output": "..."}
```

### Good Output Quality
```python
# Bad: Too short
{"input": "What is Python?", "output": "A language"}

# Good: Informative
{"input": "What is Python?", "output": "Python is a popular programming language known for its simple syntax and readability. It's used for web development, data science, AI, and more."}
```

---

## What's Next?

| Tutorial | Description |
|----------|-------------|
| [04-deploy-your-project.md](04-deploy-your-project.md) | Deploy to production |

---

## Troubleshooting

### "CUDA out of memory"
```python
# Reduce batch size:
config = TrainingConfig(batch_size=2, ...)

# Or use LoRA:
config = TrainingConfig(use_lora=True, ...)
```

### "Training is very slow"
- Use a GPU if available
- Reduce epochs for testing
- Use a smaller model first

### "Model gives bad responses"
- Add more training examples
- Check data quality
- Train for more epochs

---

**Congratulations!** You've trained your own AI model!
