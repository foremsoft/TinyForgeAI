# Preparing Training Data

Good training data is the foundation of a good model. This tutorial teaches you how to create, format, and validate your training datasets.

## Table of Contents

1. [Data Quality Matters](#data-quality-matters)
2. [Supported Data Formats](#supported-data-formats)
3. [Creating Your Dataset](#creating-your-dataset)
4. [Best Practices](#best-practices)
5. [Common Mistakes to Avoid](#common-mistakes-to-avoid)

---

## Data Quality Matters

### The 80/20 Rule of AI

> **80% of AI success comes from data quality, only 20% from model choice.**

A small, high-quality dataset will always beat a large, low-quality one.

### What Makes Good Training Data?

| Characteristic | Good Example | Bad Example |
|---------------|--------------|-------------|
| **Accurate** | "Python was created in 1991" | "Python was created in 2005" |
| **Consistent** | Same format throughout | Mixed formats |
| **Diverse** | Many different examples | Same examples repeated |
| **Relevant** | Matches your use case | Off-topic content |
| **Complete** | Full, meaningful responses | Truncated or partial |

---

## Supported Data Formats

### 1. JSONL (Recommended)

**JSON Lines** - Each line is a valid JSON object.

```jsonl
{"input": "What is machine learning?", "output": "Machine learning is a subset of AI that enables systems to learn from data."}
{"input": "Explain neural networks", "output": "Neural networks are computing systems inspired by biological neural networks in the brain."}
{"input": "What is deep learning?", "output": "Deep learning uses multiple layers of neural networks to analyze data."}
```

**Why JSONL is recommended:**
- Easy to read and edit
- Can process line by line (memory efficient)
- Easy to validate

### 2. JSON Array

```json
[
  {
    "input": "What is machine learning?",
    "output": "Machine learning is a subset of AI..."
  },
  {
    "input": "Explain neural networks",
    "output": "Neural networks are computing systems..."
  }
]
```

### 3. CSV Format

```csv
input,output
"What is machine learning?","Machine learning is a subset of AI..."
"Explain neural networks","Neural networks are computing systems..."
```

### 4. Plain Text

For simple use cases, pairs separated by delimiters:

```text
### Input:
What is machine learning?

### Output:
Machine learning is a subset of AI that enables systems to learn from data.

---

### Input:
Explain neural networks

### Output:
Neural networks are computing systems inspired by biological neural networks.
```

---

## Creating Your Dataset

### Step-by-Step Guide

#### Step 1: Define Your Task

Ask yourself:
- What should my model do?
- What inputs will it receive?
- What outputs should it produce?

**Example Tasks:**
```
Task: Customer Support Bot
Input: Customer questions
Output: Helpful, friendly responses

Task: Code Explainer
Input: Code snippets
Output: Plain English explanations

Task: Email Classifier
Input: Email text
Output: Category (support, sales, spam)
```

#### Step 2: Collect Raw Data

Sources for training data:
- Existing documentation
- FAQ pages
- Support tickets (anonymized)
- Manually written examples
- Public datasets

#### Step 3: Format Your Data

Use the TinyForgeAI data preparation tools:

```python
from backend.data_processing.dataset_loader import DatasetLoader

# Load from various sources
loader = DatasetLoader()

# From a text file
data = loader.load_from_file("raw_data.txt")

# Convert to JSONL format
loader.save_as_jsonl(data, "training_data.jsonl")
```

#### Step 4: Validate Your Data

```python
from backend.data_processing.dataset_loader import DatasetLoader

loader = DatasetLoader()

# Validate your dataset
issues = loader.validate("training_data.jsonl")

if issues:
    print("Found issues:")
    for issue in issues:
        print(f"  - Line {issue['line']}: {issue['message']}")
else:
    print("Dataset is valid!")
```

---

## Best Practices

### 1. Minimum Dataset Size

| Task Complexity | Minimum Examples | Recommended |
|----------------|------------------|-------------|
| Simple (classification) | 100 | 500+ |
| Medium (Q&A) | 500 | 2,000+ |
| Complex (generation) | 1,000 | 5,000+ |

### 2. Balance Your Dataset

**Bad (Unbalanced):**
```
90% questions about topic A
10% questions about topic B
→ Model will be biased toward topic A
```

**Good (Balanced):**
```
50% questions about topic A
50% questions about topic B
→ Model handles both topics well
```

### 3. Include Edge Cases

```jsonl
{"input": "", "output": "I need more information to help you. Could you please provide more details?"}
{"input": "asdfghjkl", "output": "I didn't understand that. Could you please rephrase your question?"}
{"input": "What is ???", "output": "Could you clarify what you're asking about?"}
```

### 4. Vary Your Examples

**Bad (Repetitive):**
```jsonl
{"input": "What is AI?", "output": "AI is artificial intelligence."}
{"input": "What is AI?", "output": "AI is artificial intelligence."}
{"input": "What is AI?", "output": "AI is artificial intelligence."}
```

**Good (Varied):**
```jsonl
{"input": "What is AI?", "output": "AI, or Artificial Intelligence, refers to computer systems that can perform tasks typically requiring human intelligence."}
{"input": "Can you explain AI?", "output": "Artificial Intelligence is the simulation of human intelligence by machines, enabling them to learn, reason, and solve problems."}
{"input": "Define artificial intelligence", "output": "Artificial Intelligence (AI) is a branch of computer science focused on creating systems capable of intelligent behavior."}
```

### 5. Use Consistent Formatting

Pick a style and stick to it:

```jsonl
// Consistent: All outputs start with capital letter, end with period
{"input": "What is Python?", "output": "Python is a programming language."}
{"input": "What is Java?", "output": "Java is a programming language."}

// Inconsistent: Mixed styles
{"input": "What is Python?", "output": "python is a programming language"}
{"input": "What is Java?", "output": "Java is a Programming Language!"}
```

---

## Common Mistakes to Avoid

### 1. ❌ Too Little Data

```
Only 20 examples
→ Model will overfit and not generalize
```

**Fix:** Aim for at least 100-500 examples minimum.

### 2. ❌ Duplicate Data

```jsonl
{"input": "Hello", "output": "Hi there!"}
{"input": "Hello", "output": "Hi there!"}  // Duplicate!
{"input": "Hello", "output": "Hi there!"}  // Duplicate!
```

**Fix:** Remove duplicates using:
```python
# Simple deduplication
seen = set()
unique_data = []
for item in data:
    key = (item['input'], item['output'])
    if key not in seen:
        seen.add(key)
        unique_data.append(item)
```

### 3. ❌ Inconsistent Quality

```jsonl
{"input": "What is AI?", "output": "AI is artificial intelligence, a field of computer science that focuses on creating intelligent machines that work and react like humans."}
{"input": "What is ML?", "output": "ml = machine learning"}  // Too short!
```

**Fix:** Review all examples for consistent quality and length.

### 4. ❌ Incorrect Information

```jsonl
{"input": "Who created Python?", "output": "Python was created by Bill Gates."}  // Wrong!
```

**Fix:** Verify facts before including them in training data.

### 5. ❌ Training on Test Data

```
Using 100% of data for training
→ No way to measure if model actually learned
```

**Fix:** Split your data:
- 80% for training
- 10% for validation
- 10% for testing

---

## Data Preparation Tools

### Using the CLI

```bash
# Validate a dataset
python -m cli.foremforge data validate my_data.jsonl

# Convert formats
python -m cli.foremforge data convert input.csv output.jsonl

# Split dataset
python -m cli.foremforge data split my_data.jsonl --train 0.8 --val 0.1 --test 0.1

# Check statistics
python -m cli.foremforge data stats my_data.jsonl
```

### Expected Output

```
Dataset Statistics:
  Total examples: 1,000
  Average input length: 45 characters
  Average output length: 120 characters
  Unique inputs: 998 (2 duplicates)

Quality Checks:
  ✓ No empty inputs
  ✓ No empty outputs
  ✓ Consistent formatting
  ⚠ 2 duplicate entries found
```

---

## Template: Customer Support Dataset

Here's a template to get you started:

```jsonl
{"input": "How do I reset my password?", "output": "To reset your password, click 'Forgot Password' on the login page, enter your email, and follow the instructions sent to your inbox."}
{"input": "What are your business hours?", "output": "Our business hours are Monday through Friday, 9 AM to 5 PM Eastern Time. We're closed on weekends and major holidays."}
{"input": "How can I contact support?", "output": "You can reach our support team via email at support@example.com or by calling 1-800-EXAMPLE during business hours."}
{"input": "I want a refund", "output": "I'd be happy to help with your refund request. Could you please provide your order number so I can look into this for you?"}
{"input": "Your product doesn't work!", "output": "I'm sorry to hear you're having trouble. Let me help you troubleshoot. Could you describe what's happening and any error messages you see?"}
```

---

## Next Steps

Now that you understand how to prepare data, continue to:

→ **[03-training-your-first-model.md](03-training-your-first-model.md)** - Train your model with this data!

---

## Quick Reference Card

```
CHECKLIST FOR GOOD TRAINING DATA:

□ At least 100+ examples (more is better)
□ Accurate and factual information
□ Consistent formatting throughout
□ Diverse examples covering your use case
□ No duplicates
□ Includes edge cases
□ Split into train/validation/test sets
□ Validated using TinyForgeAI tools
```
