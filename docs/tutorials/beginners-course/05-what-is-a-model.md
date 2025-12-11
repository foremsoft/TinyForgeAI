# Module 5: What is a Model?

**Time needed:** 20 minutes
**Prerequisites:** Module 4 (built a simple bot)
**Goal:** Understand AI models, training, and why they work

---

## The Big Picture

In Module 4, we built a bot using **text similarity** - comparing strings character by character. It works, but it's limited:

```
❌ "What time do you open?" vs "When do you start?"
   Text similarity: ~35% (words are different)
   But meaning: Same question!

❌ "I want to return this" vs "Refund please"
   Text similarity: ~20% (completely different words)
   But meaning: Same intent!
```

**AI models understand MEANING, not just characters.** That's the magic!

---

## What is a Model?

### Simple Definition

A **model** is a mathematical function that has learned patterns from data.

```
┌─────────────────────────────────────────────────────────────┐
│                      What is a Model?                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input    →    Model (learned patterns)    →    Output     │
│                                                             │
│   "refund"  →   [millions of numbers]       →   "Return"   │
│              →   that encode meaning         →   "Policy"   │
│                                                             │
│   The model learned that "refund", "return", "money back"   │
│   all have similar meanings from seeing many examples.      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Analogy: A Model is Like a Recipe

```
Cooking:
  Ingredients → Recipe → Dish
  (flour, eggs) → (instructions) → (cake)

AI:
  Input → Model → Output
  (question) → (learned patterns) → (answer)
```

The recipe was created by learning from many cooking experiments. The model was created by learning from many examples.

---

## Inside a Model: Weights

### What are Weights?

A model is made up of **millions of numbers** called "weights" (or parameters).

```
Simple Model (3 weights):
┌─────────────────────────────────────────────────────────────┐
│   Input: "Hello"                                            │
│      ↓                                                      │
│   weight_1 = 0.73  ─┐                                       │
│   weight_2 = -0.45  ├── Calculation → Output: "Greeting"    │
│   weight_3 = 0.92  ─┘                                       │
└─────────────────────────────────────────────────────────────┘

Real Language Model (millions of weights):
┌─────────────────────────────────────────────────────────────┐
│   DistilBERT:     66 million weights                        │
│   BERT:           110 million weights                       │
│   GPT-2 Small:    124 million weights                       │
│   GPT-2 Large:    774 million weights                       │
│   Llama 7B:       7,000 million (7 billion) weights         │
└─────────────────────────────────────────────────────────────┘
```

### Where Do Weights Come From?

**Before training:** Weights are random numbers (the model knows nothing)

**After training:** Weights are adjusted to encode knowledge

```
Before Training:
  Input: "What are your hours?"
  Weights: random
  Output: "djfksldfj" (garbage)

After Training:
  Input: "What are your hours?"
  Weights: learned patterns
  Output: "We're open 9-5" (correct!)
```

---

## Training: How Models Learn

### The Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    The Training Loop                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. Show example: "Hello" → "Hi there!"                    │
│         ↓                                                   │
│   2. Model predicts: "Hxkqj" (wrong, weights are random)    │
│         ↓                                                   │
│   3. Compare to correct answer: "Hi there!"                 │
│         ↓                                                   │
│   4. Calculate error: Very wrong!                           │
│         ↓                                                   │
│   5. Adjust weights slightly to reduce error                │
│         ↓                                                   │
│   6. Repeat with next example...                            │
│         ↓                                                   │
│   After thousands of examples:                              │
│   Model predicts: "Hi there!" (correct!)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Analogy: Learning to Throw Darts

```
Attempt 1: Throw → Miss badly → Adjust aim
Attempt 2: Throw → Miss less → Adjust aim
Attempt 3: Throw → Miss a little → Adjust aim
...
Attempt 1000: Throw → Bullseye!

AI Training:
Example 1: Predict → Big error → Adjust weights
Example 2: Predict → Smaller error → Adjust weights
...
Example 10000: Predict → Correct!
```

---

## Key Terms Explained

### Loss (Error)

How wrong the model's prediction is.

```
Prediction: "Hi there!"
Actual: "Hi there!"
Loss: 0 (perfect!)

Prediction: "Goodbye!"
Actual: "Hi there!"
Loss: 5.7 (very wrong)
```

**Goal of training:** Minimize loss (make predictions more accurate)

### Epoch

One complete pass through all training data.

```
Training Data: 1000 examples

Epoch 1: Model sees examples 1-1000
Epoch 2: Model sees examples 1-1000 again
Epoch 3: Model sees examples 1-1000 again
...

More epochs = more learning (usually)
But too many epochs = overfitting (bad)
```

### Batch Size

How many examples to process at once.

```
Training Data: 1000 examples
Batch Size: 32

Batches per epoch: 1000 / 32 = ~31 batches

Larger batch = faster training, needs more memory
Smaller batch = slower training, less memory
```

### Learning Rate

How much to adjust weights after each batch.

```
Large learning rate (0.01):
  Big adjustments → Fast but might overshoot

Small learning rate (0.0001):
  Small adjustments → Slow but precise

Think: Turning a shower knob
  Big turn = temperature changes a lot (might get too hot)
  Small turn = temperature changes gradually (finds right spot)
```

---

## Pre-trained Models: Standing on Giants

### Why Start from Scratch When You Don't Have To?

```
┌─────────────────────────────────────────────────────────────┐
│              Training from Scratch vs Fine-Tuning            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   From Scratch:                                             │
│   ├── Needs: Millions of examples                           │
│   ├── Time: Weeks/months on powerful computers              │
│   ├── Cost: $10,000 - $1,000,000+                          │
│   └── Result: Model learns language from zero               │
│                                                             │
│   Fine-Tuning (What TinyForgeAI does):                      │
│   ├── Needs: 100-1000 examples                              │
│   ├── Time: Minutes/hours on regular computer               │
│   ├── Cost: Free (your electricity)                         │
│   └── Result: Pre-trained model + your knowledge            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### What Pre-trained Models Already Know

```
Pre-trained DistilBERT knows:
✓ English grammar and spelling
✓ Word meanings and synonyms
✓ Sentence structure
✓ Common knowledge (from Wikipedia, books)
✓ That "refund" ≈ "money back" ≈ "return"

What it doesn't know:
✗ Your company's specific FAQ
✗ Your products and services
✗ Your business hours
✗ Your policies

Fine-tuning = Teaching it YOUR specific knowledge
```

---

## Types of Models for Different Tasks

### Text Classification

"What category does this text belong to?"

```
Input: "I want my money back!"
Output: Category → "Refund Request"

Input: "When do you open?"
Output: Category → "Business Hours"
```

Use case: Routing customer questions to the right department

### Text Generation

"Complete this text or write a response"

```
Input: "The customer asked about shipping"
Output: "We offer free shipping on orders over $50..."
```

Use case: Chatbots, content generation

### Question Answering

"Find the answer in this document"

```
Context: "Our store is located at 123 Main St. We're open 9-5."
Question: "Where is the store?"
Output: "123 Main St"
```

Use case: Document search, FAQ bots

### Semantic Similarity

"How similar are these two texts in MEANING?"

```
Text 1: "I want a refund"
Text 2: "Can I get my money back?"
Similarity: 0.95 (very similar meaning!)

Text 1: "I want a refund"
Text 2: "What are your hours?"
Similarity: 0.12 (different meaning)
```

Use case: FAQ matching, duplicate detection

---

## Why TinyForgeAI Uses Fine-Tuning

### The Smart Approach

```
┌─────────────────────────────────────────────────────────────┐
│                  TinyForgeAI's Approach                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. Start with a pre-trained model (e.g., DistilBERT)      │
│      └── Already knows English and general knowledge        │
│                                                             │
│   2. Add your training data (100-1000 examples)             │
│      └── Your specific Q&A, documents, knowledge            │
│                                                             │
│   3. Fine-tune the model (train on your data)               │
│      └── Model learns your domain                           │
│                                                             │
│   4. Result: Model that understands both                    │
│      └── General language + your specific domain            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Comparison: Text Matching vs AI Model

```python
# Text Matching (Module 4):
from difflib import SequenceMatcher

def text_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# "refund" vs "money back"
score = text_similarity("I want a refund", "Can I get my money back?")
print(score)  # 0.29 - Low! (different words)


# AI Model (What you'll learn):
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(a, b):
    embeddings = model.encode([a, b])
    # Calculate cosine similarity
    return cosine_similarity(embeddings[0], embeddings[1])

# "refund" vs "money back"
score = semantic_similarity("I want a refund", "Can I get my money back?")
print(score)  # 0.89 - High! (same meaning)
```

---

## A Simple Neural Network Visualization

### What Happens Inside

```
Input Layer        Hidden Layer       Output Layer
(text as numbers)  (learned patterns)  (prediction)

    ○──────────────────○
     \      w1       / \
      \            /    \
    ○──────────────────○──────────────○
      \            /    /
       \   w2    /    /
    ○──────────────────○
         weights

Input: "Hello"
↓
Convert to numbers: [0.5, 0.2, 0.8, ...]
↓
Multiply by weights, sum, apply function
↓
Output: "Greeting" (with 95% confidence)
```

### The Magic of Layers

```
Layer 1: Recognizes individual words
         "refund" → [concept: money-related]

Layer 2: Understands relationships
         "I want" + "refund" → [intent: request]

Layer 3: Grasps full meaning
         "I want a refund" → [action: process return]

More layers = deeper understanding = "deep learning"
```

---

## Practical Exercise: See a Model in Action

Let's use a pre-trained model to understand text:

```python
# see_model_in_action.py - Understanding embeddings

# Note: Run 'pip install sentence-transformers' first

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print("Loading pre-trained model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded!\n")

    # Test sentences
    sentences = [
        "I want a refund",
        "Can I get my money back?",
        "What are your business hours?",
        "When do you open?",
        "Hello there!",
        "I need help with my order"
    ]

    # Get embeddings (convert text to numbers)
    print("Converting sentences to embeddings...")
    embeddings = model.encode(sentences)

    print(f"Each sentence becomes a vector of {embeddings.shape[1]} numbers\n")

    # Show first sentence's embedding (abbreviated)
    print(f"'{sentences[0]}' becomes:")
    print(f"  [{embeddings[0][0]:.3f}, {embeddings[0][1]:.3f}, {embeddings[0][2]:.3f}, ... , {embeddings[0][-1]:.3f}]")
    print(f"  (384 numbers that encode the meaning)\n")

    # Calculate similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("Semantic Similarities:")
    print("=" * 60)

    pairs = [
        (0, 1),  # refund vs money back
        (2, 3),  # hours vs when open
        (0, 2),  # refund vs hours (different topics)
        (0, 4),  # refund vs hello (different topics)
    ]

    for i, j in pairs:
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"\n'{sentences[i]}'")
        print(f"'{sentences[j]}'")
        print(f"Similarity: {sim:.2%}")

except ImportError:
    print("sentence-transformers not installed.")
    print("Run: pip install sentence-transformers")
    print("\nThis module demonstrates how AI understands meaning!")
```

### Expected Output:

```
Loading pre-trained model...
✓ Model loaded!

Converting sentences to embeddings...
Each sentence becomes a vector of 384 numbers

'I want a refund' becomes:
  [0.042, -0.089, 0.123, ... , 0.056]
  (384 numbers that encode the meaning)

Semantic Similarities:
============================================================

'I want a refund'
'Can I get my money back?'
Similarity: 86.23%   ← Same meaning!

'What are your business hours?'
'When do you open?'
Similarity: 78.45%   ← Same meaning!

'I want a refund'
'What are your business hours?'
Similarity: 23.12%   ← Different topics

'I want a refund'
'Hello there!'
Similarity: 15.67%   ← Very different
```

---

## Checkpoint Quiz

**1. What are "weights" in a model?**
<details>
<summary>Click for answer</summary>

Weights are the millions of numbers that make up a model. They encode patterns learned from training data. During training, weights are adjusted to make the model more accurate.

</details>

**2. What's the difference between training from scratch and fine-tuning?**
<details>
<summary>Click for answer</summary>

- **Training from scratch**: Start with random weights, needs millions of examples, very expensive
- **Fine-tuning**: Start with pre-trained model (already knows language), needs only hundreds of examples, fast and cheap

</details>

**3. Why can an AI model understand that "refund" and "money back" mean the same thing?**
<details>
<summary>Click for answer</summary>

The model was trained on massive amounts of text where these words appeared in similar contexts. It learned that they have similar "meanings" (represented as similar numerical embeddings) even though the characters are different.

</details>

**4. What is an "epoch"?**
<details>
<summary>Click for answer</summary>

One complete pass through all training data. If you have 1000 examples and train for 3 epochs, the model sees each example 3 times.

</details>

---

## Summary

| Concept | Simple Explanation |
|---------|-------------------|
| Model | A mathematical function that learned patterns from data |
| Weights | The numbers that store what the model learned |
| Training | Adjusting weights to make predictions more accurate |
| Loss | How wrong the prediction is (lower = better) |
| Epoch | One pass through all training data |
| Pre-trained | Model already trained on lots of general data |
| Fine-tuning | Teaching a pre-trained model your specific knowledge |
| Embedding | Converting text to numbers that capture meaning |

---

## What's Next?

In **Module 6: Prepare Training Data**, you'll:
- Use TinyForgeAI's connectors to load data from various sources
- Load from CSV, databases, documents, and APIs
- Clean and validate your data
- Create a production-ready training dataset

**Now you understand what's under the hood. Let's prepare real data!**

---

[← Back to Module 4](04-build-a-simple-bot.md) | [Continue to Module 6 →](06-prepare-training-data.md)
