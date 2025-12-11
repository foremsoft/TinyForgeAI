# Module 2: Your First AI Script

**Time needed:** 15 minutes
**Prerequisites:** Module 1 (setup complete)
**Goal:** Write and understand your first AI-powered code

---

## What We're Building

A simple text similarity finder - your first taste of AI magic!

```
Your Input: "What time do you open?"

AI Output:
  "What are your business hours?" → 85% similar
  "How do I contact support?" → 23% similar
  "Where is your store?" → 31% similar

Best Match: "What are your business hours?"
```

This is the foundation of how AI chatbots work!

---

## The Magic Script (10 Lines!)

Create a new file called `first_ai.py`:

```python
# first_ai.py - Your first AI script!

from difflib import SequenceMatcher

# Our "knowledge base" - questions we know about
questions = [
    "What are your business hours?",
    "How do I contact support?",
    "Where is your store located?",
    "Do you offer refunds?",
    "How do I reset my password?"
]

# The question someone is asking
user_question = "What time do you open?"

# Find the most similar question
print(f"\nYour question: {user_question}\n")
print("Comparing to known questions:")
print("-" * 40)

for question in questions:
    # Calculate similarity (0 to 1, where 1 = identical)
    similarity = SequenceMatcher(None, user_question.lower(), question.lower()).ratio()
    percentage = similarity * 100
    print(f"  {percentage:5.1f}% | {question}")

print("-" * 40)
print("\nThis is how AI chatbots find answers!")
```

## Run It!

```bash
python first_ai.py
```

## Output

```
Your question: What time do you open?

Comparing to known questions:
----------------------------------------
   54.9% | What are your business hours?
   32.6% | How do I contact support?
   37.2% | Where is your store located?
   27.9% | Do you offer refunds?
   35.6% | How do I reset my password?
----------------------------------------

This is how AI chatbots find answers!
```

---

## Understanding Every Line

Let's break down what each line does:

### Line 1: The Import
```python
from difflib import SequenceMatcher
```
**What it does:** Imports a tool that compares text similarity
**Analogy:** Like getting a ruler to measure how similar two strings are

### Lines 4-10: The Knowledge Base
```python
questions = [
    "What are your business hours?",
    "How do I contact support?",
    ...
]
```
**What it does:** Creates a list of questions we know
**Analogy:** Like a FAQ sheet with all the questions

### Line 13: User's Question
```python
user_question = "What time do you open?"
```
**What it does:** Stores what the user is asking
**Analogy:** The customer walking up and asking something

### Lines 20-23: The Comparison Loop
```python
for question in questions:
    similarity = SequenceMatcher(None, user_question.lower(), question.lower()).ratio()
    percentage = similarity * 100
    print(f"  {percentage:5.1f}% | {question}")
```
**What it does:**
1. Goes through each known question
2. Calculates how similar it is to the user's question
3. Converts to percentage
4. Prints the result

**Analogy:** Checking each FAQ item to see which one matches best

---

## Try It Yourself: Experiments

### Experiment 1: Change the Question

Edit line 13 and try different questions:

```python
user_question = "How can I get my money back?"  # Try this
```

Run again - see how "Do you offer refunds?" now has the highest score!

### Experiment 2: Add More Questions

Add to the questions list:

```python
questions = [
    "What are your business hours?",
    "How do I contact support?",
    "Where is your store located?",
    "Do you offer refunds?",
    "How do I reset my password?",
    "What time do you close?",        # NEW
    "Are you open on weekends?",       # NEW
]
```

### Experiment 3: Find the Best Match

Add this code at the end to automatically find the best match:

```python
# Find the best match
best_score = 0
best_match = ""

for question in questions:
    similarity = SequenceMatcher(None, user_question.lower(), question.lower()).ratio()
    if similarity > best_score:
        best_score = similarity
        best_match = question

print(f"\n🎯 Best match: {best_match}")
print(f"   Confidence: {best_score * 100:.1f}%")
```

---

## Make It Interactive!

Let's make the script ask for input:

```python
# interactive_ai.py - Interactive version

from difflib import SequenceMatcher

questions = [
    "What are your business hours?",
    "How do I contact support?",
    "Where is your store located?",
    "Do you offer refunds?",
    "How do I reset my password?",
]

# Answers for each question
answers = [
    "We are open Monday-Friday, 9 AM to 6 PM.",
    "Email support@example.com or call 555-1234.",
    "We're at 123 Main Street, Downtown.",
    "Yes! 30-day money-back guarantee.",
    "Click 'Forgot Password' on the login page.",
]

print("=" * 50)
print("   Welcome to AI Assistant!")
print("   Type 'quit' to exit")
print("=" * 50)

while True:
    # Get user input
    user_question = input("\nYour question: ").strip()

    if user_question.lower() == 'quit':
        print("Goodbye!")
        break

    if not user_question:
        print("Please enter a question.")
        continue

    # Find best match
    best_score = 0
    best_index = 0

    for i, question in enumerate(questions):
        similarity = SequenceMatcher(
            None,
            user_question.lower(),
            question.lower()
        ).ratio()

        if similarity > best_score:
            best_score = similarity
            best_index = i

    # Show answer
    if best_score > 0.4:  # 40% threshold
        print(f"\n💬 {answers[best_index]}")
        print(f"   (Confidence: {best_score * 100:.0f}%)")
    else:
        print("\n🤔 I'm not sure about that. Please try rephrasing.")
```

### Run It:

```bash
python interactive_ai.py
```

### Example Session:

```
==================================================
   Welcome to AI Assistant!
   Type 'quit' to exit
==================================================

Your question: When are you open?

💬 We are open Monday-Friday, 9 AM to 6 PM.
   (Confidence: 55%)

Your question: How do I get help?

💬 Email support@example.com or call 555-1234.
   (Confidence: 47%)

Your question: I want a refund

💬 Yes! 30-day money-back guarantee.
   (Confidence: 43%)

Your question: quit
Goodbye!
```

---

## How This Relates to Real AI

What you just built is a **simplified version** of how real chatbots work:

```
Simple Version (What you built):
┌─────────────────────────────────────────────────────────────┐
│  User Question → Text Similarity → Best Match → Answer      │
│                                                             │
│  "What time?"  →  Compare strings →  "Business hours" → "9-6" │
└─────────────────────────────────────────────────────────────┘

Real AI Version (What TinyForgeAI helps you build):
┌─────────────────────────────────────────────────────────────┐
│  User Question → AI Model → Understanding → Smart Answer    │
│                                                             │
│  "What time?"  →  Neural Net → Meaning extracted → "9-6"   │
│                                                             │
│  The AI UNDERSTANDS that:                                   │
│  "What time?" ≈ "When open?" ≈ "Business hours?" ≈ "Hours?" │
└─────────────────────────────────────────────────────────────┘
```

In later modules, we'll upgrade from simple text matching to real AI!

---

## The Complete Script (With TinyForgeAI Style)

Here's how you'd structure this for a real project:

```python
# faq_bot_v1.py - FAQ Bot Version 1 (TinyForgeAI style)

"""
A simple FAQ bot using text similarity.
This is the foundation - we'll upgrade to AI in later modules!
"""

from difflib import SequenceMatcher
import json

class SimpleFAQBot:
    """A simple FAQ bot using text similarity matching."""

    def __init__(self):
        self.qa_pairs = []

    def add_qa(self, question: str, answer: str):
        """Add a question-answer pair."""
        self.qa_pairs.append({
            "question": question,
            "answer": answer
        })

    def load_from_file(self, filepath: str):
        """Load Q&A pairs from a JSONL file."""
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.add_qa(data['input'], data['output'])
        print(f"Loaded {len(self.qa_pairs)} Q&A pairs")

    def get_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        return SequenceMatcher(
            None,
            text1.lower(),
            text2.lower()
        ).ratio()

    def find_answer(self, question: str, threshold: float = 0.4):
        """Find the best answer for a question."""
        best_score = 0
        best_answer = None

        for qa in self.qa_pairs:
            score = self.get_similarity(question, qa['question'])
            if score > best_score:
                best_score = score
                best_answer = qa['answer']

        if best_score >= threshold:
            return {
                "answer": best_answer,
                "confidence": best_score,
                "found": True
            }
        else:
            return {
                "answer": "I don't have information about that.",
                "confidence": best_score,
                "found": False
            }

    def chat(self):
        """Start an interactive chat session."""
        print("\n" + "=" * 50)
        print("FAQ Bot - Type 'quit' to exit")
        print("=" * 50 + "\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Goodbye!")
                break

            if not user_input:
                continue

            result = self.find_answer(user_input)
            confidence = int(result['confidence'] * 100)

            print(f"Bot: {result['answer']}")
            print(f"     (Confidence: {confidence}%)\n")


# Main program
if __name__ == "__main__":
    # Create bot
    bot = SimpleFAQBot()

    # Add some Q&A pairs
    bot.add_qa("What are your business hours?",
               "We're open Monday-Friday, 9 AM to 6 PM EST.")
    bot.add_qa("How do I contact support?",
               "Email support@example.com or call 1-800-123-4567.")
    bot.add_qa("What is your return policy?",
               "30-day money-back guarantee on all products.")
    bot.add_qa("How do I reset my password?",
               "Click 'Forgot Password' on the login page.")
    bot.add_qa("Do you ship internationally?",
               "Yes! We ship to over 50 countries.")

    # Start chatting
    bot.chat()
```

This is proper, clean Python code structured like real TinyForgeAI components!

---

## Checkpoint Quiz

**1. What does SequenceMatcher do?**
<details>
<summary>Click for answer</summary>
It compares two strings and returns a score between 0 and 1 indicating how similar they are. 1 = identical, 0 = completely different.
</details>

**2. Why do we convert text to lowercase before comparing?**
<details>
<summary>Click for answer</summary>
So that "Hello" and "hello" are considered the same. Without this, they'd have lower similarity because of the capital letter.
</details>

**3. What is a "threshold" and why is it useful?**
<details>
<summary>Click for answer</summary>
A threshold is a minimum similarity score (like 0.4 or 40%). If the best match is below this threshold, we say "I don't know" instead of giving a wrong answer.
</details>

---

## What You Learned

| Concept | What It Is |
|---------|-----------|
| Text similarity | Measuring how alike two strings are |
| Knowledge base | A collection of questions and answers |
| Threshold | Minimum confidence needed to give an answer |
| Class structure | Organizing code into reusable components |

---

## What's Next?

In **Module 3: Understanding Data**, you'll:
- Learn about data formats (CSV, JSON, JSONL)
- Create your own dataset
- Load data from files
- Understand why data quality matters

**You've written your first AI code! It only gets more exciting from here.**

---

[← Back to Module 1](01-setup-your-computer.md) | [Continue to Module 3 →](03-understanding-data.md)
