# Build a FAQ Bot in 15 Minutes

**Time needed:** 15 minutes
**Skill level:** Beginner
**What you'll build:** A chatbot that answers questions about your business/product

---

## What We're Building

```
User: "What are your business hours?"
Bot:  "We're open Monday-Friday, 9 AM to 6 PM."

User: "How do I reset my password?"
Bot:  "Go to Settings > Account > Reset Password, or click 'Forgot Password' on the login page."
```

By the end of this tutorial, you'll have a working FAQ bot trained on YOUR questions and answers.

---

## Prerequisites

- Completed [00-quickstart.md](00-quickstart.md)
- TinyForgeAI installed

---

## Step 1: Prepare Your FAQ Data

First, let's create a CSV file with your FAQs. This is the easiest format for beginners.

Create a file called `my_faqs.csv`:

```csv
question,answer
What are your business hours?,We are open Monday through Friday from 9 AM to 6 PM EST.
How do I contact support?,You can email support@example.com or call 1-800-123-4567.
What is your return policy?,We offer a 30-day money-back guarantee on all products.
How do I reset my password?,Go to the login page and click 'Forgot Password'. Enter your email to receive a reset link.
Do you offer free shipping?,Yes! Free shipping on all orders over $50.
How long does shipping take?,Standard shipping takes 5-7 business days. Express shipping takes 2-3 days.
Can I track my order?,Yes. Once shipped you'll receive an email with your tracking number.
What payment methods do you accept?,We accept Visa Mastercard American Express PayPal and Apple Pay.
Do you have a mobile app?,Yes! Download our app from the App Store or Google Play.
How do I cancel my subscription?,Go to Account Settings > Subscription > Cancel. Your access continues until the billing period ends.
```

> **Tip:** Replace these with YOUR actual FAQs!

---

## Step 2: Convert CSV to Training Format

TinyForgeAI uses JSONL format for training. Let's convert your CSV:

```python
# convert_faq.py - Convert CSV to JSONL training format

import csv
import json

# Read the CSV file
faqs = []
with open("my_faqs.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        faqs.append({
            "input": row["question"],
            "output": row["answer"]
        })

# Save as JSONL
with open("faq_training_data.jsonl", "w", encoding="utf-8") as f:
    for faq in faqs:
        f.write(json.dumps(faq) + "\n")

print(f"Converted {len(faqs)} FAQs to training format!")
print("Output file: faq_training_data.jsonl")

# Show a preview
print("\nPreview of first 3 items:")
print("-" * 50)
for faq in faqs[:3]:
    print(f"Q: {faq['input']}")
    print(f"A: {faq['output'][:60]}...")
    print()
```

Run it:

```bash
python convert_faq.py
```

**Expected output:**
```
Converted 10 FAQs to training format!
Output file: faq_training_data.jsonl

Preview of first 3 items:
--------------------------------------------------
Q: What are your business hours?
A: We are open Monday through Friday from 9 AM to 6 PM...

Q: How do I contact support?
A: You can email support@example.com or call 1-800-123-456...
```

---

## Step 3: Build a Simple FAQ Matcher

Before training a full AI model, let's build a simple FAQ bot that works immediately:

```python
# simple_faq_bot.py - A working FAQ bot (no training needed!)

import json
from difflib import SequenceMatcher

class SimpleFAQBot:
    def __init__(self, faq_file):
        """Load FAQs from JSONL file."""
        self.faqs = []
        with open(faq_file, "r", encoding="utf-8") as f:
            for line in f:
                self.faqs.append(json.loads(line))
        print(f"Loaded {len(self.faqs)} FAQs")

    def find_similarity(self, text1, text2):
        """Calculate how similar two texts are (0 to 1)."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def get_answer(self, question):
        """Find the best matching FAQ for a question."""
        best_match = None
        best_score = 0

        for faq in self.faqs:
            score = self.find_similarity(question, faq["input"])
            if score > best_score:
                best_score = score
                best_match = faq

        if best_score > 0.5:  # At least 50% similar
            return best_match["output"], best_score
        else:
            return "I'm not sure about that. Please contact support for help.", 0

    def chat(self):
        """Start an interactive chat session."""
        print("\n" + "=" * 50)
        print("FAQ Bot Ready! Type 'quit' to exit.")
        print("=" * 50 + "\n")

        while True:
            question = input("You: ").strip()

            if question.lower() in ["quit", "exit", "bye"]:
                print("Bot: Goodbye! Have a great day!")
                break

            if not question:
                continue

            answer, confidence = self.get_answer(question)

            if confidence > 0.7:
                print(f"Bot: {answer}")
            elif confidence > 0.5:
                print(f"Bot: I think this might help: {answer}")
            else:
                print(f"Bot: {answer}")
            print()


# Run the bot
if __name__ == "__main__":
    bot = SimpleFAQBot("faq_training_data.jsonl")
    bot.chat()
```

Run it:

```bash
python simple_faq_bot.py
```

**Try these questions:**
```
You: What time do you open?
Bot: We are open Monday through Friday from 9 AM to 6 PM EST.

You: How can I get my money back?
Bot: I think this might help: We offer a 30-day money-back guarantee on all products.

You: password help
Bot: Go to the login page and click 'Forgot Password'. Enter your email to receive a reset link.
```

---

## Step 4: Load FAQs from a Database

If your FAQs are in a database, TinyForgeAI makes it easy:

```python
# load_from_database.py - Load FAQs from SQLite database

import sqlite3
import json
from connectors.db_connector import DBConnector

# First, let's create a sample database
conn = sqlite3.connect("faqs.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS faqs (
        id INTEGER PRIMARY KEY,
        question TEXT,
        answer TEXT,
        category TEXT
    )
""")

# Insert sample data
sample_faqs = [
    ("What are your hours?", "Open 9-6 weekdays", "General"),
    ("How do I return an item?", "Ship it back within 30 days", "Returns"),
    ("Where do you ship to?", "We ship worldwide", "Shipping"),
]

cursor.executemany(
    "INSERT OR REPLACE INTO faqs (question, answer, category) VALUES (?, ?, ?)",
    sample_faqs
)
conn.commit()
conn.close()

print("Database created!")

# Now use TinyForgeAI to load it
db = DBConnector(db_url="sqlite:///faqs.db")

# Test connection
if db.test_connection():
    print("Connected to database!")

# Stream FAQ data as training samples
print("\nLoading FAQs from database:")
print("-" * 40)

mapping = {"input": "question", "output": "answer"}
for sample in db.stream_samples("SELECT question, answer FROM faqs", mapping):
    print(f"Q: {sample['input']}")
    print(f"A: {sample['output']}")
    print()
```

---

## Step 5: Add More Data Sources

You can combine FAQs from multiple sources:

```python
# combine_sources.py - Combine FAQs from multiple sources

import json
from connectors.file_ingest import ingest_file
from connectors.db_connector import DBConnector

all_faqs = []

# Source 1: From CSV/JSONL file
print("Loading from file...")
with open("faq_training_data.jsonl", "r") as f:
    for line in f:
        all_faqs.append(json.loads(line))

# Source 2: From database (if exists)
try:
    db = DBConnector(db_url="sqlite:///faqs.db")
    if db.test_connection():
        print("Loading from database...")
        mapping = {"input": "question", "output": "answer"}
        for sample in db.stream_samples("SELECT question, answer FROM faqs", mapping):
            all_faqs.append(sample)
except Exception as e:
    print(f"No database found, skipping: {e}")

# Source 3: From a text document (extract Q&A patterns)
try:
    text = ingest_file("sample_faq.txt")
    # Simple pattern: lines starting with "Q:" and "A:"
    lines = text.split("\n")
    current_q = None
    for line in lines:
        if line.strip().startswith("Q:"):
            current_q = line.replace("Q:", "").strip()
        elif line.strip().startswith("A:") and current_q:
            answer = line.replace("A:", "").strip()
            all_faqs.append({"input": current_q, "output": answer})
            current_q = None
    print("Loading from text file...")
except FileNotFoundError:
    print("No text file found, skipping")

# Remove duplicates
seen = set()
unique_faqs = []
for faq in all_faqs:
    key = faq["input"].lower()
    if key not in seen:
        seen.add(key)
        unique_faqs.append(faq)

# Save combined data
with open("combined_faqs.jsonl", "w") as f:
    for faq in unique_faqs:
        f.write(json.dumps(faq) + "\n")

print(f"\nCombined {len(unique_faqs)} unique FAQs!")
print("Saved to: combined_faqs.jsonl")
```

---

## Step 6: Improve Your Bot with Keyword Matching

Make your bot smarter with keyword detection:

```python
# smart_faq_bot.py - FAQ bot with keyword matching

import json
import re
from difflib import SequenceMatcher

class SmartFAQBot:
    def __init__(self, faq_file):
        self.faqs = []
        self.keywords = {}  # keyword -> list of FAQ indices

        with open(faq_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                faq = json.loads(line)
                self.faqs.append(faq)

                # Extract keywords from question
                words = re.findall(r'\b\w+\b', faq["input"].lower())
                for word in words:
                    if len(word) > 3:  # Skip short words
                        if word not in self.keywords:
                            self.keywords[word] = []
                        self.keywords[word].append(i)

        print(f"Loaded {len(self.faqs)} FAQs with {len(self.keywords)} keywords")

    def get_answer(self, question):
        """Find best answer using keywords + similarity."""
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w+\b', question_lower))

        # Find FAQs with matching keywords
        candidate_indices = set()
        for word in question_words:
            if word in self.keywords:
                candidate_indices.update(self.keywords[word])

        # If no keyword matches, check all FAQs
        if not candidate_indices:
            candidate_indices = range(len(self.faqs))

        # Find best match among candidates
        best_match = None
        best_score = 0

        for i in candidate_indices:
            faq = self.faqs[i]
            score = SequenceMatcher(None, question_lower, faq["input"].lower()).ratio()

            # Boost score if multiple keywords match
            faq_words = set(re.findall(r'\b\w+\b', faq["input"].lower()))
            keyword_overlap = len(question_words & faq_words) / max(len(question_words), 1)
            score = score * 0.7 + keyword_overlap * 0.3

            if score > best_score:
                best_score = score
                best_match = faq

        if best_score > 0.4:
            return best_match["output"], best_score
        else:
            return "I couldn't find an answer. Try rephrasing or contact support.", 0

    def chat(self):
        """Interactive chat loop."""
        print("\n" + "=" * 50)
        print("Smart FAQ Bot Ready!")
        print("Type your question, or 'quit' to exit")
        print("=" * 50 + "\n")

        while True:
            question = input("You: ").strip()
            if question.lower() in ["quit", "exit", "bye"]:
                print("Bot: Goodbye!")
                break
            if not question:
                continue

            answer, score = self.get_answer(question)
            confidence = "high" if score > 0.7 else "medium" if score > 0.5 else "low"
            print(f"Bot [{confidence}]: {answer}\n")


if __name__ == "__main__":
    bot = SmartFAQBot("faq_training_data.jsonl")
    bot.chat()
```

---

## Step 7: Create a Web API for Your Bot

Make your bot accessible via HTTP:

```python
# faq_api.py - REST API for your FAQ bot

from fastapi import FastAPI
from pydantic import BaseModel
import json
from difflib import SequenceMatcher

app = FastAPI(title="FAQ Bot API")

# Load FAQs at startup
faqs = []
with open("faq_training_data.jsonl", "r") as f:
    for line in f:
        faqs.append(json.loads(line))

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    confidence: float
    matched_question: str

@app.get("/")
def home():
    return {
        "message": "FAQ Bot API",
        "endpoints": {
            "/ask": "POST a question to get an answer",
            "/faqs": "GET all FAQs",
            "/health": "GET API health status"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "faq_count": len(faqs)}

@app.get("/faqs")
def get_faqs():
    return {"faqs": faqs}

@app.post("/ask", response_model=Answer)
def ask_question(question: Question):
    best_match = None
    best_score = 0

    for faq in faqs:
        score = SequenceMatcher(
            None,
            question.text.lower(),
            faq["input"].lower()
        ).ratio()
        if score > best_score:
            best_score = score
            best_match = faq

    if best_score > 0.4:
        return Answer(
            answer=best_match["output"],
            confidence=round(best_score, 2),
            matched_question=best_match["input"]
        )
    else:
        return Answer(
            answer="Sorry, I don't have an answer for that question.",
            confidence=0,
            matched_question=""
        )

# Run with: uvicorn faq_api:app --reload
```

Run the API:

```bash
pip install fastapi uvicorn
uvicorn faq_api:app --reload
```

Test it:

```bash
# In another terminal:
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"text": "What are your hours?"}'
```

---

## What You've Learned

1. How to prepare FAQ data in CSV and JSONL formats
2. Building a simple FAQ matcher without AI training
3. Loading data from databases
4. Combining multiple data sources
5. Creating a REST API for your bot

---

## What's Next?

| Tutorial | Description |
|----------|-------------|
| [02-document-search.md](02-document-search.md) | Search through PDFs and documents |
| [03-train-your-model.md](03-train-your-model.md) | Train a real AI model on your FAQs |

---

## Complete Files Checklist

After this tutorial, you should have:

- [ ] `my_faqs.csv` - Your FAQ data
- [ ] `faq_training_data.jsonl` - Training format data
- [ ] `simple_faq_bot.py` - Basic FAQ bot
- [ ] `smart_faq_bot.py` - Improved bot with keywords
- [ ] `faq_api.py` - REST API

---

**Great job!** You've built a working FAQ bot!
