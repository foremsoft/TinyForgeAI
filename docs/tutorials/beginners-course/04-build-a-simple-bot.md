# Module 4: Build a Simple Bot

**Time needed:** 25 minutes
**Prerequisites:** Module 3 (understanding data)
**Goal:** Create a working FAQ chatbot with TinyForgeAI

---

## What We're Building

A complete FAQ bot with:
- Data loading from files
- Smart answer matching
- REST API for web integration
- Interactive chat mode

```
┌─────────────────────────────────────────────────────────────┐
│                    Your FAQ Bot                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   User: "What time do you close?"                          │
│         ↓                                                   │
│   Bot:  Load knowledge base                                 │
│         ↓                                                   │
│   Bot:  Find most similar question                          │
│         ↓                                                   │
│   Bot:  "We close at 6 PM EST on weekdays."                │
│         (Confidence: 72%)                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Project Structure

First, let's organize our project properly:

```
faq_bot_project/
├── data/
│   └── faq_data.jsonl       # Your training data
├── src/
│   ├── __init__.py          # Makes src a Python package
│   ├── bot.py               # The bot logic
│   └── api.py               # REST API
├── tests/
│   └── test_bot.py          # Tests
└── main.py                  # Entry point
```

### Create the Folders

```bash
# Create project structure
mkdir -p faq_bot_project/data
mkdir -p faq_bot_project/src
mkdir -p faq_bot_project/tests
cd faq_bot_project
```

Or in Python:

```python
# setup_project.py - Create project structure
import os

folders = [
    'faq_bot_project/data',
    'faq_bot_project/src',
    'faq_bot_project/tests'
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")

# Create __init__.py
open('faq_bot_project/src/__init__.py', 'w').close()
print("Created: faq_bot_project/src/__init__.py")

print("\n✅ Project structure created!")
```

---

## Step 2: Create Training Data

Save this as `data/faq_data.jsonl`:

```python
# create_faq_data.py - Generate FAQ training data

import json

faq_data = [
    # Greetings (important for natural conversation)
    {"input": "Hello", "output": "Hello! Welcome to our support. How can I help you?"},
    {"input": "Hi", "output": "Hi there! What can I assist you with today?"},
    {"input": "Hey", "output": "Hey! How can I help you?"},
    {"input": "Good morning", "output": "Good morning! How may I assist you?"},
    {"input": "Good afternoon", "output": "Good afternoon! What can I do for you?"},
    {"input": "Good evening", "output": "Good evening! How can I help?"},

    # Business Hours (multiple variations)
    {"input": "What are your business hours?", "output": "We're open Monday through Friday, 9 AM to 6 PM EST."},
    {"input": "What are your hours?", "output": "Our hours are Monday-Friday, 9 AM to 6 PM EST."},
    {"input": "When do you open?", "output": "We open at 9 AM EST, Monday through Friday."},
    {"input": "When do you close?", "output": "We close at 6 PM EST on weekdays."},
    {"input": "Are you open on Saturday?", "output": "No, we're closed on weekends. We're open Monday-Friday, 9 AM to 6 PM EST."},
    {"input": "Are you open on Sunday?", "output": "No, we're closed on Sundays. Our business hours are Monday-Friday, 9 AM to 6 PM EST."},
    {"input": "What time do you open?", "output": "We open at 9 AM EST, Monday through Friday."},
    {"input": "What time do you close?", "output": "We close at 6 PM EST on weekdays."},

    # Contact Information
    {"input": "How do I contact you?", "output": "You can email us at support@example.com or call 1-800-555-1234."},
    {"input": "How do I contact support?", "output": "Reach our support team at support@example.com or call 1-800-555-1234."},
    {"input": "What is your email?", "output": "Our email is support@example.com"},
    {"input": "What is your phone number?", "output": "Our phone number is 1-800-555-1234"},
    {"input": "How can I reach you?", "output": "Email support@example.com or call 1-800-555-1234 during business hours."},
    {"input": "Contact information", "output": "Email: support@example.com | Phone: 1-800-555-1234"},

    # Products & Pricing
    {"input": "What do you sell?", "output": "We provide TinyForgeAI - a platform for training and deploying custom AI models."},
    {"input": "What is TinyForgeAI?", "output": "TinyForgeAI is an open-source platform for fine-tuning language models. It makes AI training accessible to everyone."},
    {"input": "How much does it cost?", "output": "TinyForgeAI is free and open source! You only pay for compute if using cloud training."},
    {"input": "Is it free?", "output": "Yes! TinyForgeAI is completely free under the Apache 2.0 license."},
    {"input": "What is the price?", "output": "TinyForgeAI is free and open source. No license fees or subscriptions."},

    # Account Management
    {"input": "How do I create an account?", "output": "Click 'Sign Up' on our website, enter your email and password, then verify your email."},
    {"input": "How do I sign up?", "output": "Visit our website and click 'Sign Up'. Enter your details and verify your email."},
    {"input": "How do I reset my password?", "output": "Click 'Forgot Password' on the login page, enter your email, and follow the reset link."},
    {"input": "I forgot my password", "output": "No problem! Click 'Forgot Password' on the login page and enter your email for a reset link."},
    {"input": "How do I change my password?", "output": "Go to Account Settings > Security > Change Password."},
    {"input": "How do I delete my account?", "output": "Go to Account Settings > Privacy > Delete Account. Warning: this is permanent."},

    # Technical Support
    {"input": "How do I get started?", "output": "1) Install with 'pip install tinyforgeai' 2) Prepare your data as JSONL 3) Run the trainer. Check our tutorials for details!"},
    {"input": "How do I install TinyForgeAI?", "output": "Run 'pip install tinyforgeai' or clone from GitHub and run 'pip install -e .'"},
    {"input": "What models are supported?", "output": "We support DistilBERT, BERT, GPT-2, Llama, Mistral, and other Hugging Face transformer models."},
    {"input": "I'm getting an error", "output": "Sorry to hear that! Please email support@example.com with the error message and we'll help you."},
    {"input": "Something is broken", "output": "I apologize for the inconvenience. Please contact support@example.com with details."},

    # Shipping (for e-commerce use cases)
    {"input": "Do you ship internationally?", "output": "Yes! We ship to over 50 countries. International shipping takes 7-14 business days."},
    {"input": "How long does shipping take?", "output": "Standard shipping: 5-7 business days. Express shipping: 2-3 business days."},
    {"input": "Is shipping free?", "output": "Free shipping on orders over $50. Under $50 has a $5.99 flat fee."},
    {"input": "Can I track my order?", "output": "Yes! You'll receive a tracking number via email once your order ships."},

    # Returns & Refunds
    {"input": "What is your return policy?", "output": "We offer a 30-day money-back guarantee on all products."},
    {"input": "How do I return something?", "output": "Email support@example.com with your order number for a return shipping label."},
    {"input": "Can I get a refund?", "output": "Yes! We offer a 30-day money-back guarantee. Contact support to initiate."},
    {"input": "How long do refunds take?", "output": "Refunds are processed within 5-7 business days after receiving the return."},

    # Closing
    {"input": "Thank you", "output": "You're welcome! Is there anything else I can help you with?"},
    {"input": "Thanks", "output": "Happy to help! Let me know if you have other questions."},
    {"input": "Thanks for your help", "output": "My pleasure! Don't hesitate to reach out if you need more assistance."},
    {"input": "Goodbye", "output": "Goodbye! Have a wonderful day!"},
    {"input": "Bye", "output": "Bye! Thanks for chatting with us!"},
    {"input": "See you later", "output": "See you! Come back anytime you have questions."},
]

# Save to file
with open('data/faq_data.jsonl', 'w', encoding='utf-8') as f:
    for item in faq_data:
        f.write(json.dumps(item) + '\n')

print(f"✅ Created data/faq_data.jsonl with {len(faq_data)} Q&A pairs")
```

---

## Step 3: Build the Bot Core

Save this as `src/bot.py`:

```python
# src/bot.py - The FAQ Bot core logic

"""
TinyForgeAI FAQ Bot
A simple but effective FAQ chatbot using text similarity matching.
"""

import json
from difflib import SequenceMatcher
from typing import Optional
from pathlib import Path


class FAQBot:
    """
    A FAQ chatbot that matches user questions to known Q&A pairs.

    How it works:
    1. Load Q&A pairs from a JSONL file
    2. When user asks a question, compare to all known questions
    3. Return the answer for the most similar question
    4. If similarity is too low, say "I don't know"
    """

    def __init__(self, threshold: float = 0.4):
        """
        Initialize the FAQ Bot.

        Args:
            threshold: Minimum similarity score (0-1) to return an answer.
                      Default 0.4 (40%) works well for most cases.
        """
        self.qa_pairs = []
        self.threshold = threshold
        self.stats = {
            "questions_asked": 0,
            "answers_found": 0,
            "no_match": 0
        }

    def load_data(self, filepath: str) -> int:
        """
        Load Q&A pairs from a JSONL file.

        Args:
            filepath: Path to the JSONL file

        Returns:
            Number of Q&A pairs loaded
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        self.qa_pairs = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if 'input' in data and 'output' in data:
                        self.qa_pairs.append({
                            'question': data['input'],
                            'answer': data['output']
                        })
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} is not valid JSON: {e}")

        return len(self.qa_pairs)

    def add_qa(self, question: str, answer: str):
        """
        Add a single Q&A pair to the knowledge base.

        Args:
            question: The question text
            answer: The answer text
        """
        self.qa_pairs.append({
            'question': question,
            'answer': answer
        })

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.

        Uses SequenceMatcher which compares character sequences.
        Returns a score between 0 (completely different) and 1 (identical).
        """
        return SequenceMatcher(
            None,
            text1.lower().strip(),
            text2.lower().strip()
        ).ratio()

    def find_answer(self, question: str) -> dict:
        """
        Find the best answer for a question.

        Args:
            question: The user's question

        Returns:
            Dictionary with:
            - answer: The best matching answer (or default message)
            - confidence: Similarity score (0-1)
            - matched_question: The question that was matched
            - found: Whether a good match was found
        """
        self.stats["questions_asked"] += 1

        if not self.qa_pairs:
            return {
                "answer": "I don't have any knowledge loaded. Please add some Q&A pairs.",
                "confidence": 0.0,
                "matched_question": None,
                "found": False
            }

        # Find the best match
        best_score = 0.0
        best_qa = None

        for qa in self.qa_pairs:
            score = self._calculate_similarity(question, qa['question'])
            if score > best_score:
                best_score = score
                best_qa = qa

        # Check if the match is good enough
        if best_score >= self.threshold and best_qa:
            self.stats["answers_found"] += 1
            return {
                "answer": best_qa['answer'],
                "confidence": best_score,
                "matched_question": best_qa['question'],
                "found": True
            }
        else:
            self.stats["no_match"] += 1
            return {
                "answer": "I'm not sure about that. Could you rephrase your question or contact support@example.com for help?",
                "confidence": best_score,
                "matched_question": best_qa['question'] if best_qa else None,
                "found": False
            }

    def get_stats(self) -> dict:
        """Get usage statistics."""
        total = self.stats["questions_asked"]
        found = self.stats["answers_found"]

        return {
            **self.stats,
            "knowledge_base_size": len(self.qa_pairs),
            "success_rate": (found / total * 100) if total > 0 else 0
        }

    def chat(self):
        """
        Start an interactive chat session.

        Type 'quit' to exit, 'stats' to see statistics.
        """
        print("\n" + "=" * 60)
        print("  FAQ Bot - Interactive Mode")
        print("  Commands: 'quit' to exit, 'stats' for statistics")
        print("=" * 60 + "\n")

        if not self.qa_pairs:
            print("⚠️  Warning: No Q&A pairs loaded!")
            print("   Use bot.load_data('path/to/data.jsonl') first.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("Bot: Goodbye! Have a great day!")
                break

            if user_input.lower() == 'stats':
                stats = self.get_stats()
                print(f"\n📊 Statistics:")
                print(f"   Questions asked: {stats['questions_asked']}")
                print(f"   Answers found: {stats['answers_found']}")
                print(f"   No match: {stats['no_match']}")
                print(f"   Success rate: {stats['success_rate']:.1f}%")
                print(f"   Knowledge base size: {stats['knowledge_base_size']}\n")
                continue

            if not user_input:
                continue

            # Get answer
            result = self.find_answer(user_input)
            confidence_pct = int(result['confidence'] * 100)

            print(f"Bot: {result['answer']}")
            print(f"     (Confidence: {confidence_pct}%)")

            if result['matched_question'] and result['found']:
                print(f"     (Matched: \"{result['matched_question']}\")")

            print()


# Main entry point for testing
if __name__ == "__main__":
    # Create bot
    bot = FAQBot(threshold=0.4)

    # Load data
    try:
        count = bot.load_data("data/faq_data.jsonl")
        print(f"✅ Loaded {count} Q&A pairs")
    except FileNotFoundError:
        print("⚠️  Data file not found. Adding sample data...")
        bot.add_qa("Hello", "Hello! How can I help you?")
        bot.add_qa("What are your hours?", "We're open 9 AM to 6 PM, Monday-Friday.")
        bot.add_qa("Thank you", "You're welcome!")

    # Start chat
    bot.chat()
```

---

## Step 4: Test It!

### Run the bot directly:

```bash
cd faq_bot_project
python src/bot.py
```

### Expected output:

```
✅ Loaded 52 Q&A pairs

============================================================
  FAQ Bot - Interactive Mode
  Commands: 'quit' to exit, 'stats' for statistics
============================================================

You: Hello
Bot: Hello! Welcome to our support. How can I help you?
     (Confidence: 100%)
     (Matched: "Hello")

You: What time do you open?
Bot: We open at 9 AM EST, Monday through Friday.
     (Confidence: 90%)
     (Matched: "What time do you open?")

You: How much does it cost?
Bot: TinyForgeAI is free and open source! You only pay for compute if using cloud training.
     (Confidence: 100%)
     (Matched: "How much does it cost?")

You: asdfghjkl
Bot: I'm not sure about that. Could you rephrase your question or contact support@example.com for help?
     (Confidence: 15%)
     (Matched: "How do I contact you?")

You: stats

📊 Statistics:
   Questions asked: 4
   Answers found: 3
   No match: 1
   Success rate: 75.0%
   Knowledge base size: 52

You: quit
Bot: Goodbye! Have a great day!
```

---

## Step 5: Create a REST API

Save this as `src/api.py`:

```python
# src/api.py - REST API for the FAQ Bot

"""
FastAPI REST API for the FAQ Bot.
This allows other applications to use your bot over HTTP.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import sys

# Add parent directory to path so we can import bot
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bot import FAQBot


# Create FastAPI app
app = FastAPI(
    title="FAQ Bot API",
    description="A simple FAQ chatbot API built with TinyForgeAI",
    version="1.0.0"
)

# Allow cross-origin requests (for web apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create and initialize bot
bot = FAQBot(threshold=0.4)

# Try to load data
DATA_FILE = os.getenv("FAQ_DATA_FILE", "data/faq_data.jsonl")
try:
    count = bot.load_data(DATA_FILE)
    print(f"✅ API initialized with {count} Q&A pairs")
except FileNotFoundError:
    print(f"⚠️  Data file not found: {DATA_FILE}")
    print("   Add some Q&A pairs manually or set FAQ_DATA_FILE env variable")


# Request/Response Models
class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str

    class Config:
        json_schema_extra = {
            "example": {"question": "What are your business hours?"}
        }


class AnswerResponse(BaseModel):
    """Response model for an answer."""
    answer: str
    confidence: float
    matched_question: Optional[str]
    found: bool


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    knowledge_base_size: int
    version: str


class StatsResponse(BaseModel):
    """Response model for statistics."""
    questions_asked: int
    answers_found: int
    no_match: int
    success_rate: float
    knowledge_base_size: int


class AddQARequest(BaseModel):
    """Request model for adding a Q&A pair."""
    question: str
    answer: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is your favorite color?",
                "answer": "As an AI, I don't have personal preferences!"
            }
        }


# API Endpoints

@app.get("/", tags=["General"])
def root():
    """Welcome message and API information."""
    return {
        "message": "Welcome to the FAQ Bot API!",
        "docs": "/docs",
        "endpoints": {
            "POST /ask": "Ask a question",
            "GET /health": "Check API health",
            "GET /stats": "View usage statistics",
            "POST /add": "Add a Q&A pair"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Check if the API is running properly."""
    return HealthResponse(
        status="healthy",
        knowledge_base_size=len(bot.qa_pairs),
        version="1.0.0"
    )


@app.get("/stats", response_model=StatsResponse, tags=["General"])
def stats():
    """Get usage statistics."""
    s = bot.get_stats()
    return StatsResponse(
        questions_asked=s["questions_asked"],
        answers_found=s["answers_found"],
        no_match=s["no_match"],
        success_rate=s["success_rate"],
        knowledge_base_size=s["knowledge_base_size"]
    )


@app.post("/ask", response_model=AnswerResponse, tags=["Chat"])
def ask(request: QuestionRequest):
    """
    Ask a question and get an answer.

    The bot searches its knowledge base for the most similar question
    and returns the corresponding answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = bot.find_answer(request.question)

    return AnswerResponse(
        answer=result["answer"],
        confidence=round(result["confidence"], 3),
        matched_question=result["matched_question"],
        found=result["found"]
    )


@app.post("/add", tags=["Management"])
def add_qa(request: AddQARequest):
    """
    Add a new Q&A pair to the knowledge base.

    Note: This only adds to memory. Restart will lose the addition.
    For persistence, add to your JSONL file.
    """
    if not request.question.strip() or not request.answer.strip():
        raise HTTPException(status_code=400, detail="Question and answer cannot be empty")

    bot.add_qa(request.question, request.answer)

    return {
        "message": "Q&A pair added successfully",
        "knowledge_base_size": len(bot.qa_pairs)
    }


@app.get("/questions", tags=["Management"])
def list_questions(limit: int = 10, offset: int = 0):
    """List questions in the knowledge base (paginated)."""
    questions = [qa["question"] for qa in bot.qa_pairs]
    total = len(questions)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "questions": questions[offset:offset + limit]
    }


# Run with: uvicorn src.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run the API:

```bash
# From the faq_bot_project directory
cd faq_bot_project
python src/api.py
```

Or with uvicorn (auto-reload for development):

```bash
uvicorn src.api:app --reload
```

### Test the API:

Open your browser to: http://localhost:8000/docs

You'll see an interactive API documentation page!

### Test with curl:

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are your hours?"}'

# Get stats
curl http://localhost:8000/stats
```

### Test with Python:

```python
# test_api.py - Test the API with Python

import requests

API_URL = "http://localhost:8000"

# Test health
response = requests.get(f"{API_URL}/health")
print("Health:", response.json())

# Ask questions
questions = [
    "What time do you open?",
    "How do I contact support?",
    "Do you ship internationally?",
    "What is the meaning of life?"
]

print("\nAsking questions:")
print("-" * 50)

for question in questions:
    response = requests.post(
        f"{API_URL}/ask",
        json={"question": question}
    )
    result = response.json()

    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"   Confidence: {result['confidence']*100:.0f}%")
    print(f"   Found: {result['found']}")
```

---

## Step 6: Create a Main Entry Point

Save this as `main.py`:

```python
# main.py - Main entry point for the FAQ Bot

"""
TinyForgeAI FAQ Bot - Main Entry Point

Usage:
    python main.py chat      # Interactive chat mode
    python main.py api       # Start REST API server
    python main.py test      # Run quick tests
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bot import FAQBot


def run_chat():
    """Run interactive chat mode."""
    bot = FAQBot(threshold=0.4)

    # Load data
    data_file = "data/faq_data.jsonl"
    try:
        count = bot.load_data(data_file)
        print(f"✅ Loaded {count} Q&A pairs from {data_file}")
    except FileNotFoundError:
        print(f"⚠️  Data file not found: {data_file}")
        print("   Creating sample data...")
        bot.add_qa("Hello", "Hello! How can I help you?")
        bot.add_qa("What are your hours?", "We're open 9 AM - 6 PM, Monday-Friday.")
        bot.add_qa("Thank you", "You're welcome!")
        print(f"   Added {len(bot.qa_pairs)} sample Q&A pairs")

    # Start chat
    bot.chat()


def run_api():
    """Start the REST API server."""
    try:
        import uvicorn
        from api import app
        print("🚀 Starting FAQ Bot API...")
        print("   Docs: http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("❌ uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)


def run_tests():
    """Run quick tests."""
    print("Running tests...")
    print("=" * 50)

    bot = FAQBot(threshold=0.4)

    # Test 1: Add Q&A pairs
    bot.add_qa("Hello", "Hi there!")
    bot.add_qa("What are your hours?", "9 AM to 6 PM")
    print(f"✅ Test 1: Added {len(bot.qa_pairs)} Q&A pairs")

    # Test 2: Find exact match
    result = bot.find_answer("Hello")
    assert result["found"] == True
    assert result["confidence"] == 1.0
    print(f"✅ Test 2: Exact match works (confidence: {result['confidence']*100}%)")

    # Test 3: Find similar match
    result = bot.find_answer("What time do you open?")
    assert result["found"] == True
    print(f"✅ Test 3: Similar match works (confidence: {result['confidence']*100:.0f}%)")

    # Test 4: No match
    result = bot.find_answer("asdfghjkl")
    assert result["found"] == False
    print(f"✅ Test 4: No match handled correctly")

    # Test 5: Load from file
    try:
        count = bot.load_data("data/faq_data.jsonl")
        print(f"✅ Test 5: Loaded {count} Q&A pairs from file")
    except FileNotFoundError:
        print("⚠️  Test 5: Skipped (no data file)")

    print("=" * 50)
    print("All tests passed!")


def show_help():
    """Show usage information."""
    print(__doc__)
    print("\nExamples:")
    print("  python main.py chat    # Start chatting")
    print("  python main.py api     # Start API server")
    print("  python main.py test    # Run tests")


# Main entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "chat":
        run_chat()
    elif command == "api":
        run_api()
    elif command == "test":
        run_tests()
    elif command in ["-h", "--help", "help"]:
        show_help()
    else:
        print(f"Unknown command: {command}")
        show_help()
        sys.exit(1)
```

### Usage:

```bash
# Run interactive chat
python main.py chat

# Start API server
python main.py api

# Run tests
python main.py test
```

---

## Checkpoint Quiz

**1. What does the `threshold` parameter control?**
<details>
<summary>Click for answer</summary>

The threshold is the minimum similarity score needed to return an answer. If the best match is below this value (default 0.4 or 40%), the bot says "I don't know" instead of giving a possibly wrong answer.

</details>

**2. Why do we use CORS middleware in the API?**
<details>
<summary>Click for answer</summary>

CORS (Cross-Origin Resource Sharing) allows web browsers to make requests to our API from different domains. Without it, a web page on example.com couldn't use our API running on localhost:8000.

</details>

**3. What's the benefit of having both chat and API modes?**
<details>
<summary>Click for answer</summary>

- **Chat mode**: Great for testing and direct interaction
- **API mode**: Allows other applications (websites, mobile apps, other services) to use your bot

</details>

---

## What You Built

```
faq_bot_project/
├── data/
│   └── faq_data.jsonl       ✅ 52 Q&A pairs
├── src/
│   ├── __init__.py          ✅ Package marker
│   ├── bot.py               ✅ Core bot logic (FAQBot class)
│   └── api.py               ✅ REST API with FastAPI
├── main.py                   ✅ Entry point (chat/api/test)
```

**Features:**
- Load Q&A from JSONL files
- Text similarity matching
- Confidence scores
- Interactive chat mode
- REST API with documentation
- Usage statistics

---

## What's Next?

In **Module 5: What is a Model?**, you'll:
- Understand what AI models really are
- Learn about weights, layers, and training
- See why we need to "train" instead of just "program"
- Prepare for actual model training

**You've built a working bot! Now let's understand the AI magic underneath.**

---

[← Back to Module 3](03-understanding-data.md) | [Continue to Module 5 →](05-what-is-a-model.md)
