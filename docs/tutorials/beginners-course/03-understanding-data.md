# Module 3: Understanding Data

**Time needed:** 20 minutes
**Prerequisites:** Module 2 (first AI script)
**Goal:** Learn data formats and create your own dataset

---

## Why Data Matters

```
┌─────────────────────────────────────────────────────────────┐
│                    The AI Equation                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Good Data  +  Good Training  =  Good AI                   │
│                                                             │
│   Bad Data   +  Any Training   =  Bad AI                    │
│                                                             │
│   "Garbage in, garbage out"                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

In AI, **your data IS your AI**. The best algorithms in the world can't fix bad data.

---

## Data Formats Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Common Data Formats                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   CSV    - Spreadsheet data (Excel-compatible)              │
│   JSON   - Structured data (web APIs)                       │
│   JSONL  - One JSON object per line (AI training) ⭐        │
│   TXT    - Plain text documents                             │
│   DOCX   - Word documents                                   │
│   PDF    - Document files                                   │
│                                                             │
│   ⭐ JSONL is the standard format for AI training           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Format 1: CSV (Comma-Separated Values)

### What It Looks Like

```csv
question,answer
What are your hours?,We're open 9-5
How do I contact support?,Email us at help@example.com
Do you offer refunds?,Yes 30-day guarantee
```

### Think of It As

An Excel spreadsheet:

| question | answer |
|----------|--------|
| What are your hours? | We're open 9-5 |
| How do I contact support? | Email us at help@example.com |
| Do you offer refunds? | Yes 30-day guarantee |

### Reading CSV in Python

```python
# read_csv.py - Reading CSV files

import csv

# Open and read the CSV file
with open('faq_data.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    for row in reader:
        print(f"Q: {row['question']}")
        print(f"A: {row['answer']}")
        print("-" * 40)
```

### Writing CSV in Python

```python
# write_csv.py - Creating CSV files

import csv

# Your data
data = [
    {"question": "What are your hours?", "answer": "We're open 9-5"},
    {"question": "How do I contact support?", "answer": "Email help@example.com"},
    {"question": "Do you offer refunds?", "answer": "Yes, 30-day guarantee"},
]

# Write to CSV
with open('my_faq.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['question', 'answer'])
    writer.writeheader()  # Write the column names
    writer.writerows(data)  # Write the data

print("CSV file created!")
```

### Pros and Cons

| Pros | Cons |
|------|------|
| ✅ Easy to edit in Excel | ❌ Struggles with commas in text |
| ✅ Human readable | ❌ No nested data |
| ✅ Universal format | ❌ Not ideal for complex data |

---

## Format 2: JSON (JavaScript Object Notation)

### What It Looks Like

```json
{
  "faqs": [
    {
      "question": "What are your hours?",
      "answer": "We're open 9-5",
      "category": "general"
    },
    {
      "question": "How do I contact support?",
      "answer": "Email help@example.com",
      "category": "support"
    }
  ],
  "version": "1.0",
  "last_updated": "2024-01-15"
}
```

### Think of It As

A nested structure with labels:

```
faq_data
├── faqs (list)
│   ├── item 1
│   │   ├── question: "What are your hours?"
│   │   ├── answer: "We're open 9-5"
│   │   └── category: "general"
│   └── item 2
│       ├── question: "How do I contact support?"
│       └── ...
├── version: "1.0"
└── last_updated: "2024-01-15"
```

### Reading JSON in Python

```python
# read_json.py - Reading JSON files

import json

# Read the JSON file
with open('faq_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Access the data
print(f"Version: {data['version']}")
print(f"Number of FAQs: {len(data['faqs'])}")
print()

for faq in data['faqs']:
    print(f"Q: {faq['question']}")
    print(f"A: {faq['answer']}")
    print(f"Category: {faq['category']}")
    print("-" * 40)
```

### Writing JSON in Python

```python
# write_json.py - Creating JSON files

import json

# Your data
data = {
    "faqs": [
        {
            "question": "What are your hours?",
            "answer": "We're open 9-5",
            "category": "general"
        },
        {
            "question": "How do I contact support?",
            "answer": "Email help@example.com",
            "category": "support"
        }
    ],
    "version": "1.0",
    "created_by": "TinyForgeAI Tutorial"
}

# Write to JSON file
with open('my_faq.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=2)  # indent=2 makes it readable

print("JSON file created!")
```

### Pros and Cons

| Pros | Cons |
|------|------|
| ✅ Handles complex/nested data | ❌ Harder to edit manually |
| ✅ Standard web format | ❌ Must load entire file |
| ✅ Supports any data type | ❌ One syntax error breaks all |

---

## Format 3: JSONL (JSON Lines) ⭐ AI Standard

### What It Looks Like

```jsonl
{"input": "What are your hours?", "output": "We're open 9-5"}
{"input": "How do I contact support?", "output": "Email help@example.com"}
{"input": "Do you offer refunds?", "output": "Yes, 30-day guarantee"}
```

**Key difference:** Each line is a complete, independent JSON object!

### Why JSONL is Best for AI Training

```
┌─────────────────────────────────────────────────────────────┐
│                   Why Use JSONL?                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. Stream Processing                                      │
│      - Read one line at a time                              │
│      - Don't need to load entire file into memory           │
│      - Can process millions of examples                     │
│                                                             │
│   2. Easy to Append                                         │
│      - Just add new lines                                   │
│      - No need to parse entire file                         │
│                                                             │
│   3. Error Isolation                                        │
│      - If one line is bad, others still work                │
│      - Easy to find and fix problems                        │
│                                                             │
│   4. Industry Standard                                      │
│      - OpenAI, Hugging Face, TinyForgeAI all use it         │
│      - Your data works everywhere                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Reading JSONL in Python

```python
# read_jsonl.py - Reading JSONL files

import json

# Read JSONL file
data = []
with open('training_data.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        if line.strip():  # Skip empty lines
            item = json.loads(line)
            data.append(item)

# Print what we loaded
print(f"Loaded {len(data)} examples\n")

for i, item in enumerate(data, 1):
    print(f"Example {i}:")
    print(f"  Input:  {item['input']}")
    print(f"  Output: {item['output']}")
    print()
```

### Writing JSONL in Python

```python
# write_jsonl.py - Creating JSONL files

import json

# Your training data
training_data = [
    {"input": "Hello", "output": "Hello! How can I help you today?"},
    {"input": "What are your hours?", "output": "We're open Monday-Friday, 9 AM to 5 PM."},
    {"input": "How do I contact support?", "output": "Email support@example.com or call 555-1234."},
    {"input": "Do you offer refunds?", "output": "Yes! We have a 30-day money-back guarantee."},
    {"input": "Thank you", "output": "You're welcome! Have a great day!"},
]

# Write to JSONL file
with open('my_training_data.jsonl', 'w', encoding='utf-8') as file:
    for item in training_data:
        line = json.dumps(item)  # Convert dict to JSON string
        file.write(line + '\n')   # Write line + newline

print(f"Created training file with {len(training_data)} examples")
```

### Adding to Existing JSONL

```python
# append_jsonl.py - Adding more data

import json

# New example to add
new_example = {
    "input": "Where are you located?",
    "output": "We're at 123 Main Street, Downtown."
}

# Append to file (note: 'a' for append mode)
with open('my_training_data.jsonl', 'a', encoding='utf-8') as file:
    file.write(json.dumps(new_example) + '\n')

print("Added new example!")
```

---

## Hands-On: Create Your Dataset

Let's create a complete FAQ dataset for training:

```python
# create_dataset.py - Create your first training dataset

import json

# Define your FAQ data
faq_data = [
    # Greetings
    {"input": "Hello", "output": "Hello! Welcome to our support. How can I help you today?"},
    {"input": "Hi", "output": "Hi there! What can I assist you with?"},
    {"input": "Hey", "output": "Hey! How can I help you?"},
    {"input": "Good morning", "output": "Good morning! How may I assist you today?"},
    {"input": "Good afternoon", "output": "Good afternoon! What can I do for you?"},

    # Business Information
    {"input": "What are your business hours?", "output": "We're open Monday through Friday, 9 AM to 6 PM EST. Closed on weekends and major holidays."},
    {"input": "When do you open?", "output": "We open at 9 AM EST on weekdays."},
    {"input": "When do you close?", "output": "We close at 6 PM EST on weekdays."},
    {"input": "Are you open on weekends?", "output": "No, we're closed on weekends. Our business hours are Monday-Friday, 9 AM to 6 PM EST."},

    # Contact Information
    {"input": "How do I contact support?", "output": "You can reach us at support@example.com or call 1-800-123-4567 during business hours."},
    {"input": "What is your email?", "output": "Our support email is support@example.com"},
    {"input": "What is your phone number?", "output": "Our phone number is 1-800-123-4567, available during business hours."},

    # Products & Services
    {"input": "What do you sell?", "output": "We offer AI training tools and services to help businesses build custom AI solutions."},
    {"input": "How much does it cost?", "output": "TinyForgeAI is free and open source! You only pay for compute if you use cloud services."},
    {"input": "Is there a free trial?", "output": "TinyForgeAI is completely free! It's open source under the Apache 2.0 license."},

    # Shipping (if applicable)
    {"input": "Do you ship internationally?", "output": "Yes! We ship to over 50 countries. International shipping takes 7-14 business days."},
    {"input": "How long does shipping take?", "output": "Standard shipping takes 5-7 business days. Express shipping (additional cost) takes 2-3 days."},
    {"input": "Is shipping free?", "output": "Free shipping on orders over $50. Orders under $50 have a $5.99 shipping fee."},

    # Returns & Refunds
    {"input": "What is your return policy?", "output": "We offer a 30-day money-back guarantee. Simply contact support to initiate a return."},
    {"input": "How do I return an item?", "output": "Contact support@example.com with your order number, and we'll send you a return shipping label."},
    {"input": "How long do refunds take?", "output": "Refunds are processed within 5-7 business days after we receive the returned item."},

    # Account
    {"input": "How do I create an account?", "output": "Click 'Sign Up' on our website, enter your email and create a password. You'll receive a confirmation email."},
    {"input": "How do I reset my password?", "output": "Click 'Forgot Password' on the login page, enter your email, and we'll send you a reset link."},
    {"input": "How do I delete my account?", "output": "Go to Account Settings > Privacy > Delete Account. Note: this action is permanent."},

    # Goodbyes
    {"input": "Thank you", "output": "You're welcome! Is there anything else I can help you with?"},
    {"input": "Thanks", "output": "Happy to help! Let me know if you have any other questions."},
    {"input": "Goodbye", "output": "Goodbye! Have a wonderful day!"},
    {"input": "Bye", "output": "Bye! Thanks for chatting with us!"},
]

# Save as JSONL
with open('faq_training_data.jsonl', 'w', encoding='utf-8') as f:
    for item in faq_data:
        f.write(json.dumps(item) + '\n')

print(f"✅ Created faq_training_data.jsonl with {len(faq_data)} examples")

# Also save statistics
stats = {
    "total_examples": len(faq_data),
    "categories": {
        "greetings": 5,
        "business_info": 4,
        "contact": 3,
        "products": 3,
        "shipping": 3,
        "returns": 3,
        "account": 3,
        "goodbyes": 4
    }
}

print(f"\n📊 Dataset Statistics:")
print(f"   Total examples: {stats['total_examples']}")
for category, count in stats['categories'].items():
    print(f"   - {category}: {count}")
```

Run it:
```bash
python create_dataset.py
```

---

## Converting Between Formats

### CSV to JSONL (Most Common!)

```python
# csv_to_jsonl.py - Convert CSV to JSONL

import csv
import json

def csv_to_jsonl(csv_file: str, jsonl_file: str,
                 input_column: str, output_column: str):
    """Convert a CSV file to JSONL format for training."""

    count = 0
    with open(csv_file, 'r', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)

        with open(jsonl_file, 'w', encoding='utf-8') as jsonl_f:
            for row in reader:
                # Create training example
                example = {
                    "input": row[input_column],
                    "output": row[output_column]
                }
                jsonl_f.write(json.dumps(example) + '\n')
                count += 1

    print(f"✅ Converted {count} rows from {csv_file} to {jsonl_file}")

# Example usage
csv_to_jsonl(
    csv_file='my_data.csv',
    jsonl_file='training_data.jsonl',
    input_column='question',
    output_column='answer'
)
```

### JSON to JSONL

```python
# json_to_jsonl.py - Convert JSON to JSONL

import json

def json_to_jsonl(json_file: str, jsonl_file: str, data_key: str,
                  input_field: str, output_field: str):
    """Convert JSON file to JSONL format."""

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items = data[data_key]  # Get the list of items
    count = 0

    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in items:
            example = {
                "input": item[input_field],
                "output": item[output_field]
            }
            f.write(json.dumps(example) + '\n')
            count += 1

    print(f"✅ Converted {count} items to {jsonl_file}")

# Example usage
json_to_jsonl(
    json_file='faq_data.json',
    jsonl_file='training_data.jsonl',
    data_key='faqs',
    input_field='question',
    output_field='answer'
)
```

---

## Data Quality Checklist

Before training, verify your data:

```python
# validate_data.py - Check data quality

import json

def validate_jsonl(filepath: str) -> dict:
    """Validate a JSONL training file."""

    issues = []
    stats = {
        "total_lines": 0,
        "valid_examples": 0,
        "empty_inputs": 0,
        "empty_outputs": 0,
        "duplicates": 0,
        "avg_input_length": 0,
        "avg_output_length": 0
    }

    seen_inputs = set()
    input_lengths = []
    output_lengths = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stats["total_lines"] += 1

            # Skip empty lines
            if not line.strip():
                continue

            # Try to parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            # Check required fields
            if "input" not in data:
                issues.append(f"Line {line_num}: Missing 'input' field")
                continue

            if "output" not in data:
                issues.append(f"Line {line_num}: Missing 'output' field")
                continue

            # Check for empty values
            if not data["input"].strip():
                stats["empty_inputs"] += 1
                issues.append(f"Line {line_num}: Empty input")

            if not data["output"].strip():
                stats["empty_outputs"] += 1
                issues.append(f"Line {line_num}: Empty output")

            # Check for duplicates
            if data["input"] in seen_inputs:
                stats["duplicates"] += 1
                issues.append(f"Line {line_num}: Duplicate input")
            seen_inputs.add(data["input"])

            # Track lengths
            input_lengths.append(len(data["input"]))
            output_lengths.append(len(data["output"]))

            stats["valid_examples"] += 1

    # Calculate averages
    if input_lengths:
        stats["avg_input_length"] = sum(input_lengths) / len(input_lengths)
    if output_lengths:
        stats["avg_output_length"] = sum(output_lengths) / len(output_lengths)

    return {"stats": stats, "issues": issues}


# Run validation
print("Validating training data...")
print("=" * 50)

result = validate_jsonl('faq_training_data.jsonl')

print("\n📊 Statistics:")
for key, value in result["stats"].items():
    if isinstance(value, float):
        print(f"   {key}: {value:.1f}")
    else:
        print(f"   {key}: {value}")

if result["issues"]:
    print(f"\n⚠️  Found {len(result['issues'])} issues:")
    for issue in result["issues"][:10]:  # Show first 10
        print(f"   - {issue}")
    if len(result["issues"]) > 10:
        print(f"   ... and {len(result['issues']) - 10} more")
else:
    print("\n✅ No issues found! Data is ready for training.")
```

---

## Using TinyForgeAI's Sample Data

TinyForgeAI includes sample data you can use:

```python
# use_sample_data.py - Using TinyForgeAI's included data

import json
import os

# Path to sample data (adjust based on your installation)
sample_data_path = "examples/tutorial_data/sample_training_data.jsonl"

if os.path.exists(sample_data_path):
    print("Loading TinyForgeAI sample data...")

    with open(sample_data_path, 'r') as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(examples)} examples\n")

    # Show first 5 examples
    print("Sample examples:")
    print("-" * 50)
    for example in examples[:5]:
        print(f"Input:  {example['input']}")
        print(f"Output: {example['output'][:60]}...")
        print()
else:
    print(f"Sample data not found at {sample_data_path}")
    print("Make sure TinyForgeAI is installed correctly.")
```

---

## Checkpoint Quiz

**1. Why is JSONL preferred for AI training over regular JSON?**
<details>
<summary>Click for answer</summary>

- Can process one line at a time (memory efficient)
- Easy to append new data
- If one line has an error, others still work
- Industry standard format

</details>

**2. What should every JSONL training example contain?**
<details>
<summary>Click for answer</summary>

At minimum: `input` (what the user says) and `output` (what the AI should respond). Example:
```json
{"input": "Hello", "output": "Hi! How can I help?"}
```

</details>

**3. Why should we validate data before training?**
<details>
<summary>Click for answer</summary>

- Find errors before they waste training time
- Remove duplicates that bias the model
- Ensure consistent formatting
- Verify all required fields exist

</details>

---

## Summary

| Format | Best For | Structure |
|--------|----------|-----------|
| CSV | Spreadsheet data | Rows and columns |
| JSON | Complex nested data | Single structure |
| JSONL | AI training ⭐ | One object per line |

| Concept | Why It Matters |
|---------|----------------|
| Data quality | Determines AI quality |
| Input/Output pairs | How AI learns Q&A |
| Validation | Catches problems early |
| Conversion | Use data from any source |

---

## What's Next?

In **Module 4: Build a Simple Bot**, you'll:
- Use your dataset to create a working FAQ bot
- Learn about TinyForgeAI's built-in tools
- Create a REST API for your bot
- Test it with real questions

**You now understand the fuel that powers AI. Let's put it to use!**

---

[← Back to Module 2](02-your-first-ai-script.md) | [Continue to Module 4 →](04-build-a-simple-bot.md)
