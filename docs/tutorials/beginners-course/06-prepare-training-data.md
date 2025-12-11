# Module 6: Prepare Training Data

**Time needed:** 25 minutes
**Prerequisites:** Module 5 (understanding models)
**Goal:** Use TinyForgeAI connectors to load data from any source

---

## The Data Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                TinyForgeAI Data Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Data Sources              Connectors           Output     │
│                                                             │
│   📄 CSV files      →   ┌──────────────┐                   │
│   📊 Databases      →   │  TinyForgeAI │   →  training.jsonl │
│   📑 PDFs/DOCX      →   │  Connectors  │                   │
│   🌐 APIs           →   └──────────────┘                   │
│   📝 Text files     →                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

TinyForgeAI provides **connectors** that load data from various sources and convert them to the JSONL format needed for training.

---

## Available Connectors

| Connector | Source | Best For |
|-----------|--------|----------|
| CSV Loader | .csv files | Spreadsheet data |
| Database Connector | SQLite, PostgreSQL, MySQL | Existing databases |
| File Ingest | .txt, .md, .pdf, .docx | Documents |
| API Connector | REST APIs | External data |
| Google Docs | Google Docs | Cloud documents |

---

## Method 1: Load from CSV Files

The most common way to start - export from Excel/Google Sheets!

### Your CSV File

Create a file `company_faq.csv`:

```csv
question,answer
What are your hours?,We're open Monday-Friday 9 AM to 6 PM EST.
How do I contact support?,Email support@company.com or call 555-1234.
What is your return policy?,30-day money-back guarantee on all products.
Do you ship internationally?,Yes! We ship to 50+ countries.
```

### Load and Convert to JSONL

```python
# load_from_csv.py - Load FAQ data from CSV

import csv
import json
from pathlib import Path


def csv_to_training_data(
    csv_file: str,
    output_file: str,
    input_column: str = "question",
    output_column: str = "answer"
) -> int:
    """
    Convert a CSV file to JSONL training format.

    Args:
        csv_file: Path to input CSV
        output_file: Path to output JSONL
        input_column: Column name for questions
        output_column: Column name for answers

    Returns:
        Number of examples converted
    """
    examples = []

    # Read CSV
    print(f"Reading {csv_file}...")
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Verify columns exist
        if input_column not in reader.fieldnames:
            raise ValueError(f"Column '{input_column}' not found in CSV")
        if output_column not in reader.fieldnames:
            raise ValueError(f"Column '{output_column}' not found in CSV")

        for row in reader:
            question = row[input_column].strip()
            answer = row[output_column].strip()

            # Skip empty rows
            if question and answer:
                examples.append({
                    "input": question,
                    "output": answer
                })

    # Write JSONL
    print(f"Writing {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Converted {len(examples)} examples")
    return len(examples)


# Usage
if __name__ == "__main__":
    csv_to_training_data(
        csv_file="company_faq.csv",
        output_file="training_data.jsonl",
        input_column="question",
        output_column="answer"
    )
```

### Advanced: Handle Different Column Names

```python
# flexible_csv_loader.py - Handle any CSV structure

import csv
import json


def load_csv_flexible(
    csv_file: str,
    column_mapping: dict,
    output_file: str
) -> int:
    """
    Load CSV with flexible column mapping.

    Args:
        csv_file: Path to CSV
        column_mapping: {"input": "your_question_col", "output": "your_answer_col"}
        output_file: Output JSONL path

    Example:
        load_csv_flexible(
            "data.csv",
            {"input": "user_query", "output": "bot_response"},
            "training.jsonl"
        )
    """
    examples = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            example = {}
            for target_key, source_col in column_mapping.items():
                if source_col in row:
                    example[target_key] = row[source_col].strip()

            if example.get("input") and example.get("output"):
                examples.append(example)

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Loaded {len(examples)} examples from {csv_file}")
    return len(examples)


# Example: Your CSV has columns "Question Text" and "Response"
load_csv_flexible(
    "weird_columns.csv",
    {"input": "Question Text", "output": "Response"},
    "training.jsonl"
)
```

---

## Method 2: Load from Database

Got existing FAQ data in a database? Load it directly!

### Using TinyForgeAI's DB Connector

```python
# load_from_database.py - Load from any SQL database

from connectors.db_connector import DBConnector


def load_from_database(
    db_url: str,
    query: str,
    column_mapping: dict,
    output_file: str
) -> int:
    """
    Load training data from a database.

    Args:
        db_url: Database connection string
        query: SQL query to get data
        column_mapping: {"input": "question_col", "output": "answer_col"}
        output_file: Output JSONL path
    """
    # Connect to database
    db = DBConnector(db_url=db_url)

    if not db.test_connection():
        raise ConnectionError(f"Cannot connect to {db_url}")

    print(f"Connected to database")

    # Load data
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in db.stream_samples(query, column_mapping):
            f.write(json.dumps(sample) + '\n')
            count += 1

    print(f"✅ Loaded {count} examples from database")
    return count


# Example: SQLite database
import json

load_from_database(
    db_url="sqlite:///company_data.db",
    query="SELECT question, answer FROM faq_table WHERE active = 1",
    column_mapping={"input": "question", "output": "answer"},
    output_file="training_data.jsonl"
)

# Example: PostgreSQL database
load_from_database(
    db_url="postgresql://user:pass@localhost/mydb",
    query="SELECT user_question, bot_answer FROM conversations",
    column_mapping={"input": "user_question", "output": "bot_answer"},
    output_file="training_data.jsonl"
)
```

### Create a Test Database

```python
# create_test_db.py - Create a sample SQLite database

import sqlite3

# Create database
conn = sqlite3.connect('company_data.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS faq_table (
        id INTEGER PRIMARY KEY,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        category TEXT,
        active INTEGER DEFAULT 1
    )
''')

# Insert sample data
faq_data = [
    ("What are your hours?", "Monday-Friday 9 AM to 6 PM EST.", "general", 1),
    ("How do I contact support?", "Email support@company.com", "support", 1),
    ("What's your return policy?", "30-day money-back guarantee", "policy", 1),
    ("Old question", "Old answer", "deprecated", 0),  # inactive
]

cursor.executemany(
    "INSERT INTO faq_table (question, answer, category, active) VALUES (?, ?, ?, ?)",
    faq_data
)

conn.commit()
conn.close()

print("✅ Created company_data.db with sample FAQ data")
```

---

## Method 3: Load from Documents

Turn your PDFs, Word docs, and text files into training data!

### Using TinyForgeAI's File Ingest

```python
# load_from_documents.py - Load from PDF, DOCX, TXT

from connectors.file_ingest import ingest_file, ingest_folder
import json


def documents_to_training_data(
    folder_path: str,
    output_file: str,
    chunk_size: int = 500
) -> int:
    """
    Convert documents to training data.

    For documents, we create Q&A pairs where:
    - Input: A chunk of text
    - Output: The same text (for retrieval) or summary

    Args:
        folder_path: Folder containing documents
        output_file: Output JSONL path
        chunk_size: Characters per chunk
    """
    examples = []

    # Ingest all files
    print(f"Processing documents in {folder_path}...")
    chunks = ingest_folder(folder_path, chunk_size=chunk_size)

    for chunk in chunks:
        # For retrieval: input = query-like text, output = full chunk
        # Create a simple "question" from the first sentence
        first_sentence = chunk['content'].split('.')[0] + '?'

        examples.append({
            "input": first_sentence.strip(),
            "output": chunk['content'],
            "metadata": {
                "source": chunk.get('source', 'unknown'),
                "chunk_id": chunk.get('chunk_id', 0)
            }
        })

    # Write JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Created {len(examples)} training examples from documents")
    return len(examples)


# Usage
documents_to_training_data(
    folder_path="./documents",
    output_file="doc_training_data.jsonl",
    chunk_size=500
)
```

### Simple Text File Loading

```python
# load_text_files.py - Simple text file loading

import os
import json
from pathlib import Path


def load_text_files(folder_path: str, output_file: str) -> int:
    """Load all .txt and .md files from a folder."""
    examples = []
    folder = Path(folder_path)

    for file_path in folder.glob("**/*.txt"):
        print(f"  Reading {file_path.name}...")
        content = file_path.read_text(encoding='utf-8')

        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        for i, para in enumerate(paragraphs):
            if len(para) > 50:  # Skip very short paragraphs
                examples.append({
                    "input": f"What does the document say about this topic?",
                    "output": para,
                    "source": file_path.name
                })

    # Also process markdown files
    for file_path in folder.glob("**/*.md"):
        print(f"  Reading {file_path.name}...")
        content = file_path.read_text(encoding='utf-8')

        # Split by headers
        sections = content.split('\n#')
        for section in sections:
            if section.strip():
                lines = section.strip().split('\n')
                title = lines[0].strip('# ')
                body = '\n'.join(lines[1:]).strip()

                if body:
                    examples.append({
                        "input": f"What is {title}?",
                        "output": body,
                        "source": file_path.name
                    })

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Loaded {len(examples)} examples from text files")
    return len(examples)


# Usage
load_text_files("./docs", "text_training_data.jsonl")
```

---

## Method 4: Load from APIs

Pull data from external services!

### Using TinyForgeAI's API Connector

```python
# load_from_api.py - Load from REST API

from connectors.api_connector import APIConnector
import json


def load_from_api(
    base_url: str,
    endpoint: str,
    field_mapping: dict,
    output_file: str,
    headers: dict = None
) -> int:
    """
    Load training data from a REST API.

    Args:
        base_url: API base URL
        endpoint: API endpoint
        field_mapping: {"input": "api_field_name", "output": "api_field_name"}
        output_file: Output JSONL path
        headers: Optional headers (auth, etc.)
    """
    api = APIConnector(
        base_url=base_url,
        headers=headers or {}
    )

    # Fetch data
    print(f"Fetching from {base_url}{endpoint}...")
    response = api.fetch(endpoint)

    if not response:
        raise ValueError("API returned no data")

    # Handle different response formats
    if isinstance(response, list):
        items = response
    elif isinstance(response, dict):
        # Try common keys
        items = response.get('data') or response.get('items') or response.get('results') or [response]
    else:
        items = [response]

    # Convert to training format
    examples = []
    for item in items:
        input_field = field_mapping.get("input")
        output_field = field_mapping.get("output")

        if input_field in item and output_field in item:
            examples.append({
                "input": str(item[input_field]),
                "output": str(item[output_field])
            })

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Loaded {len(examples)} examples from API")
    return len(examples)


# Example: Load from a public FAQ API
load_from_api(
    base_url="https://api.example.com",
    endpoint="/v1/faqs",
    field_mapping={"input": "question", "output": "answer"},
    output_file="api_training_data.jsonl",
    headers={"Authorization": "Bearer your-token"}
)
```

---

## Method 5: Manual Data Entry Helper

Sometimes you need to create data from scratch!

```python
# data_entry_helper.py - Interactive data entry

import json


def interactive_data_entry(output_file: str):
    """
    Interactive tool to create training data.

    Great for:
    - Starting from scratch
    - Adding examples your bot got wrong
    - Domain expert knowledge entry
    """
    print("=" * 60)
    print("  Training Data Entry Tool")
    print("  Type 'done' when finished, 'show' to see entries")
    print("=" * 60)

    examples = []

    # Load existing data if file exists
    try:
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        print(f"Loaded {len(examples)} existing examples")
    except FileNotFoundError:
        print("Starting fresh dataset")

    while True:
        print(f"\n--- Entry #{len(examples) + 1} ---")

        # Get question
        question = input("Question (or 'done'/'show'): ").strip()

        if question.lower() == 'done':
            break

        if question.lower() == 'show':
            print(f"\nCurrent entries ({len(examples)}):")
            for i, ex in enumerate(examples[-5:], 1):  # Show last 5
                print(f"  {i}. Q: {ex['input'][:50]}...")
            continue

        if not question:
            continue

        # Get answer
        answer = input("Answer: ").strip()

        if not answer:
            print("Skipped (no answer provided)")
            continue

        # Add example
        examples.append({
            "input": question,
            "output": answer
        })
        print(f"✓ Added! Total: {len(examples)} examples")

    # Save all examples
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n✅ Saved {len(examples)} examples to {output_file}")


# Run it
if __name__ == "__main__":
    interactive_data_entry("my_training_data.jsonl")
```

---

## Data Quality Pipeline

### Complete Data Preparation Script

```python
# prepare_data.py - Complete data preparation pipeline

import json
import csv
from pathlib import Path
from collections import Counter


class DataPreparer:
    """Complete data preparation pipeline for TinyForgeAI."""

    def __init__(self):
        self.examples = []
        self.stats = Counter()

    def load_csv(self, filepath: str, input_col: str, output_col: str):
        """Load from CSV file."""
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get(input_col, '').strip()
                a = row.get(output_col, '').strip()
                if q and a:
                    self.examples.append({"input": q, "output": a})
                    count += 1
        self.stats['csv'] += count
        print(f"  CSV: Loaded {count} examples")

    def load_jsonl(self, filepath: str):
        """Load from existing JSONL file."""
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get('input') and data.get('output'):
                        self.examples.append({
                            "input": data['input'],
                            "output": data['output']
                        })
                        count += 1
        self.stats['jsonl'] += count
        print(f"  JSONL: Loaded {count} examples")

    def add_manual(self, question: str, answer: str):
        """Add a single example manually."""
        self.examples.append({"input": question, "output": answer})
        self.stats['manual'] += 1

    def deduplicate(self):
        """Remove duplicate questions."""
        seen = set()
        unique = []
        duplicates = 0

        for ex in self.examples:
            key = ex['input'].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(ex)
            else:
                duplicates += 1

        self.examples = unique
        print(f"  Removed {duplicates} duplicates")

    def validate(self):
        """Validate data quality."""
        issues = []

        for i, ex in enumerate(self.examples):
            # Check length
            if len(ex['input']) < 3:
                issues.append(f"#{i}: Input too short")
            if len(ex['output']) < 3:
                issues.append(f"#{i}: Output too short")
            if len(ex['output']) > 2000:
                issues.append(f"#{i}: Output very long ({len(ex['output'])} chars)")

        if issues:
            print(f"  ⚠️  Found {len(issues)} potential issues:")
            for issue in issues[:5]:
                print(f"     - {issue}")
            if len(issues) > 5:
                print(f"     ... and {len(issues) - 5} more")
        else:
            print("  ✓ No issues found")

    def augment_variations(self):
        """Add simple variations to increase data diversity."""
        new_examples = []

        for ex in self.examples:
            # Add lowercase version if different
            lower_input = ex['input'].lower()
            if lower_input != ex['input']:
                new_examples.append({
                    "input": lower_input,
                    "output": ex['output']
                })

            # Add without question mark if present
            if ex['input'].endswith('?'):
                new_examples.append({
                    "input": ex['input'][:-1],
                    "output": ex['output']
                })

        self.examples.extend(new_examples)
        print(f"  Added {len(new_examples)} variations")

    def split_train_test(self, test_ratio: float = 0.1):
        """Split into training and test sets."""
        import random
        random.shuffle(self.examples)

        split_idx = int(len(self.examples) * (1 - test_ratio))
        train = self.examples[:split_idx]
        test = self.examples[split_idx:]

        return train, test

    def save(self, filepath: str, data: list = None):
        """Save to JSONL file."""
        data = data or self.examples
        with open(filepath, 'w', encoding='utf-8') as f:
            for ex in data:
                f.write(json.dumps(ex) + '\n')
        print(f"  Saved {len(data)} examples to {filepath}")

    def summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 50)
        print("Data Preparation Summary")
        print("=" * 50)
        print(f"Total examples: {len(self.examples)}")
        print(f"Sources: {dict(self.stats)}")

        if self.examples:
            avg_input = sum(len(ex['input']) for ex in self.examples) / len(self.examples)
            avg_output = sum(len(ex['output']) for ex in self.examples) / len(self.examples)
            print(f"Avg input length: {avg_input:.0f} chars")
            print(f"Avg output length: {avg_output:.0f} chars")


# Example usage
if __name__ == "__main__":
    print("Starting data preparation...")
    print("-" * 50)

    prep = DataPreparer()

    # Load from multiple sources
    try:
        prep.load_csv("data/faq.csv", "question", "answer")
    except FileNotFoundError:
        print("  CSV: File not found (skipping)")

    try:
        prep.load_jsonl("data/existing_data.jsonl")
    except FileNotFoundError:
        print("  JSONL: File not found (skipping)")

    # Add some manual examples
    prep.add_manual("Hello", "Hello! How can I help you today?")
    prep.add_manual("Thanks", "You're welcome!")

    # Process
    print("\nProcessing...")
    prep.deduplicate()
    prep.validate()
    prep.augment_variations()

    # Split and save
    print("\nSaving...")
    train, test = prep.split_train_test(test_ratio=0.1)
    prep.save("train.jsonl", train)
    prep.save("test.jsonl", test)

    # Summary
    prep.summary()
```

---

## Quick Reference: Data Sources

```
┌─────────────────────────────────────────────────────────────┐
│                 Quick Reference: Load Data                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CSV:                                                       │
│    csv_to_training_data("data.csv", "output.jsonl",        │
│                         "question", "answer")               │
│                                                             │
│  Database:                                                  │
│    db = DBConnector("sqlite:///data.db")                   │
│    for sample in db.stream_samples(query, mapping):        │
│        ...                                                  │
│                                                             │
│  Documents:                                                 │
│    from connectors.file_ingest import ingest_folder        │
│    chunks = ingest_folder("./docs")                        │
│                                                             │
│  API:                                                       │
│    api = APIConnector("https://api.example.com")           │
│    data = api.fetch("/endpoint")                           │
│                                                             │
│  Manual:                                                    │
│    examples.append({"input": q, "output": a})              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Checkpoint Quiz

**1. Why do we need to deduplicate training data?**
<details>
<summary>Click for answer</summary>

Duplicates bias the model toward those specific examples. If "What are your hours?" appears 100 times but "How do I contact support?" appears only once, the model will be much better at the first question. We want balanced, diverse data.

</details>

**2. What's the benefit of splitting into train/test sets?**
<details>
<summary>Click for answer</summary>

The test set lets us evaluate the model on data it hasn't seen during training. This shows how well the model generalizes to new questions, not just memorized answers.

</details>

**3. When would you use the API connector vs CSV loader?**
<details>
<summary>Click for answer</summary>

- **CSV**: When you have static data exported from spreadsheets, one-time loads
- **API**: When you need live data from external services, regular updates, or the data is too large to export

</details>

---

## What's Next?

In **Module 7: Train Your First Model**, you'll:
- Use TinyForgeAI's trainer on your prepared data
- Train a real AI model (DistilBERT)
- Understand training parameters
- See your model learn!

**Your data is ready. Let's train!**

---

[← Back to Module 5](05-what-is-a-model.md) | [Continue to Module 7 →](07-train-your-model.md)
