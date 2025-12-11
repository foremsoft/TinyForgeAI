# Tutorial Sample Data

This folder contains sample data files for the TinyForgeAI hands-on tutorials.

## Files

| File | Description | Used In |
|------|-------------|---------|
| `sample_faqs.csv` | 15 FAQ question/answer pairs | Tutorial 01 - FAQ Bot |
| `sample_training_data.jsonl` | 25 training examples in JSONL format | Tutorial 03 - Training |

## Sample FAQ Data (CSV)

```csv
question,answer
What are your business hours?,We are open Monday through Friday...
How do I contact support?,You can email support@example.com...
```

**Usage:**
```python
import csv
with open('sample_faqs.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Q: {row['question']}")
        print(f"A: {row['answer']}")
```

## Sample Training Data (JSONL)

```json
{"input": "Hello", "output": "Hello! How can I help you today?"}
{"input": "What is TinyForgeAI?", "output": "TinyForgeAI is a platform..."}
```

**Usage:**
```python
import json
with open('sample_training_data.jsonl', 'r') as f:
    for line in f:
        example = json.loads(line)
        print(f"Input: {example['input']}")
        print(f"Output: {example['output']}")
```

## Creating Your Own Data

### From CSV:
```python
import csv
import json

with open('your_data.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    with open('training_data.jsonl', 'w') as jsonl_file:
        for row in reader:
            example = {
                "input": row["question"],
                "output": row["answer"]
            }
            jsonl_file.write(json.dumps(example) + "\n")
```

### From Database:
```python
from connectors.db_connector import DBConnector

db = DBConnector(db_url="sqlite:///your_database.db")
mapping = {"input": "question_column", "output": "answer_column"}

for sample in db.stream_samples("SELECT * FROM faq_table", mapping):
    print(sample)
```

## Data Quality Tips

1. **Be consistent**: Use the same format for all examples
2. **Be complete**: Provide full, helpful answers
3. **Be diverse**: Include variations of similar questions
4. **Check spelling**: Proofread your data
5. **Remove duplicates**: Each example should be unique
