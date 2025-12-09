# TinyForgeAI Connectors Documentation

This document describes the data source connectors available in TinyForgeAI for loading training data from various sources.

## Database Connector

The database connector allows you to stream training samples directly from SQL databases. It uses Python's built-in `sqlite3` module for SQLite databases and provides a clean interface for converting database rows to training samples.

### Configuration

Database connection settings are configured via environment variables or the `DBSettings` class:

```python
from connectors.db_config import db_settings

# Default: sqlite:///:memory:
print(db_settings.DB_URL)
```

Set the `DB_URL` environment variable to configure your database:

```bash
# SQLite file database
export DB_URL="sqlite:///./data/training.db"

# In-memory SQLite
export DB_URL="sqlite:///:memory:"
```

### Programmatic Usage

```python
from connectors.db_connector import DBConnector

# Create connector (uses DB_URL from environment by default)
conn = DBConnector(db_url="sqlite:///./data/qa.db")

# Test connection
if conn.test_connection():
    print("Connected successfully!")

# Stream training samples
query = "SELECT question, answer FROM qa_pairs"
mapping = {"input": "question", "output": "answer"}

for sample in conn.stream_samples(query, mapping):
    print(sample)
    # Output:
    # {
    #     "input": "How do I reset my password?",
    #     "output": "Go to Settings > Reset Password.",
    #     "metadata": {"source": "db", "raw_row": {...}}
    # }
```

### Column Mapping

The `mapping` parameter tells the connector which database columns correspond to the training sample fields:

```python
# Map 'question' column to 'input', 'answer' column to 'output'
mapping = {"input": "question", "output": "answer"}

# Using SQL aliases
query = "SELECT q AS question, a AS answer FROM faq"
mapping = {"input": "question", "output": "answer"}
```

### CLI Usage

The connector includes a CLI for quick testing and data extraction:

```bash
# Stream samples as JSONL
python connectors/cli.py db-stream \
    --query "SELECT question AS q, answer AS a FROM qa" \
    --mapping '{"input":"q","output":"a"}'

# Limit output
python connectors/cli.py db-stream \
    --query "SELECT question, answer FROM qa" \
    --mapping '{"input":"question","output":"answer"}' \
    --limit 10

# Specify database URL
python connectors/cli.py db-stream \
    --db-url "sqlite:///./data/qa.db" \
    --query "SELECT question, answer FROM qa" \
    --mapping '{"input":"question","output":"answer"}'

# Test connection
python connectors/cli.py test-connection --db-url "sqlite:///./data/qa.db"
```

### Direct Script Usage

You can also use the db_connector.py script directly:

```bash
python connectors/db_connector.py \
    --query "SELECT question AS q, answer AS a FROM qa" \
    --mapping '{"input":"q","output":"a"}' \
    --limit 5
```

### Row Mapping Utility

The `row_to_sample` function converts individual database rows to training samples:

```python
from connectors.mappers import row_to_sample

row = {"question": "What is 2+2?", "answer": "4", "category": "math"}
mapping = {"input": "question", "output": "answer"}

sample = row_to_sample(row, mapping)
# {
#     "input": "What is 2+2?",
#     "output": "4",
#     "metadata": {
#         "source": "db",
#         "raw_row": {"question": "What is 2+2?", "answer": "4", "category": "math"}
#     }
# }
```

### Error Handling

The connector raises clear errors for common issues:

- `KeyError`: When required columns are missing from the row or mapping
- `sqlite3.OperationalError`: When database operations fail

```python
# Missing column in mapping
try:
    row_to_sample(row, {"input": "nonexistent"})
except KeyError as e:
    print(f"Mapping error: {e}")
```

### Local Development with SQLite

For local development and testing, SQLite is the default database. Create a test database:

```python
import sqlite3

conn = sqlite3.connect("./data/test.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE qa (
        question TEXT,
        answer TEXT
    )
""")
cursor.execute("INSERT INTO qa VALUES (?, ?)",
               ("How do I reset my password?", "Go to Settings > Reset Password."))
conn.commit()
conn.close()
```

Then stream from it:

```bash
export DB_URL="sqlite:///./data/test.db"
python connectors/cli.py db-stream \
    --query "SELECT question, answer FROM qa" \
    --mapping '{"input":"question","output":"answer"}'
```

## Google Docs Connector

The Google Docs connector allows you to fetch and extract text content from Google Docs. It includes a **mock mode** for offline development and testing.

### Mock Mode (Default)

By default, the connector runs in mock mode (`GOOGLE_OAUTH_DISABLED=true`). In this mode, it reads from local sample files instead of making API calls, enabling fully offline development and testing.

Mock mode uses sample files from `examples/google_docs_samples/`:
- `sample_doc1.txt` - Sample documentation content
- `sample_doc2.txt` - Sample FAQ content

### Configuration

Set the `GOOGLE_OAUTH_DISABLED` environment variable in your `.env` file:

```bash
# Mock mode (default) - uses local sample files
GOOGLE_OAUTH_DISABLED=true

# Real mode - uses Google Docs API (requires OAuth setup)
GOOGLE_OAUTH_DISABLED=false
```

### Programmatic Usage

```python
from connectors.google_docs_connector import fetch_doc_text, list_docs_in_folder

# Fetch document text (uses mock mode by default)
text = fetch_doc_text("sample_doc1")
print(text)

# List available documents
docs = list_docs_in_folder("any_folder_id")
for doc in docs:
    print(f"{doc['id']}: {doc['title']}")
```

### CLI Usage

```bash
# Fetch a sample document (mock mode)
python connectors/google_docs_connector.py --doc-id sample_doc1

# List available sample documents
python connectors/google_docs_connector.py --doc-id sample_doc1 --list-samples
```

### Setting Up Google OAuth (Real Mode)

To use the connector with real Google Docs, you need to set up OAuth credentials:

1. **Create a Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable the Google Docs API**
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Docs API" and enable it
   - Also enable "Google Drive API" if you want to list documents in folders

3. **Create OAuth Credentials**
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Select "Desktop application" as the application type
   - Download the credentials JSON file

4. **Install Required Dependencies**
   ```bash
   pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
   ```

5. **Set Environment Variable**
   ```bash
   export GOOGLE_OAUTH_DISABLED=false
   ```

6. **Implement OAuth Flow**
   - The current implementation provides a placeholder for real API calls
   - See `connectors/google_docs_connector.py` for implementation guidance

### Text Utilities

The connector includes utility functions for text processing:

```python
from connectors.google_utils import extract_text_from_html, normalize_text

# Strip HTML tags
html = "<p>Hello <b>world</b>!</p>"
text = extract_text_from_html(html)  # "Hello world!"

# Normalize whitespace
messy = "Too   many    spaces\n\n\n\nand lines"
clean = normalize_text(messy)  # "Too many spaces\n\nand lines"
```

### Error Handling

```python
from connectors.google_docs_connector import fetch_doc_text

try:
    text = fetch_doc_text("nonexistent_doc")
except FileNotFoundError as e:
    print(f"Document not found: {e}")
except RuntimeError as e:
    print(f"API error: {e}")
```

## File Ingestion Connector

The file ingestion connector extracts text content from various document formats including TXT, Markdown, DOCX, and PDF files.

### Supported Formats

| Format | Extension | Dependency | Notes |
|--------|-----------|------------|-------|
| Plain Text | `.txt` | None (built-in) | UTF-8 encoding by default |
| Markdown | `.md` | None (built-in) | Returns raw markdown (no rendering) |
| Word Document | `.docx` | python-docx | Extracts paragraph text |
| PDF | `.pdf` | PyMuPDF or pdfminer.six | PyMuPDF preferred for speed |

### Installation

TXT and MD files work out of the box. For DOCX and PDF support, install optional dependencies:

```bash
# For DOCX support
pip install python-docx

# For PDF support (choose one)
pip install PyMuPDF        # Recommended: faster, more accurate
pip install pdfminer.six   # Alternative: pure Python
```

### Programmatic Usage

```python
from connectors.file_ingest import ingest_file, get_supported_formats, check_dependencies

# Ingest any supported file
text = ingest_file("path/to/document.txt")
text = ingest_file("path/to/document.md")
text = ingest_file("path/to/document.docx")
text = ingest_file("path/to/document.pdf")

# Check what formats are available
formats = get_supported_formats()
print(formats[".pdf"]["available"])  # True if PDF lib installed

# Check which optional dependencies are installed
deps = check_dependencies()
print(deps)
# {'python-docx': True, 'PyMuPDF': True, 'pdfminer.six': False}
```

### Custom Encoding

For TXT and MD files, you can specify a custom encoding:

```python
# Read a file with Latin-1 encoding
text = ingest_file("legacy_doc.txt", encoding="latin-1")
```

### Error Handling

```python
from connectors.file_ingest import ingest_file

try:
    text = ingest_file("document.pdf")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Unsupported format: {e}")
except RuntimeError as e:
    print(f"Missing dependency: {e}")
```

### Sample Files

Sample files for testing are available in `examples/files/`:
- `sample.txt` - Plain text sample
- `sample.md` - Markdown sample with headings and formatting
- `sample.docx` - Word document sample
- `sample.pdf` - PDF document sample
