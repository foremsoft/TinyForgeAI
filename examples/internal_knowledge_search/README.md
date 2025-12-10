# Internal Knowledge Search Example

A complete implementation of an enterprise knowledge search system using TinyForgeAI's RAG capabilities.

## Overview

This example demonstrates how to build a semantic search system that:
- Indexes documents from multiple sources (Google Drive, Notion, Confluence, local files)
- Provides semantic search across your knowledge base
- Generates context-aware answers using fine-tuned models
- Exposes a REST API for integration with other tools

## Project Structure

```
internal_knowledge_search/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── configs/
│   └── search_config.yaml       # Search and indexing configuration
├── data/
│   └── sample_documents/        # Sample documents for testing
│       ├── company_policies.txt
│       ├── product_guide.txt
│       └── faq.txt
├── index_documents.py           # Document indexing script
├── search_service.py            # FastAPI search service
├── train_qa_model.py            # Optional: Fine-tune Q&A model
└── test_search.py               # Test the search system
```

## Quick Start

### 1. Install Dependencies

```bash
# From project root
pip install -e ".[rag]"

# Or install example-specific dependencies
pip install -r requirements.txt
```

### 2. Index Sample Documents

```bash
# Index the sample documents
python index_documents.py --source local --path ./data/sample_documents

# Or index from Google Drive (requires credentials)
python index_documents.py --source gdrive --folder-id YOUR_FOLDER_ID
```

### 3. Start the Search Service

```bash
python search_service.py --port 8001
```

### 4. Test the Search

```bash
# Run automated tests
python test_search.py --url http://localhost:8001

# Or query directly
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the vacation policy?"}'
```

## Configuration

Edit `configs/search_config.yaml` to customize:

```yaml
# Embedding model for semantic search
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2

# Chunking strategy
chunking:
  size: 500
  overlap: 50

# Search parameters
search:
  top_k: 5
  min_score: 0.3
```

## Data Sources

### Local Files
```bash
python index_documents.py --source local --path /path/to/docs
```

### Google Drive
```bash
python index_documents.py --source gdrive \
  --credentials ./credentials/google_service_account.json \
  --folder-id YOUR_FOLDER_ID
```

### Notion
```bash
python index_documents.py --source notion \
  --token YOUR_NOTION_TOKEN \
  --database-id YOUR_DATABASE_ID
```

### Confluence
```bash
python index_documents.py --source confluence \
  --url https://your-company.atlassian.net \
  --space-key DOCS
```

## API Endpoints

### Search Documents
```
POST /search
{
  "query": "your search query",
  "top_k": 5,
  "filters": {"department": "engineering"}
}
```

### Generate Answer
```
POST /answer
{
  "question": "What is the vacation policy?",
  "include_sources": true
}
```

### Index Status
```
GET /status
```

## Fine-Tuning (Optional)

For better answer generation, you can fine-tune a Q&A model:

```bash
# Prepare Q&A training data from your documents
python prepare_qa_data.py --index ./data/knowledge_index

# Train the model
python train_qa_model.py --config configs/qa_training.yaml
```

## Integration Examples

### Slack Bot
See `integrations/slack_bot.py` for a complete Slack integration.

### Web Interface
See `integrations/web_ui.html` for a simple search interface.

## Wiki Tutorial

This example accompanies the wiki tutorial:
[Tutorial: Internal Knowledge Search](https://github.com/foremsoft/TinyForgeAI/wiki/Tutorial-Internal-Knowledge-Search)

## License

Apache 2.0 - See LICENSE in project root.
