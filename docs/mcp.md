# MCP Integration Guide

This guide covers TinyForgeAI's integration with the Model Context Protocol (MCP), enabling AI assistants like Claude, ChatGPT, and Gemini to interact with your training pipeline.

## What is MCP?

[Model Context Protocol](https://modelcontextprotocol.io/) is an open standard developed by Anthropic that allows AI assistants to connect to external tools and data sources. Think of it as a universal adapter that lets AI models interact with your applications.

**Key Benefits:**
- **Universal**: Works with Claude, ChatGPT, Gemini, and other MCP-compatible assistants
- **Secure**: Runs locally, no data sent to external servers
- **Extensible**: Add custom tools and resources
- **Standard**: Industry-wide adoption (OpenAI, Google, Microsoft)

## Quick Start

### 1. Install Dependencies

```bash
# Install MCP SDK
pip install mcp

# Install TinyForgeAI with all features
pip install -e ".[all]"
```

### 2. Configure Claude Desktop

Add to your Claude Desktop config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
    "mcpServers": {
        "tinyforge": {
            "command": "python",
            "args": ["-m", "mcp.server"],
            "cwd": "/path/to/TinyForgeAI"
        }
    }
}
```

### 3. Restart Claude Desktop

After saving the config, restart Claude Desktop. You should see TinyForgeAI tools available.

### 4. Start Using

Ask Claude:
- "What models are available in TinyForgeAI?"
- "Train a Q&A model on my FAQ data"
- "Search my documents for authentication"

## Available Tools

### Training Tools

| Tool | Description | Example |
|------|-------------|---------|
| `train_model` | Submit training job | "Train a model on ./data/faqs.jsonl" |
| `get_training_status` | Check job progress | "What's the status of job abc123?" |
| `list_training_jobs` | List all jobs | "Show my training jobs" |
| `cancel_training` | Cancel a job | "Cancel training job abc123" |

**Train Model Parameters:**
```json
{
    "data_path": "./data/training.jsonl",  // Required
    "model_name": "distilbert-base-uncased",
    "output_dir": "./output",
    "num_epochs": 3,
    "batch_size": 8,
    "use_lora": true,
    "dry_run": false
}
```

### Inference Tools

| Tool | Description | Example |
|------|-------------|---------|
| `run_inference` | Single prediction | "Ask my model: What is your return policy?" |
| `batch_inference` | Multiple predictions | "Test my model with these 5 questions" |

**Inference Parameters:**
```json
{
    "model_path": "./output/my_model",
    "input_text": "What is your return policy?",
    "max_length": 256,
    "temperature": 0.7
}
```

### Data Connector Tools

| Tool | Description | Example |
|------|-------------|---------|
| `fetch_data` | Fetch from connectors | "Get data from my Notion workspace" |
| `ingest_files` | Process documents | "Convert PDFs in ./docs to training data" |
| `index_documents` | Index for RAG | "Index all documents in ./knowledge" |
| `search_documents` | Semantic search | "Search for authentication flow" |

**Supported Connectors:**
- `database` - SQLite, PostgreSQL, MySQL
- `api` - REST APIs with auth and pagination
- `google_docs` - Google Docs (mock mode available)
- `google_drive` - Google Drive files
- `notion` - Notion pages and databases
- `slack` - Slack messages
- `confluence` - Confluence pages
- `file` - Local files (PDF, DOCX, TXT, MD)

### Model Zoo Tools

| Tool | Description | Example |
|------|-------------|---------|
| `list_models` | Browse available models | "What models can I use for classification?" |
| `get_model_info` | Get model details | "Tell me about qa_flan_t5_small" |
| `export_model` | Export to ONNX | "Export my model to ONNX format" |

## Available Resources

Resources are read-only data that AI assistants can access:

| Resource URI | Description |
|--------------|-------------|
| `tinyforge://models/registry` | Complete model registry with configurations |
| `tinyforge://jobs/active` | Active and recent training jobs |
| `tinyforge://connectors/available` | List of available data connectors |
| `tinyforge://docs/quickstart` | Quick start documentation |

## Example Conversations

### Training a Model

```
User: I have FAQ data in ./data/faqs.jsonl. Can you train a Q&A model?

Claude: I'll train a Q&A model on your FAQ data. Let me submit a training job.

[Calls train_model with data_path="./data/faqs.jsonl", model_name="google/flan-t5-small"]

Training job submitted! Job ID: abc123. The model will be saved to ./output.
Would you like me to monitor the progress?
```

### Searching Documents

```
User: Index all the PDFs in ./docs and search for how to reset passwords.

Claude: I'll index your documents and search for password reset information.

[Calls index_documents with input_path="./docs"]
[Calls search_documents with query="how to reset password"]

Found 3 relevant sections:
1. (Score: 0.89) "To reset your password, click the 'Forgot Password' link..."
2. (Score: 0.76) "Password requirements: minimum 8 characters..."
3. (Score: 0.71) "If you're locked out, contact support..."
```

### Exploring Models

```
User: What models are good for sentiment analysis?

Claude: Let me check the Model Zoo.

[Calls list_models with task_type="sentiment"]

For sentiment analysis, I recommend:

1. **sentiment_roberta** (125M params)
   - Pre-trained on Twitter data
   - Great for social media sentiment
   - Use LoRA for efficient fine-tuning

2. **classification_distilbert** (66M params)
   - Faster inference
   - Good for general classification
   - No LoRA needed

Would you like more details on either model?
```

## Configuration

### Environment Variables

```bash
# Default model for training
export TINYFORGE_MODEL_NAME=distilbert-base-uncased

# Default output directory
export TINYFORGE_OUTPUT_DIR=./output

# Logging level
export TINYFORGE_LOG_LEVEL=INFO
```

### Mock Mode

For testing without real credentials, connectors support mock mode:

```json
{
    "connector_type": "notion",
    "config": {
        "mock_mode": true
    }
}
```

## Troubleshooting

### MCP Server Not Starting

```bash
# Check MCP installation
pip show mcp

# Run server directly to see errors
python -m mcp.server
```

### Tools Not Appearing in Claude

1. Verify config file path is correct
2. Check `cwd` points to TinyForgeAI directory
3. Restart Claude Desktop completely
4. Check Claude Desktop logs

### Import Errors

```bash
# Ensure all dependencies are installed
pip install -e ".[all]"

# Check specific dependencies
pip show transformers torch sentence-transformers
```

### Training Fails

1. Verify data file exists and is valid JSONL
2. Check GPU availability for real training
3. Use `dry_run: true` for testing
4. Check disk space for model output

## Security Considerations

- **Local execution**: MCP server runs locally, data stays on your machine
- **No stored credentials**: API keys passed per-request, not stored
- **File system access**: Server can access files in TinyForgeAI directory
- **Mock mode**: Use for testing without exposing real credentials

## Extending the MCP Server

### Adding a Custom Tool

1. Create handler in `mcp/tools/`:

```python
# mcp/tools/custom.py
class CustomTools:
    async def my_tool(self, param: str) -> dict:
        return {"result": f"Processed: {param}"}
```

2. Register in `mcp/server.py`:

```python
from mcp.tools.custom import CustomTools
custom_tools = CustomTools()

@server.list_tools()
async def list_tools():
    tools.append(Tool(
        name="my_tool",
        description="My custom tool",
        inputSchema={...}
    ))

@server.call_tool()
async def call_tool(name, args):
    if name == "my_tool":
        return await custom_tools.my_tool(**args)
```

### Adding a Custom Resource

```python
# In mcp/server.py

@server.list_resources()
async def list_resources():
    resources.append(Resource(
        uri="tinyforge://custom/data",
        name="Custom Data",
        mimeType="application/json"
    ))

@server.read_resource()
async def read_resource(uri):
    if uri == "tinyforge://custom/data":
        return json.dumps({"custom": "data"})
```

## Links

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Claude Desktop MCP Setup](https://docs.anthropic.com/en/docs/claude-desktop/mcp)
- [TinyForgeAI MCP README](../mcp/README.md)
