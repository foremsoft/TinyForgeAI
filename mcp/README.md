# TinyForgeAI MCP Server

Model Context Protocol (MCP) integration for TinyForgeAI. Enables AI assistants like Claude, ChatGPT, and Gemini to interact with TinyForgeAI's training, inference, and data connector capabilities.

## What is MCP?

[Model Context Protocol](https://modelcontextprotocol.io/) is an open standard that allows AI assistants to connect to external tools and data sources. Think of it as a "USB-C port for AI" - a universal connector that lets AI models plug into various tools consistently.

## Features

The TinyForgeAI MCP server exposes:

### Tools

| Tool | Description |
|------|-------------|
| `train_model` | Submit model training jobs with configurable parameters |
| `get_training_status` | Monitor training progress and metrics |
| `list_training_jobs` | List all training jobs with filtering |
| `cancel_training` | Cancel running training jobs |
| `run_inference` | Run predictions on trained models |
| `batch_inference` | Batch prediction on multiple inputs |
| `fetch_data` | Fetch data from connectors (DB, API, Notion, etc.) |
| `ingest_files` | Convert documents to training data format |
| `index_documents` | Index documents for RAG search |
| `search_documents` | Semantic search across indexed documents |
| `list_models` | Browse Model Zoo |
| `get_model_info` | Get detailed model information |
| `export_model` | Export models to ONNX format |

### Resources

| Resource | URI | Description |
|----------|-----|-------------|
| Model Registry | `tinyforge://models/registry` | Pre-configured models in Model Zoo |
| Active Jobs | `tinyforge://jobs/active` | Training job status |
| Connectors | `tinyforge://connectors/available` | Available data connectors |
| Quick Start | `tinyforge://docs/quickstart` | Getting started guide |

## Installation

```bash
# Install MCP SDK
pip install mcp

# Install TinyForgeAI with MCP support
cd TinyForgeAI
pip install -e ".[all]"
```

## Usage

### With Claude Desktop

Add to your `claude_desktop_config.json`:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

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

Then restart Claude Desktop.

### With Claude Code

```bash
# In your TinyForgeAI directory
claude --mcp-server "python -m mcp.server"
```

### Standalone Server

```bash
python -m mcp.server
```

## Example Conversations

Once configured, you can ask Claude:

**Training:**
- "List available models in TinyForgeAI"
- "Train a Q&A model on my FAQ data at ./data/faqs.jsonl"
- "What's the status of my training job?"
- "Train a sentiment analysis model with 5 epochs"

**Inference:**
- "Run inference on my trained model with the input 'What is your return policy?'"
- "Test my model with these 5 questions"

**Data:**
- "Fetch data from my Notion workspace"
- "Index all PDFs in ./documents for search"
- "Search my documents for 'authentication flow'"

**Model Zoo:**
- "What models are available for text classification?"
- "Tell me about the qa_flan_t5_small model"
- "Export my model to ONNX format"

## Architecture

```
mcp/
├── __init__.py          # Package initialization
├── server.py            # Main MCP server
├── README.md            # This file
├── tools/
│   ├── __init__.py
│   ├── training.py      # Training job management
│   ├── inference.py     # Model inference
│   ├── connectors.py    # Data connector operations
│   └── model_zoo.py     # Model Zoo browsing
└── resources/
    ├── __init__.py
    ├── models.py        # Model registry resource
    └── jobs.py          # Job status resource
```

## Configuration

The MCP server uses TinyForgeAI's existing configuration. Environment variables:

```bash
# Optional: Set default model
export TINYFORGE_MODEL_NAME=distilbert-base-uncased

# Optional: Set output directory
export TINYFORGE_OUTPUT_DIR=./output

# Optional: Enable debug logging
export TINYFORGE_LOG_LEVEL=DEBUG
```

## Security Considerations

- The MCP server runs locally and has access to your filesystem
- API keys and tokens are passed through tool arguments, not stored
- Use mock mode (`mock_mode: true`) for connectors during testing
- Review tool calls before execution in sensitive environments

## Troubleshooting

### Server not starting

```bash
# Check MCP SDK installation
pip show mcp

# Run with debug output
python -m mcp.server 2>&1 | tee mcp.log
```

### Tools not appearing in Claude

1. Restart Claude Desktop after config changes
2. Check config file path is correct
3. Verify `cwd` points to TinyForgeAI directory

### Import errors

```bash
# Ensure all dependencies installed
pip install -e ".[all]"
```

## Development

### Adding a new tool

1. Create handler in `mcp/tools/`
2. Register in `server.py` `list_tools()`
3. Add call handler in `call_tool()`

### Adding a new resource

1. Create handler in `mcp/resources/`
2. Register in `server.py` `list_resources()`
3. Add read handler in `read_resource()`

## Links

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [TinyForgeAI Documentation](../docs/)
- [Claude Desktop MCP Setup](https://docs.anthropic.com/en/docs/claude-desktop/mcp)
