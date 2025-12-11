# Tutorial 05: MCP Integration - AI-Powered Training

**Time:** 15 minutes
**Difficulty:** Beginner
**Prerequisites:** TinyForgeAI installed, Claude Desktop (optional)

## What You'll Learn

- What MCP (Model Context Protocol) is
- How to set up TinyForgeAI with Claude Desktop
- How to train models using natural language
- How to search documents with AI assistance

## What is MCP?

Model Context Protocol (MCP) is like a "USB-C port for AI" - it lets AI assistants (Claude, ChatGPT, Gemini) connect to external tools. With TinyForgeAI's MCP server, you can:

- Train models by asking: *"Train a Q&A model on my FAQ data"*
- Search documents: *"Find information about password resets"*
- Monitor jobs: *"What's the status of my training?"*

No coding required!

## Step 1: Install MCP SDK

```bash
# Install MCP Python SDK
pip install mcp

# Verify installation
python -c "import mcp; print('MCP installed!')"
```

## Step 2: Test the MCP Server

Let's make sure the server works:

```bash
cd /path/to/TinyForgeAI

# Run the MCP server (it will wait for connections)
python -m mcp.server
```

You should see: `Starting TinyForgeAI MCP Server...`

Press `Ctrl+C` to stop.

## Step 3: Configure Claude Desktop

### Find Your Config File

| OS | Location |
|----|----------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

### Add TinyForgeAI Server

Create or edit the config file:

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

**Important:** Replace `/path/to/TinyForgeAI` with your actual path!

### Restart Claude Desktop

Completely quit and reopen Claude Desktop for changes to take effect.

## Step 4: Try It Out!

Open Claude Desktop and try these prompts:

### Browse Models

```
What models are available in TinyForgeAI for question answering?
```

Claude will call `list_models` and show you options like:
- qa_flan_t5_small (80M params)
- qa_flan_t5_base (250M params)

### Train a Model (Dry Run)

```
Train a Q&A model on the demo dataset at examples/data/demo_dataset.jsonl using dry run mode.
```

Claude will:
1. Call `train_model` with your parameters
2. Return a job ID
3. You can ask "What's the status of job [ID]?"

### Search Documents

First, let's index some documents:

```
Index all files in the examples/tutorial_data folder for search.
```

Then search:

```
Search my indexed documents for "how to train a model"
```

## Step 5: Real-World Example

Let's do a complete workflow:

### 1. Prepare Your Data

Create a file `my_faqs.jsonl`:

```json
{"input": "What are your hours?", "output": "We're open Monday-Friday, 9am-5pm."}
{"input": "How do I return an item?", "output": "Visit our returns page or call customer service."}
{"input": "Do you ship internationally?", "output": "Yes, we ship to over 50 countries."}
```

### 2. Ask Claude to Train

```
I have FAQ data in my_faqs.jsonl. Can you train a Q&A model on it?
Use the qa_flan_t5_small model with 3 epochs. Do a dry run first.
```

### 3. Monitor Progress

```
How is my training job doing?
```

### 4. Test the Model

```
Run inference on my trained model with the question "What time do you open?"
```

## Available Commands

Here's a quick reference of what you can ask Claude:

| Task | Example Prompt |
|------|---------------|
| List models | "What models can I use for sentiment analysis?" |
| Train model | "Train a model on data.jsonl with 5 epochs" |
| Check status | "What's the status of training job abc123?" |
| Cancel job | "Cancel my training job" |
| Run inference | "Ask my model: What is your return policy?" |
| Fetch data | "Get data from my Notion workspace" |
| Index docs | "Index PDFs in ./documents for search" |
| Search | "Search for authentication in my docs" |
| Export model | "Export my model to ONNX format" |

## Troubleshooting

### Claude doesn't see TinyForgeAI tools

1. Check config file path is correct
2. Verify `cwd` path exists
3. Restart Claude Desktop completely
4. Check Claude Desktop logs for errors

### "MCP not installed" error

```bash
pip install mcp
pip show mcp  # Verify installation
```

### Training fails

1. Check data file exists
2. Verify JSONL format is correct
3. Try `dry_run: true` first
4. Check disk space

## What's Next?

- [Deploy Your Model](04-deploy-your-project.md) - Put your trained model online
- [MCP Full Guide](../../mcp.md) - Complete MCP documentation
- [Connectors Guide](../../connectors.md) - Connect to more data sources

## Summary

You learned how to:
- Install and configure MCP for TinyForgeAI
- Use natural language to train models
- Search documents with AI assistance
- Monitor training jobs conversationally

MCP makes TinyForgeAI accessible without writing code - just describe what you want!
