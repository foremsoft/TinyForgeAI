"""
MCP Model Resources

Resources for accessing model registry and documentation.
"""

from typing import Any
from mcp.tools.model_zoo import MODEL_REGISTRY


class ModelResources:
    """Model registry and documentation resources."""

    async def get_registry(self) -> dict[str, Any]:
        """Get the complete model registry."""
        models_by_task = {}

        for model_id, info in MODEL_REGISTRY.items():
            task_type = info["task_type"]
            if task_type not in models_by_task:
                models_by_task[task_type] = []

            models_by_task[task_type].append({
                "model_id": model_id,
                "name": info["name"],
                "base_model": info["base_model"],
                "parameters": info["parameters"],
                "description": info["description"],
                "use_lora": info["use_lora"],
                "recommended_epochs": info["recommended_epochs"],
                "recommended_batch_size": info["recommended_batch_size"]
            })

        return {
            "total_models": len(MODEL_REGISTRY),
            "task_types": list(models_by_task.keys()),
            "models_by_task": models_by_task
        }

    async def get_quickstart_docs(self) -> str:
        """Get quick start documentation."""
        return """# TinyForgeAI Quick Start Guide

## What is TinyForgeAI?

TinyForgeAI is a framework for training small, focused language models from your own data
and deploying them as microservices.

## Installation

```bash
# Basic installation
pip install -e .

# With training support
pip install -e ".[training]"

# With RAG support
pip install -e ".[rag]"

# Everything
pip install -e ".[all]"
```

## Quick Training Example

### Using CLI (Dry Run)
```bash
foremforge train --data examples/data/demo_dataset.jsonl --out ./model --dry-run
```

### Using Python
```python
from backend.training.real_trainer import RealTrainer, TrainingConfig

config = TrainingConfig(
    model_name="distilbert-base-uncased",
    output_dir="./my_model",
    num_epochs=3,
    use_lora=True,
)

trainer = RealTrainer(config)
trainer.train("your_data.jsonl")
```

## Using MCP with Claude

1. Install MCP SDK: `pip install mcp`

2. Add to Claude Desktop config:
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

3. Ask Claude to:
   - "List available models in TinyForgeAI"
   - "Train a Q&A model on my FAQ data"
   - "Search my indexed documents for X"

## Available Connectors

- **Database**: SQLite, PostgreSQL, MySQL
- **Google Docs**: Document extraction
- **Google Drive**: File fetching
- **Notion**: Pages and databases
- **Slack**: Channel messages
- **Confluence**: Wiki pages
- **REST API**: Any API with pagination
- **Files**: PDF, DOCX, TXT, MD

## Model Zoo

Pre-configured models for:
- Question Answering
- Summarization
- Classification
- Sentiment Analysis
- Code Generation
- Conversation
- Named Entity Recognition
- Translation
- Text Generation

Run `python -m model_zoo.cli list` to see all models.

## Documentation

- [Full Documentation](https://github.com/foremsoft/TinyForgeAI/tree/main/docs)
- [Tutorials](https://github.com/foremsoft/TinyForgeAI/tree/main/docs/tutorials)
- [API Reference](https://github.com/foremsoft/TinyForgeAI/blob/main/docs/api_reference.md)
"""
