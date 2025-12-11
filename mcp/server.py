"""
TinyForgeAI MCP Server

A Model Context Protocol server that exposes TinyForgeAI capabilities
to AI assistants like Claude, ChatGPT, and Gemini.

Features:
- Training job submission and monitoring
- Model inference
- Data connector operations
- Model registry browsing
- RAG document indexing and search

Usage:
    python -m mcp.server
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Resource,
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
        LoggingLevel,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)

# TinyForgeAI imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.tools.training import TrainingTools
from mcp.tools.inference import InferenceTools
from mcp.tools.connectors import ConnectorTools
from mcp.tools.model_zoo import ModelZooTools
from mcp.resources.models import ModelResources
from mcp.resources.jobs import JobResources

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tinyforge-mcp")

# Server instance
server = Server("tinyforge")

# Tool handlers
training_tools = TrainingTools()
inference_tools = InferenceTools()
connector_tools = ConnectorTools()
model_zoo_tools = ModelZooTools()

# Resource handlers
model_resources = ModelResources()
job_resources = JobResources()


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available MCP tools."""
    tools = []

    # Training tools
    tools.extend([
        Tool(
            name="train_model",
            description="Submit a new model training job. Trains a language model on your data using HuggingFace Transformers with optional LoRA fine-tuning.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to training data file (JSONL format with 'input' and 'output' fields)"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Base model to fine-tune (e.g., 'distilbert-base-uncased', 'gpt2', 'google/flan-t5-small')",
                        "default": "distilbert-base-uncased"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save trained model",
                        "default": "./output"
                    },
                    "num_epochs": {
                        "type": "integer",
                        "description": "Number of training epochs",
                        "default": 3
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Training batch size",
                        "default": 8
                    },
                    "use_lora": {
                        "type": "boolean",
                        "description": "Use LoRA for efficient fine-tuning (recommended)",
                        "default": True
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Simulate training without GPU (for testing)",
                        "default": False
                    }
                },
                "required": ["data_path"]
            }
        ),
        Tool(
            name="get_training_status",
            description="Get the status of a training job including progress, metrics, and logs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Training job ID"
                    }
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="list_training_jobs",
            description="List all training jobs with their status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["all", "running", "completed", "failed"],
                        "description": "Filter by job status",
                        "default": "all"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of jobs to return",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="cancel_training",
            description="Cancel a running training job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Training job ID to cancel"
                    }
                },
                "required": ["job_id"]
            }
        ),
    ])

    # Inference tools
    tools.extend([
        Tool(
            name="run_inference",
            description="Run inference on a trained model. Send text input and get model predictions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to trained model directory"
                    },
                    "input_text": {
                        "type": "string",
                        "description": "Text input for the model"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum output length",
                        "default": 256
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature (0.0-2.0)",
                        "default": 0.7
                    }
                },
                "required": ["model_path", "input_text"]
            }
        ),
        Tool(
            name="batch_inference",
            description="Run inference on multiple inputs at once.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to trained model directory"
                    },
                    "inputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text inputs"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum output length per input",
                        "default": 256
                    }
                },
                "required": ["model_path", "inputs"]
            }
        ),
    ])

    # Connector tools
    tools.extend([
        Tool(
            name="fetch_data",
            description="Fetch training data from a connector source (database, API, Google Docs, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "connector_type": {
                        "type": "string",
                        "enum": ["database", "api", "google_docs", "google_drive", "notion", "slack", "confluence", "file"],
                        "description": "Type of data connector to use"
                    },
                    "config": {
                        "type": "object",
                        "description": "Connector-specific configuration",
                        "properties": {
                            "connection_string": {"type": "string"},
                            "base_url": {"type": "string"},
                            "api_key": {"type": "string"},
                            "file_path": {"type": "string"},
                            "mock_mode": {"type": "boolean", "default": True}
                        }
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save fetched data (JSONL format)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of records to fetch",
                        "default": 1000
                    }
                },
                "required": ["connector_type", "output_path"]
            }
        ),
        Tool(
            name="ingest_files",
            description="Ingest documents (PDF, DOCX, TXT, MD) and convert to training data format.",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to file or directory to ingest"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save converted data (JSONL format)"
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Size of text chunks for splitting documents",
                        "default": 512
                    }
                },
                "required": ["input_path", "output_path"]
            }
        ),
        Tool(
            name="index_documents",
            description="Index documents for RAG (Retrieval-Augmented Generation) search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to documents to index"
                    },
                    "index_name": {
                        "type": "string",
                        "description": "Name for the document index",
                        "default": "default"
                    },
                    "embedding_model": {
                        "type": "string",
                        "description": "Sentence transformer model for embeddings",
                        "default": "sentence-transformers/all-MiniLM-L6-v2"
                    }
                },
                "required": ["input_path"]
            }
        ),
        Tool(
            name="search_documents",
            description="Search indexed documents using semantic search (RAG).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "index_name": {
                        "type": "string",
                        "description": "Name of document index to search",
                        "default": "default"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
    ])

    # Model Zoo tools
    tools.extend([
        Tool(
            name="list_models",
            description="List all pre-configured models in the Model Zoo with their capabilities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": ["all", "qa", "summarization", "classification", "sentiment", "code_generation", "conversation", "ner", "translation", "text_generation"],
                        "description": "Filter by task type",
                        "default": "all"
                    }
                }
            }
        ),
        Tool(
            name="get_model_info",
            description="Get detailed information about a specific model from the Model Zoo.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model ID (e.g., 'qa_flan_t5_small', 'sentiment_roberta')"
                    }
                },
                "required": ["model_id"]
            }
        ),
        Tool(
            name="export_model",
            description="Export a trained model to ONNX format for optimized inference.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to trained model"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path for exported ONNX model"
                    },
                    "quantize": {
                        "type": "boolean",
                        "description": "Apply quantization for smaller model size",
                        "default": False
                    }
                },
                "required": ["model_path", "output_path"]
            }
        ),
    ])

    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    logger.info(f"Tool called: {name} with args: {arguments}")

    try:
        # Training tools
        if name == "train_model":
            result = await training_tools.train_model(**arguments)
        elif name == "get_training_status":
            result = await training_tools.get_status(**arguments)
        elif name == "list_training_jobs":
            result = await training_tools.list_jobs(**arguments)
        elif name == "cancel_training":
            result = await training_tools.cancel_job(**arguments)

        # Inference tools
        elif name == "run_inference":
            result = await inference_tools.run_inference(**arguments)
        elif name == "batch_inference":
            result = await inference_tools.batch_inference(**arguments)

        # Connector tools
        elif name == "fetch_data":
            result = await connector_tools.fetch_data(**arguments)
        elif name == "ingest_files":
            result = await connector_tools.ingest_files(**arguments)
        elif name == "index_documents":
            result = await connector_tools.index_documents(**arguments)
        elif name == "search_documents":
            result = await connector_tools.search_documents(**arguments)

        # Model Zoo tools
        elif name == "list_models":
            result = await model_zoo_tools.list_models(**arguments)
        elif name == "get_model_info":
            result = await model_zoo_tools.get_model_info(**arguments)
        elif name == "export_model":
            result = await model_zoo_tools.export_model(**arguments)

        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    except Exception as e:
        logger.error(f"Tool error: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    resources = []

    # Model registry resource
    resources.append(Resource(
        uri="tinyforge://models/registry",
        name="Model Registry",
        description="Pre-configured models available in TinyForgeAI Model Zoo",
        mimeType="application/json"
    ))

    # Training jobs resource
    resources.append(Resource(
        uri="tinyforge://jobs/active",
        name="Active Training Jobs",
        description="Currently running and recently completed training jobs",
        mimeType="application/json"
    ))

    # Connectors resource
    resources.append(Resource(
        uri="tinyforge://connectors/available",
        name="Available Connectors",
        description="Data connectors available for fetching training data",
        mimeType="application/json"
    ))

    # Documentation resource
    resources.append(Resource(
        uri="tinyforge://docs/quickstart",
        name="Quick Start Guide",
        description="Getting started with TinyForgeAI",
        mimeType="text/markdown"
    ))

    return resources


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource by URI."""
    logger.info(f"Resource requested: {uri}")

    if uri == "tinyforge://models/registry":
        return json.dumps(await model_resources.get_registry(), indent=2)

    elif uri == "tinyforge://jobs/active":
        return json.dumps(await job_resources.get_active_jobs(), indent=2)

    elif uri == "tinyforge://connectors/available":
        return json.dumps(await connector_tools.get_available_connectors(), indent=2)

    elif uri == "tinyforge://docs/quickstart":
        return await model_resources.get_quickstart_docs()

    else:
        return json.dumps({"error": f"Unknown resource: {uri}"})


async def main():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        print("Error: MCP SDK not installed. Install with: pip install mcp")
        sys.exit(1)

    logger.info("Starting TinyForgeAI MCP Server...")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
