# TinyForgeAI

A lightweight platform for fine-tuning language models.

## Features

- **Dataset Loading**: Load and validate JSONL training data
- **Model Training**: Dry-run training with LoRA adapter support
- **ONNX Export**: Export models to ONNX format with quantization
- **Microservice Builder**: Package models into deployable inference services
- **Connectors**: Database, Google Docs, and file ingestion connectors
- **CLI Tool**: User-friendly `foremforge` command-line interface

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TinyForgeAI.git
cd TinyForgeAI

# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## CLI Usage

TinyForgeAI provides the `foremforge` CLI for common tasks:

### Initialize Project Structure

```bash
# Create standard project directories and starter files
foremforge init --yes
```

### Train a Model

```bash
# Dry-run training (validates data, creates stub artifact)
foremforge train --data examples/sample_qna.jsonl --out /tmp/tiny_model --dry-run

# With LoRA adapter
foremforge train --data examples/sample_qna.jsonl --out /tmp/tiny_model --dry-run --use-lora
```

### Export to Microservice

```bash
# Basic export
foremforge export --model /tmp/tiny_model/model_stub.json --out /tmp/tiny_service

# With ONNX export and overwrite existing
foremforge export --model /tmp/tiny_model/model_stub.json --out /tmp/tiny_service --overwrite --export-onnx
```

### Serve the Microservice

```bash
# Start the inference server
foremforge serve --dir /tmp/tiny_service --port 8000

# Preview command without starting server
foremforge serve --dir /tmp/tiny_service --dry-run
```

### Help

```bash
# Main help
foremforge --help

# Command-specific help
foremforge train --help
foremforge export --help
```

## Project Structure

```
TinyForgeAI/
├── backend/
│   ├── api/              # FastAPI application
│   ├── training/         # Training pipeline
│   └── exporter/         # Model export and packaging
├── connectors/           # Data source connectors
├── inference_server/     # Inference service template
├── cli/                  # CLI tool (foremforge)
├── examples/             # Sample data files
├── docs/                 # Documentation
├── tests/                # Test suite
└── docker/               # Docker configurations
```

## Documentation

- [Training Documentation](docs/training.md)
- [Connectors Documentation](docs/connectors.md)

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_cli.py
```

## License

MIT License
