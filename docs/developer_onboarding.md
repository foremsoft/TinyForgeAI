# TinyForgeAI — Developer Onboarding Guide

*Last updated: December 2024*

Welcome to TinyForgeAI! This guide will help you get up to speed with the codebase and start contributing.

## 1. Repository Structure

```
TinyForgeAI/
├── backend/
│   ├── dataset/           # JSONL schema and validation
│   │   └── schema.py      # Pydantic models for training data
│   ├── trainer/           # Training pipeline
│   │   └── stub_trainer.py # Dry-run trainer implementation
│   ├── exporter/          # Model export and packaging
│   │   ├── builder.py     # Microservice generator
│   │   └── onnx_export.py # ONNX export utility
│   └── api/               # FastAPI application (future)
├── connectors/            # Data source connectors
│   ├── db_connector.py    # SQLite/PostgreSQL/MySQL
│   ├── google_docs_connector.py # Google Docs (mock mode)
│   ├── google_utils.py    # OAuth utilities
│   ├── file_ingest.py     # File ingestion (PDF/DOCX/TXT)
│   └── file_helpers.py    # PDF/DOCX extraction helpers
├── inference_server/      # Inference service template
│   └── server.py          # FastAPI inference server
├── cli/                   # CLI tool
│   └── foremforge.py      # Click-based CLI commands
├── docker/                # Docker configurations
│   ├── Dockerfile.inference
│   ├── docker-compose.yml
│   └── README.md
├── examples/              # Sample data and demos
│   ├── data/              # Sample JSONL datasets
│   ├── e2e_demo.sh        # Bash demo script
│   └── e2e_demo.py        # Python demo script
├── docs/                  # Documentation
├── tests/                 # Test suite (215+ tests)
├── MODEL_ZOO/             # Pre-built model examples
└── releases/              # Release scripts and notes
```

## 2. Getting Started

### Prerequisites

- Python 3.10+
- pip or poetry
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/anthropics/TinyForgeAI.git
cd TinyForgeAI

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Verify installation
foremforge --help
pytest -q
```

## 3. Running the Full Flow Locally

### Train → Export → Test

```bash
# Step 1: Train (dry-run mode)
foremforge train \
    --data examples/data/demo_dataset.jsonl \
    --out tmp/model \
    --dry-run

# Step 2: Export as microservice
foremforge export \
    --model tmp/model/model_stub.json \
    --out tmp/service \
    --overwrite \
    --export-onnx

# Step 3: Test the service
python examples/e2e_demo.py
```

### Using the Python API

```python
from backend.trainer.stub_trainer import StubTrainer
from backend.exporter.builder import build

# Train
trainer = StubTrainer(data_path="examples/data/demo_dataset.jsonl")
trainer.train(output_dir="tmp/model", dry_run=True)

# Export
build(
    model_path="tmp/model/model_stub.json",
    output_dir="tmp/service",
    overwrite=True,
)

# Test
from fastapi.testclient import TestClient
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "service_app",
    pathlib.Path("tmp/service/app.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

client = TestClient(mod.app)
response = client.post("/predict", json={"input": "hello"})
print(response.json())  # {"output": "olleh", "confidence": 0.75}
```

## 4. Adding a New Connector

Connectors ingest data from various sources and convert them to the training format.

### Required Interface

```python
# connectors/my_connector.py

from typing import Iterator, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class MyConnectorConfig:
    """Configuration for MyConnector."""
    source: str
    mock_mode: bool = False
    # Add other config options

class MyConnector:
    """Connector for [data source description]."""

    def __init__(self, config: MyConnectorConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not self.config.source:
            raise ValueError("source is required")

    def stream_samples(self) -> Iterator[Dict[str, Any]]:
        """
        Stream samples from the data source.

        Yields:
            Dict with keys: input, output, metadata (optional)
        """
        if self.config.mock_mode:
            yield from self._mock_samples()
        else:
            yield from self._real_samples()

    def _mock_samples(self) -> Iterator[Dict[str, Any]]:
        """Generate mock samples for testing."""
        yield {
            "input": "mock input",
            "output": "mock output",
            "metadata": {"source": "mock"}
        }

    def _real_samples(self) -> Iterator[Dict[str, Any]]:
        """Generate real samples from the data source."""
        # Implement actual data fetching
        pass

    @staticmethod
    def row_to_sample(row: Any) -> Dict[str, Any]:
        """Convert a raw row to training sample format."""
        return {
            "input": str(row.get("question", "")),
            "output": str(row.get("answer", "")),
            "metadata": {"raw": row}
        }
```

### Testing Your Connector

```python
# tests/test_my_connector.py

import pytest
from connectors.my_connector import MyConnector, MyConnectorConfig

class TestMyConnector:
    def test_mock_mode(self):
        config = MyConnectorConfig(source="test", mock_mode=True)
        connector = MyConnector(config)

        samples = list(connector.stream_samples())
        assert len(samples) > 0
        assert "input" in samples[0]
        assert "output" in samples[0]

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            MyConnectorConfig(source="")
```

## 5. Adding a New Exporter Format

Exporters convert trained models to deployment formats.

### Required Interface

```python
# backend/exporter/my_format_export.py

from pathlib import Path
from typing import Dict, Any, Optional

def export_to_my_format(
    model_path: Path,
    output_path: Path,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Export model to MyFormat.

    Args:
        model_path: Path to model artifact (model_stub.json)
        output_path: Where to save the exported model
        config: Optional export configuration

    Returns:
        Path to the exported model file
    """
    # Load model metadata
    import json
    with open(model_path) as f:
        metadata = json.load(f)

    # Perform export
    output_file = output_path / "model.myformat"

    # Write exported model
    with open(output_file, "w") as f:
        # Export logic here
        pass

    return output_file

def quantize_my_format(
    model_path: Path,
    output_path: Path,
    bits: int = 8
) -> Path:
    """
    Quantize a MyFormat model.

    Args:
        model_path: Path to MyFormat model
        output_path: Where to save quantized model
        bits: Quantization bit width (4, 8, 16)

    Returns:
        Path to quantized model
    """
    pass
```

### Integration with Builder

Add to `backend/exporter/builder.py`:

```python
def build(..., export_my_format: bool = False):
    # ... existing code ...

    if export_my_format:
        from backend.exporter.my_format_export import export_to_my_format
        export_to_my_format(model_path, output_dir)
```

### Testing

```python
# tests/test_export_my_format.py

import pytest
from pathlib import Path
from backend.exporter.my_format_export import export_to_my_format

class TestMyFormatExport:
    def test_export(self, tmp_path):
        # Create mock model
        model_path = tmp_path / "model_stub.json"
        model_path.write_text('{"model_type": "test"}')

        # Export
        output = export_to_my_format(model_path, tmp_path)

        assert output.exists()
```

## 6. Training (Real Mode)

The current implementation uses a stub trainer. To implement real training:

### Step 1: Add HuggingFace Transformers

```python
# backend/trainer/hf_trainer.py

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

class HFTrainer:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Add LoRA adapter
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["c_attn"],
            lora_dropout=0.1,
        )
        self.model = get_peft_model(self.model, lora_config)

    def train(self, data_path: str, output_dir: str):
        dataset = load_dataset("json", data_files=data_path)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=100,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
        )

        trainer.train()
        trainer.save_model()
```

### Step 2: Export to ONNX

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("output_dir")
dummy_input = torch.randint(0, 1000, (1, 128))

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}}
)
```

### Step 3: Update Microservice

Modify the generated `app.py` to load ONNX:

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")

@app.post("/predict")
def predict(request: PredictRequest):
    inputs = tokenizer(request.input, return_tensors="np")
    outputs = session.run(None, dict(inputs))
    return {"output": decode(outputs), "confidence": 0.95}
```

## 7. Releasing a New Version

### Automated Release

```bash
# Run the release script
bash releases/prepare_release.sh 0.2.0

# Review changes, then commit
git add .
git commit -m "chore(release): v0.2.0"

# Tag and push
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin main --tags
```

### Manual Steps

1. Update `CHANGELOG.md`
2. Bump version in `pyproject.toml`
3. Run full test suite: `pytest -v`
4. Build package: `python -m build`
5. Upload to PyPI: `twine upload dist/*`
6. Create GitHub release

See [docs/release.md](release.md) for detailed instructions.

## 8. Code Style

- **Formatting**: Use `black` for Python formatting
- **Linting**: Use `flake8` for linting
- **Type hints**: Use type hints for all public functions
- **Docstrings**: Use Google-style docstrings
- **Tests**: Aim for >80% coverage

```bash
# Format code
black cli backend connectors inference_server tests

# Lint
flake8 cli backend connectors inference_server

# Type check (optional)
mypy cli backend connectors inference_server
```

## 9. Common Tasks

### Add a CLI Command

Edit `cli/foremforge.py`:

```python
@cli.command()
@click.option("--option", help="Description")
def my_command(option):
    """Command description."""
    click.echo(f"Running with {option}")
```

### Add an API Endpoint

Edit `inference_server/server.py`:

```python
@app.get("/my-endpoint")
def my_endpoint():
    return {"status": "ok"}
```

### Add a Test

```python
# tests/test_my_feature.py

import pytest

class TestMyFeature:
    def test_basic(self):
        assert True

    @pytest.fixture
    def setup(self):
        # Setup code
        yield
        # Teardown code
```

## 10. Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: [GitHub Issues](https://github.com/anthropics/TinyForgeAI/issues)
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

Happy coding! Welcome to the TinyForgeAI community.
