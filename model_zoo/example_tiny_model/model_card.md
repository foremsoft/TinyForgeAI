# Example Tiny Model

## Model Details

- **Name**: example_tiny_model
- **Version**: 0.1.0
- **Type**: Stub/Demo Model
- **Created**: 2025-01-15
- **License**: Apache-2.0

## Description

This is a demonstration stub model for testing TinyForgeAI's export and inference pipeline. It provides deterministic outputs for validation purposes.

The model implements a simple string reversal operation, returning the input text reversed along with a fixed confidence score.

## Intended Use

- Testing the TinyForgeAI export pipeline
- Validating inference server deployment
- Learning the model zoo format and structure
- CI/CD integration testing

**Not intended for**:
- Production NLP tasks
- Real-world inference applications

## Input/Output

### Input

```json
{
  "input": "hello world"
}
```

### Output

```json
{
  "output": "dlrow olleh",
  "confidence": 0.75
}
```

## Training Data

This stub model was not trained on any data. It provides hardcoded behavior for testing.

## Performance

| Metric | Value |
|--------|-------|
| Latency | <1ms |
| Memory | Minimal |
| Confidence | Fixed 0.75 |

## Limitations

- Not a real ML model - implements string reversal only
- Fixed confidence score regardless of input
- No actual language understanding

## Example Usage

### Python API

```python
from backend.exporter.builder import build

# Export to inference service
build(
    model_path="MODEL_ZOO/example_tiny_model/model_stub.json",
    output_dir="/tmp/demo_service",
    overwrite=True,
)

# Test the service
import sys
sys.path.insert(0, "/tmp/demo_service")
from app import app
from fastapi.testclient import TestClient

client = TestClient(app)
response = client.post("/predict", json={"input": "test"})
print(response.json())  # {"output": "tset", "confidence": 0.75}
```

### CLI

```bash
# Export
foremforge export \
    --model MODEL_ZOO/example_tiny_model/model_stub.json \
    --out /tmp/demo_service

# Serve
foremforge serve --dir /tmp/demo_service --port 8000

# Query
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"input": "hello"}'
```

## Files

- `model_card.md` - This documentation
- `model_stub.json` - Model artifact/metadata

## Citation

```
@software{tinyforgeai_example_model,
  title = {TinyForgeAI Example Tiny Model},
  author = {TinyForgeAI Contributors},
  year = {2025},
  url = {https://github.com/anthropics/TinyForgeAI}
}
```

## Contact

For questions or issues, please open a GitHub issue at:
https://github.com/anthropics/TinyForgeAI/issues
