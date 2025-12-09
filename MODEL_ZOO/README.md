# TinyForgeAI Model Zoo

The Model Zoo is a collection of pre-trained and fine-tuned models ready for deployment with TinyForgeAI.

## Purpose

- Provide ready-to-use model artifacts for quick deployment
- Showcase example configurations and model cards
- Enable benchmarking and comparison across models

## Model Format

Each model in the zoo follows this structure:

```
MODEL_ZOO/
└── model_name/
    ├── model_card.md      # Model documentation
    ├── model_stub.json    # Model metadata/artifact
    ├── config.json        # Training configuration (optional)
    └── samples/           # Example inputs/outputs (optional)
```

## Model Card Format

Each model includes a `model_card.md` with:

- **Model Name**: Human-readable identifier
- **Description**: What the model does
- **Intended Use**: Target use cases
- **Training Data**: Description of training dataset
- **Performance**: Metrics and benchmarks
- **Limitations**: Known limitations and biases
- **License**: Usage terms
- **Example Usage**: Code snippets for inference

## Available Models

| Model | Description | License |
|-------|-------------|---------|
| [example_tiny_model](example_tiny_model/) | Demo stub model for testing | Apache-2.0 |

## Adding Models

To contribute a model:

1. Create a new directory under `MODEL_ZOO/`
2. Include `model_card.md` following the template
3. Include model artifacts (JSON, ONNX, etc.)
4. Submit a pull request

## Using Models

```python
from backend.exporter.builder import build

# Export a model zoo model to inference service
build(
    model_path="MODEL_ZOO/example_tiny_model/model_stub.json",
    output_dir="/tmp/my_service",
    overwrite=True,
)
```

Or via CLI:

```bash
foremforge export \
    --model MODEL_ZOO/example_tiny_model/model_stub.json \
    --out /tmp/my_service
```

## License

Individual models may have different licenses. Check each model's `model_card.md` for specific terms.
