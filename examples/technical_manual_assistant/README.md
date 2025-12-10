# Technical Manual Assistant Example

A complete implementation of a technical documentation Q&A assistant using TinyForgeAI's fine-tuning and inference capabilities.

## Overview

This example demonstrates how to build a technical manual assistant that:
- Ingests product manuals and technical documentation
- Fine-tunes a model to answer technical questions
- Provides accurate, context-aware responses
- Exposes a REST API for integration

## Project Structure

```
technical_manual_assistant/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── configs/
│   └── assistant_config.yaml    # Model and training configuration
├── data/
│   └── sample_manuals/          # Sample technical documentation
│       ├── installation_guide.txt
│       ├── troubleshooting.txt
│       └── api_reference.txt
├── prepare_data.py              # Prepare training data from manuals
├── train_assistant.py           # Fine-tune the assistant model
├── assistant_service.py         # FastAPI inference service
└── test_assistant.py            # Test the assistant
```

## Quick Start

### 1. Install Dependencies

```bash
# From project root
pip install -e ".[training]"

# Or install example-specific dependencies
pip install -r requirements.txt
```

### 2. Prepare Training Data

```bash
# Generate Q&A pairs from sample manuals
python prepare_data.py --input ./data/sample_manuals --output ./data/training_data.jsonl
```

### 3. Train the Assistant

```bash
# Fine-tune a model on the technical Q&A data
python train_assistant.py --config configs/assistant_config.yaml

# Or use the CLI
foremforge train --config configs/assistant_config.yaml
```

### 4. Start the Service

```bash
python assistant_service.py --port 8002
```

### 5. Test the Assistant

```bash
# Run automated tests
python test_assistant.py --url http://localhost:8002

# Or query directly
curl -X POST http://localhost:8002/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I install the software?"}'
```

## Configuration

Edit `configs/assistant_config.yaml` to customize:

```yaml
# Base model for fine-tuning
model:
  base: gpt2
  output_dir: ./models/technical_assistant

# Training parameters
training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-5
  use_lora: true
  lora_r: 16

# Data parameters
data:
  train_file: ./data/training_data.jsonl
  max_length: 512
```

## Data Preparation

The `prepare_data.py` script generates training data from your technical manuals:

### From Local Files
```bash
python prepare_data.py --input ./docs --output ./data/training_data.jsonl
```

### Custom Q&A Generation
```bash
# Generate more Q&A pairs per document section
python prepare_data.py --input ./docs --output ./data/training_data.jsonl --qa-per-section 5
```

### Adding Custom Q&A Pairs
You can also add custom Q&A pairs to the training data:

```jsonl
{"input": "How do I configure SSL?", "output": "To configure SSL, edit the config.yaml file..."}
{"input": "What ports does the service use?", "output": "The service uses port 8080 for HTTP..."}
```

## API Endpoints

### Ask a Question
```
POST /ask
{
  "question": "How do I troubleshoot connection issues?",
  "max_length": 256
}
```

Response:
```json
{
  "question": "How do I troubleshoot connection issues?",
  "answer": "To troubleshoot connection issues: 1) Check network connectivity...",
  "confidence": 0.85,
  "response_time_ms": 150
}
```

### Health Check
```
GET /health
```

### Model Info
```
GET /info
```

## Fine-Tuning Tips

### For Better Technical Accuracy
1. Include specific technical terms in training data
2. Add code examples and command-line instructions
3. Use structured responses with step-by-step instructions

### For Faster Training
1. Use LoRA (enabled by default)
2. Reduce batch size for limited memory
3. Use gradient checkpointing for larger models

### For Better Generalization
1. Include variations of the same question
2. Add negative examples (what NOT to do)
3. Mix formal and informal question styles

## Integration Examples

### Python Client
```python
import requests

response = requests.post(
    "http://localhost:8002/ask",
    json={"question": "How do I reset the device?"}
)
print(response.json()["answer"])
```

### JavaScript Client
```javascript
const response = await fetch('http://localhost:8002/ask', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({question: 'How do I reset the device?'})
});
const data = await response.json();
console.log(data.answer);
```

## Wiki Tutorial

This example accompanies the wiki tutorial:
[Tutorial: Technical Manual Assistant](https://github.com/foremsoft/TinyForgeAI/wiki/Tutorial-Technical-Manual-Assistant)

## License

Apache 2.0 - See LICENSE in project root.
