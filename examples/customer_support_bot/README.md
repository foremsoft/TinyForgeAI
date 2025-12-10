# Customer Support Bot Example

This example demonstrates how to build a customer support chatbot using TinyForgeAI.
It covers the complete workflow from data preparation to deployment.

## Overview

Build an AI-powered customer support bot that can:
- Answer frequently asked questions
- Handle common support requests
- Provide consistent, accurate responses
- Run entirely on-premise for data privacy

## Directory Structure

```
customer_support_bot/
├── README.md              # This file
├── prepare_data.py        # Data ingestion and preparation
├── train.py               # Model training script
├── deploy.py              # Inference server deployment
├── test_bot.py            # Testing and validation
├── requirements.txt       # Dependencies
├── configs/
│   └── support_faq_training.yaml  # Training configuration
└── data/
    └── support_faq_dataset/
        └── faq_data.jsonl  # Sample training data
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# Or from project root:
pip install -e ".[training]"
```

### 2. Prepare Training Data

```bash
python prepare_data.py --source sample --output data/support_faq_dataset/faq_data.jsonl
```

Or use your own data sources:

```bash
# From Confluence
python prepare_data.py --source confluence --space SUPPORT --output data/training.jsonl

# From Zendesk export
python prepare_data.py --source zendesk --file tickets.json --output data/training.jsonl

# From markdown files
python prepare_data.py --source markdown --dir ./docs --output data/training.jsonl
```

### 3. Train the Model

```bash
# Quick dry-run to verify setup
python train.py --config configs/support_faq_training.yaml --dry-run

# Full training
python train.py --config configs/support_faq_training.yaml
```

### 4. Deploy the Bot

```bash
python deploy.py --model ./output/support_bot --port 8000
```

### 5. Test the Bot

```bash
python test_bot.py --url http://localhost:8000
```

## Data Format

Training data should be in JSONL format with `input` and `output` fields:

```json
{"input": "How do I reset my password?", "output": "To reset your password: 1) Click 'Forgot Password' on the login page, 2) Enter your email address, 3) Check your inbox for the reset link, 4) Click the link and create a new password. The link expires in 24 hours."}
{"input": "What are your business hours?", "output": "Our customer support team is available Monday through Friday, 9 AM to 6 PM EST. For urgent issues outside these hours, please use our emergency support line at 1-800-XXX-XXXX."}
```

## Configuration

Edit `configs/support_faq_training.yaml` to customize:

- **Model**: Base model to fine-tune (default: `google/flan-t5-small`)
- **LoRA parameters**: Rank, alpha, dropout for efficient fine-tuning
- **Training**: Epochs, batch size, learning rate
- **Output**: Model save location

## Customization

### Adding Custom Data Sources

Extend `prepare_data.py` to support additional data sources:

```python
from prepare_data import DataPreparer

class MyCustomSource(DataPreparer):
    def fetch_data(self):
        # Your custom data fetching logic
        pass
```

### Adjusting Response Style

Modify the training data to reflect your brand's tone:
- **Formal**: "We apologize for the inconvenience..."
- **Casual**: "Sorry about that! Let me help..."
- **Technical**: "To resolve this issue, execute the following..."

## Production Deployment

For production use, consider:

1. **Docker deployment**:
   ```bash
   docker build -t support-bot .
   docker run -p 8000:8000 support-bot
   ```

2. **Kubernetes**:
   See `deploy/k8s/` in the main project for manifests.

3. **Load balancing**: Deploy multiple replicas behind a load balancer.

4. **Monitoring**: Enable Prometheus metrics (included in deploy.py).

## Troubleshooting

### Out of Memory During Training

Reduce batch size in config:
```yaml
training:
  batch_size: 4  # Reduce from 8
```

Or enable gradient checkpointing:
```yaml
training:
  gradient_checkpointing: true
```

### Poor Response Quality

1. Increase training epochs
2. Add more diverse training examples
3. Adjust LoRA rank (higher = more capacity)
4. Review training data for quality issues

### Slow Inference

1. Export to ONNX: `foremforge export --model ./output/support_bot --export-onnx`
2. Enable quantization: `--quantize int8`
3. Use a smaller base model

## License

Apache 2.0 - See main project LICENSE file.
