# TinyForgeAI Training Documentation

This document describes the training capabilities of TinyForgeAI, including dataset loading, dry-run training, and PEFT/LoRA integration.

## Dataset Loading

TinyForgeAI uses JSONL (JSON Lines) format for training data. Each line must be a valid JSON object with `input` and `output` keys:

```jsonl
{"input": "How do I reset my password?", "output": "Go to Settings > Reset Password."}
{"input": "What is your refund policy?", "output": "Refunds within 30 days with receipt."}
```

### Loading Data Programmatically

```python
from backend.training.dataset import load_jsonl, summarize_dataset

# Load all records
records = load_jsonl("examples/sample_qna.jsonl")

# Get dataset statistics
summary = summarize_dataset(records)
print(f"Records: {summary['n_records']}")
print(f"Avg input length: {summary['avg_input_len']:.2f} tokens")
print(f"Avg output length: {summary['avg_output_len']:.2f} tokens")
```

## Dry-Run Training

The dry-run trainer validates your dataset and creates a stub model artifact without performing actual training:

```bash
python backend/training/train.py --data examples/sample_qna.jsonl --out /tmp/tiny_model --dry-run
```

This creates a `model_stub.json` file in the output directory with metadata about the training run.

## PEFT / LoRA Hook

### What is PEFT/LoRA?

PEFT (Parameter-Efficient Fine-Tuning) is a family of techniques for fine-tuning large language models using fewer trainable parameters. LoRA (Low-Rank Adaptation) is a popular PEFT method that injects trainable low-rank matrices into transformer layers, enabling efficient fine-tuning while preserving most of the original model weights.

### TinyForgeAI LoRA Stub

TinyForgeAI provides a stub `apply_lora(model, config)` function that simulates LoRA adapter application during dry-run training. This allows you to test your training pipeline with LoRA configuration before integrating a real PEFT implementation.

### Usage

**CLI with LoRA:**
```bash
python backend/training/train.py --data examples/sample_qna.jsonl --out /tmp/tiny_model --use-lora --dry-run
```

**Programmatic usage:**
```python
from backend.training.peft_adapter import apply_lora

model = {"model_type": "tinyforge_stub", "n_records": 3}
patched = apply_lora(model)

# patched now contains:
# - lora_applied: True
# - lora_config: {"r": 8, "alpha": 16, "target_modules": ["q", "v"]}
# - lora_timestamp: ISO8601 UTC timestamp
```

### Default LoRA Configuration

The default configuration uses these hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | 8 | LoRA rank (low-rank dimension) |
| `alpha` | 16 | Scaling factor |
| `target_modules` | `["q", "v"]` | Transformer modules to apply LoRA to |

### Replacing the Stub with Real PEFT

To integrate a real PEFT/LoRA implementation:

1. Install the Hugging Face PEFT library: `pip install peft`
2. Replace the stub `apply_lora` function in `backend/training/peft_adapter.py` with actual PEFT logic
3. Key integration points:
   - Use `peft.LoraConfig` to configure the adapter
   - Use `peft.get_peft_model()` to wrap your base model
   - Save adapters using `model.save_pretrained()`
4. Update the model loader to merge LoRA weights at inference time using `model.merge_and_unload()`

For more information, see the [Hugging Face PEFT documentation](https://huggingface.co/docs/peft).
