#!/usr/bin/env python3
"""
Training Script for Technical Manual Assistant

Fine-tunes a language model on technical Q&A data.

Usage:
    python train_assistant.py --config configs/assistant_config.yaml

    Or use the CLI:
    foremforge train --config configs/assistant_config.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def check_dependencies():
    """Check if training dependencies are available."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append('torch')

    try:
        import transformers
    except ImportError:
        missing.append('transformers')

    try:
        import peft
    except ImportError:
        missing.append('peft')

    if missing:
        print("Missing training dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install -e '.[training]'")
        return False

    return True


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration."""
    config_path = Path(config_path)

    if not config_path.exists():
        # Try relative to script directory
        config_path = Path(__file__).parent / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_dataset(config: Dict[str, Any]):
    """Load and prepare the training dataset."""
    from datasets import Dataset

    data_config = config.get('data', {})
    train_file = data_config.get('train_file', './data/training_data.jsonl')

    # Handle relative paths
    if not Path(train_file).is_absolute():
        train_file = Path(__file__).parent / train_file

    if not Path(train_file).exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n"
            "Run prepare_data.py first to generate training data."
        )

    # Load JSONL data
    data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Loaded {len(data)} training examples")

    # Create dataset
    input_col = data_config.get('input_column', 'input')
    output_col = data_config.get('output_column', 'output')

    dataset = Dataset.from_list([
        {
            'input': item[input_col],
            'output': item[output_col]
        }
        for item in data
    ])

    # Split into train/validation
    val_split = data_config.get('validation_split', 0.1)
    if val_split > 0:
        split = dataset.train_test_split(test_size=val_split, seed=42)
        return split['train'], split['test']

    return dataset, None


def format_prompt(example: Dict[str, str]) -> str:
    """Format a Q&A pair as a training prompt."""
    return f"""Question: {example['input']}

Answer: {example['output']}"""


def train_model(config: Dict[str, Any]):
    """Train the model using the provided configuration."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )

    model_config = config.get('model', {})
    train_config = config.get('training', {})
    data_config = config.get('data', {})

    # Load base model and tokenizer
    base_model = model_config.get('base', 'gpt2')
    print(f"Loading base model: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Apply LoRA if configured
    if train_config.get('use_lora', True):
        print("Applying LoRA for efficient fine-tuning")
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=train_config.get('lora_r', 16),
            lora_alpha=train_config.get('lora_alpha', 32),
            lora_dropout=train_config.get('lora_dropout', 0.1),
            target_modules=train_config.get('lora_target_modules', ['c_attn', 'c_proj']),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Prepare datasets
    train_dataset, eval_dataset = prepare_dataset(config)
    max_length = data_config.get('max_length', 512)

    def tokenize_function(examples):
        """Tokenize examples."""
        texts = [format_prompt({'input': inp, 'output': out})
                 for inp, out in zip(examples['input'], examples['output'])]
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
        )

    print("Tokenizing dataset...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    if eval_dataset:
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )

    # Training arguments
    output_dir = model_config.get('output_dir', './models/technical_assistant')
    if not Path(output_dir).is_absolute():
        output_dir = Path(__file__).parent / output_dir

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config.get('epochs', 3),
        per_device_train_batch_size=train_config.get('batch_size', 4),
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 2),
        learning_rate=train_config.get('learning_rate', 2e-5),
        warmup_steps=train_config.get('warmup_steps', 100),
        weight_decay=train_config.get('weight_decay', 0.01),
        logging_dir=str(Path(output_dir) / 'logs'),
        logging_steps=10,
        save_strategy=model_config.get('save_strategy', 'epoch'),
        evaluation_strategy='epoch' if eval_dataset else 'no',
        load_best_model_at_end=True if eval_dataset else False,
        report_to=['tensorboard'] if config.get('logging', {}).get('tensorboard', True) else [],
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print(f"\n=== Starting Training ===")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {train_config.get('epochs', 3)}")
    print(f"Batch size: {train_config.get('batch_size', 4)}")
    print()

    trainer.train()

    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(f"\n=== Training Complete! ===")
    print(f"Model saved to: {output_dir}")

    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description='Train technical manual assistant')
    parser.add_argument('--config', default='configs/assistant_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate config without training')

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    if args.dry_run:
        print("\n=== Dry Run Mode ===")
        print("Configuration validated successfully")
        print(f"Base model: {config.get('model', {}).get('base', 'gpt2')}")
        print(f"Training epochs: {config.get('training', {}).get('epochs', 3)}")
        print(f"Use LoRA: {config.get('training', {}).get('use_lora', True)}")
        return

    # Train model
    train_model(config)


if __name__ == '__main__':
    main()
