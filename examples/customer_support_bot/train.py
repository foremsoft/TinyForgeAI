#!/usr/bin/env python3
"""
Training script for Customer Support Bot.

This script handles the complete training pipeline using TrainingConfig
with LoRA optimization for efficient fine-tuning.

Usage:
    python train.py --config configs/support_faq_training.yaml
    python train.py --config configs/support_faq_training.yaml --dry-run
    python train.py --data data/support_faq_dataset/faq_data.jsonl --output ./output/support_bot
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_with_real_trainer(
    data_path: str,
    output_dir: str,
    model_name: str = "google/flan-t5-small",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    max_length: int = 512,
    gradient_checkpointing: bool = False,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    logging_steps: int = 10,
    save_steps: int = 500,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Train the support bot model using RealTrainer.

    Args:
        data_path: Path to training data JSONL file
        output_dir: Directory to save the trained model
        model_name: Base model to fine-tune
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        use_lora: Whether to use LoRA for efficient fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        max_length: Maximum sequence length
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for regularization
        logging_steps: Steps between logging
        save_steps: Steps between checkpoints
        dry_run: If True, run in dry-run mode without actual training

    Returns:
        Dictionary with training metrics
    """
    from backend.training.real_trainer import RealTrainer, TrainingConfig

    print("=" * 60)
    print("Customer Support Bot Training")
    print("=" * 60)
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Base model: {model_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Use LoRA: {use_lora}")
    if use_lora:
        print(f"  LoRA rank: {lora_r}")
        print(f"  LoRA alpha: {lora_alpha}")
        print(f"  LoRA dropout: {lora_dropout}")
    print(f"Max length: {max_length}")
    print(f"Gradient checkpointing: {gradient_checkpointing}")
    print(f"Dry run: {dry_run}")
    print("=" * 60)

    # Verify data file exists
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    # Count training samples
    with open(data_file, "r", encoding="utf-8") as f:
        sample_count = sum(1 for line in f if line.strip())
    print(f"Training samples: {sample_count}")

    if dry_run:
        print("\n[DRY RUN MODE] - Skipping actual training")
        print("Configuration validated successfully!")

        # Create output directory and save stub model info
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stub_info = {
            "model_name": model_name,
            "training_config": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "use_lora": use_lora,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "max_length": max_length,
            },
            "data_path": str(data_path),
            "sample_count": sample_count,
            "dry_run": True,
        }

        with open(output_path / "model_stub.json", "w", encoding="utf-8") as f:
            json.dump(stub_info, f, indent=2)

        print(f"Stub model info saved to {output_path / 'model_stub.json'}")

        return {
            "status": "dry_run",
            "model_name": model_name,
            "sample_count": sample_count,
            "output_dir": output_dir,
        }

    # Create training configuration
    config = TrainingConfig(
        model_name=model_name,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        max_length=max_length,
        gradient_checkpointing=gradient_checkpointing,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = RealTrainer(config)

    # Run training
    print("\nStarting training...")
    metrics = trainer.train(data_path)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final loss: {metrics.get('final_loss', 'N/A')}")
    print(f"Model saved to: {output_dir}")

    return metrics


def train_with_dry_run_trainer(
    data_path: str,
    output_dir: str,
    model_name: str = "google/flan-t5-small",
    **kwargs,
) -> dict[str, Any]:
    """
    Train using the dry-run trainer (no GPU required).

    This is useful for testing the pipeline without actual training.
    """
    from backend.training.trainer import DryRunTrainer, TrainerConfig

    print("=" * 60)
    print("Customer Support Bot Training (Dry Run Trainer)")
    print("=" * 60)
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Base model: {model_name}")
    print("=" * 60)

    # Verify data file exists
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    # Create trainer configuration
    config = TrainerConfig(
        model_name=model_name,
        output_dir=output_dir,
    )

    # Initialize trainer
    trainer = DryRunTrainer(config)

    # Run training
    print("\nRunning dry-run training...")
    metrics = trainer.train(data_path)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model stub saved to: {output_dir}")

    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Customer Support Bot model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with config file
    python train.py --config configs/support_faq_training.yaml

    # Dry run to verify setup
    python train.py --config configs/support_faq_training.yaml --dry-run

    # Train with command line args
    python train.py --data data/support_faq_dataset/faq_data.jsonl --output ./output/support_bot

    # Use dry-run trainer (no GPU)
    python train.py --data data/faq_data.jsonl --output ./output --trainer dry-run
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to training data JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-small",
        help="Base model to fine-tune (default: google/flan-t5-small)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient fine-tuning (default: True)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for memory efficiency",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode without actual training",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        choices=["real", "dry-run"],
        default="real",
        help="Trainer type: 'real' for HuggingFace training, 'dry-run' for stub (default: real)",
    )

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            parser.error(f"Config file not found: {args.config}")
        config = load_config(args.config)

    # Override config with command line args
    data_path = args.data or config.get("data", {}).get("path")
    output_dir = args.output or config.get("output", {}).get("dir", "./output/support_bot")

    if not data_path:
        parser.error("--data or config file with data.path is required")

    # Get training parameters
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    lora_config = config.get("lora", {})

    model_name = args.model if args.model != "google/flan-t5-small" else model_config.get(
        "name", "google/flan-t5-small"
    )
    num_epochs = args.epochs if args.epochs != 3 else training_config.get("epochs", 3)
    batch_size = args.batch_size if args.batch_size != 8 else training_config.get("batch_size", 8)
    learning_rate = (
        args.learning_rate
        if args.learning_rate != 3e-4
        else training_config.get("learning_rate", 3e-4)
    )
    use_lora = not args.no_lora and lora_config.get("enabled", True)
    lora_r = args.lora_r if args.lora_r != 8 else lora_config.get("r", 8)
    lora_alpha = args.lora_alpha if args.lora_alpha != 32 else lora_config.get("alpha", 32)
    lora_dropout = lora_config.get("dropout", 0.1)
    max_length = (
        args.max_length if args.max_length != 512 else training_config.get("max_length", 512)
    )
    gradient_checkpointing = args.gradient_checkpointing or training_config.get(
        "gradient_checkpointing", False
    )

    # Select trainer and run
    if args.trainer == "dry-run":
        metrics = train_with_dry_run_trainer(
            data_path=data_path,
            output_dir=output_dir,
            model_name=model_name,
        )
    else:
        try:
            metrics = train_with_real_trainer(
                data_path=data_path,
                output_dir=output_dir,
                model_name=model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                max_length=max_length,
                gradient_checkpointing=gradient_checkpointing,
                dry_run=args.dry_run,
            )
        except ImportError as e:
            print(f"Warning: Real trainer not available ({e}). Falling back to dry-run trainer.")
            metrics = train_with_dry_run_trainer(
                data_path=data_path,
                output_dir=output_dir,
                model_name=model_name,
            )

    print("\nTraining metrics:")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
