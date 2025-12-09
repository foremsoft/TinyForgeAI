#!/usr/bin/env python3
"""
TinyForgeAI LoRA Training Example

This script demonstrates efficient fine-tuning using LoRA (Low-Rank Adaptation).
LoRA allows training large models with minimal memory by only updating small
adapter matrices instead of the full model weights.

Prerequisites:
    pip install -e ".[training]"

Usage:
    python examples/training/lora_training.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run the LoRA training example."""
    print("=" * 60)
    print("TinyForgeAI LoRA Training Example")
    print("=" * 60)

    from backend.training.real_trainer import RealTrainer, TrainingConfig

    # Check dependencies
    trainer_check = RealTrainer.__new__(RealTrainer)
    trainer_check._check_dependencies()

    if not getattr(trainer_check, 'training_available', False):
        print("\nTraining dependencies not installed!")
        print("Install with: pip install -e '.[training]'")
        return

    # LoRA Configuration
    # The key difference from regular training is use_lora=True
    config = TrainingConfig(
        # Model selection - works with any HuggingFace model
        model_name="distilbert-base-uncased",

        # Output location
        output_dir="./output/lora_model",

        # Training hyperparameters
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-4,  # LoRA typically uses higher learning rates

        # Enable LoRA - this is the key setting!
        use_lora=True,

        # LoRA-specific parameters
        lora_r=8,              # Rank of the low-rank matrices (lower = smaller adapter)
        lora_alpha=16,         # Scaling factor (typically 2x lora_r)
        lora_dropout=0.1,      # Dropout for regularization

        # Training settings
        logging_steps=5,
        save_steps=50,
        max_length=256,
    )

    print("\nLoRA Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA enabled: {config.use_lora}")
    print(f"  LoRA rank (r): {config.lora_r}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  LoRA dropout: {config.lora_dropout}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Output: {config.output_dir}")

    # Memory comparison info
    print("\nMemory Benefits of LoRA:")
    print("  Full fine-tuning:  ~4GB+ VRAM for DistilBERT")
    print("  LoRA fine-tuning:  ~1-2GB VRAM for DistilBERT")
    print("  Adapter size:      ~1-10MB (vs 250MB+ for full model)")

    # Create sample data
    data_path = project_root / "examples" / "training" / "sample_data.jsonl"

    if not data_path.exists():
        print(f"\nData file not found: {data_path}")
        print("Please ensure sample_data.jsonl exists")
        return

    print(f"\nTraining data: {data_path}")

    # Create trainer
    print("\nInitializing LoRA trainer...")
    trainer = RealTrainer(config)
    print(f"  Device: {trainer.device}")

    # Show trainable parameters
    if hasattr(trainer, 'model') and trainer.model is not None:
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"\n  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Start training
    print("\nStarting LoRA training...")
    print("-" * 60)

    try:
        trainer.train(str(data_path))
    except Exception as e:
        print(f"\nTraining failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Reduce batch_size if out of memory")
        print("  2. Reduce lora_r for smaller adapters")
        print("  3. Ensure CUDA is available for GPU training")
        return

    print("-" * 60)
    print("\nLoRA Training Complete!")
    print(f"  Model saved to: {config.output_dir}")

    print("\nWhat was saved:")
    print("  - adapter_config.json: LoRA configuration")
    print("  - adapter_model.bin: LoRA weights only (~1-10MB)")
    print("  - Base model is NOT saved (uses original weights)")

    print("\nLoading the LoRA model later:")
    print("""
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, "./output/lora_model")
    """)

    print("\nNext steps:")
    print("  1. Test inference with the LoRA model")
    print("  2. Try different lora_r values (4, 8, 16, 32)")
    print("  3. Experiment with different target modules")


if __name__ == "__main__":
    main()
