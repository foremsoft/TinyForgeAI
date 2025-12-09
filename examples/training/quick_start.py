#!/usr/bin/env python3
"""
TinyForgeAI Quick Start Example

This script demonstrates the simplest way to train a model with TinyForgeAI.

Prerequisites:
    pip install -e ".[training]"

Usage:
    python examples/training/quick_start.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run the quick start example."""
    print("=" * 60)
    print("TinyForgeAI Quick Start")
    print("=" * 60)

    # Import after path setup
    from backend.training.real_trainer import RealTrainer, TrainingConfig

    # Check if training dependencies are available
    trainer = RealTrainer.__new__(RealTrainer)
    trainer._check_dependencies()

    if not hasattr(trainer, 'training_available') or not trainer.training_available:
        print("\n⚠️  Training dependencies not installed!")
        print("Install with: pip install -e '.[training]'")
        print("\nThis will install:")
        print("  - torch (PyTorch)")
        print("  - transformers (HuggingFace)")
        print("  - datasets")
        print("  - peft (for LoRA)")
        print("  - accelerate")
        return

    # Configuration - Start simple!
    config = TrainingConfig(
        # Use a small, fast model for quick testing
        model_name="distilbert-base-uncased",

        # Where to save the trained model
        output_dir="./output/quick_start_model",

        # Training settings
        num_epochs=3,           # How many times to go through the data
        batch_size=8,           # How many examples to process at once
        learning_rate=2e-5,     # How fast to learn

        # Progress logging
        logging_steps=5,        # Print progress every 5 steps
    )

    print("\n📋 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Output: {config.output_dir}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")

    # Data path
    data_path = project_root / "examples" / "training" / "sample_data.jsonl"

    if not data_path.exists():
        print(f"\n❌ Data file not found: {data_path}")
        print("Please ensure sample_data.jsonl exists in examples/training/")
        return

    print(f"\n📁 Training data: {data_path}")

    # Create trainer
    print("\n🔧 Initializing trainer...")
    trainer = RealTrainer(config)
    print(f"   Device: {trainer.device}")

    # Start training
    print("\n🚀 Starting training...")
    print("-" * 60)

    try:
        trainer.train(str(data_path))
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Reduce batch_size if out of memory")
        print("  2. Check that data file is valid JSONL")
        print("  3. Ensure all dependencies are installed")
        return

    print("-" * 60)
    print("\n✅ Training complete!")
    print(f"   Model saved to: {config.output_dir}")

    print("\n📖 Next steps:")
    print("   1. Load your model for inference")
    print("   2. Export to ONNX for deployment")
    print("   3. Try the examples in examples/training/")


if __name__ == "__main__":
    main()
