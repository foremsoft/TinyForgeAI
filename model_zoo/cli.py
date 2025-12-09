"""
Model Zoo CLI

Command-line interface for browsing and using pre-configured models.

Usage:
    python -m model_zoo.cli list                    # List all models
    python -m model_zoo.cli list --task qa          # List models for a task
    python -m model_zoo.cli info qa_flan_t5_small   # Get model info
    python -m model_zoo.cli train qa_flan_t5_small  # Train with a model config
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from model_zoo.registry import (
    list_models,
    get_model_info,
    load_model_config,
    TaskType,
    MODEL_REGISTRY,
)


def cmd_list(args: argparse.Namespace) -> int:
    """List available models."""
    task_type = None
    if args.task:
        try:
            task_type = TaskType(args.task)
        except ValueError:
            # Try matching by prefix
            for t in TaskType:
                if t.value.startswith(args.task) or args.task in t.value:
                    task_type = t
                    break
            if task_type is None:
                print(f"Unknown task type: {args.task}")
                print(f"Available: {', '.join(t.value for t in TaskType)}")
                return 1

    models = list_models(task_type)

    if not models:
        print("No models found.")
        return 0

    if args.json:
        print(json.dumps(models, indent=2))
        return 0

    # Group by task type
    by_task = {}
    for m in models:
        task = m["task_type"]
        if task not in by_task:
            by_task[task] = []
        by_task[task].append(m)

    print("=" * 70)
    print("TinyForgeAI Model Zoo")
    print("=" * 70)

    for task, task_models in sorted(by_task.items()):
        print(f"\n{task.upper().replace('_', ' ')}")
        print("-" * 40)
        for m in task_models:
            print(f"  {m['name']:<30} {m['display_name']}")
            if args.verbose:
                print(f"    Base: {m['base_model']}")
                print(f"    {m['description'][:60]}...")
                print()

    print(f"\nTotal: {len(models)} models")
    print("\nUse 'python -m model_zoo.cli info <name>' for details")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed model information."""
    try:
        info = get_model_info(args.model)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    print("=" * 70)
    print(f"{info['display_name']}")
    print("=" * 70)

    print(f"\nName:        {info['name']}")
    print(f"Task:        {info['task_type']}")
    print(f"Base Model:  {info['base_model']}")
    print(f"Model Type:  {info['model_type']}")

    print(f"\nDescription:\n  {info['description']}")

    print("\nTraining Defaults:")
    td = info["training_defaults"]
    print(f"  Epochs:         {td['epochs']}")
    print(f"  Batch Size:     {td['batch_size']}")
    print(f"  Learning Rate:  {td['learning_rate']}")
    print(f"  Max Input:      {td['max_input_length']} tokens")
    print(f"  Max Output:     {td['max_output_length']} tokens")

    lora = info["lora"]
    print(f"\nLoRA Settings:")
    print(f"  Recommended:    {'Yes' if lora['recommended'] else 'No'}")
    if lora["recommended"]:
        print(f"  Rank:           {lora['rank']}")
        print(f"  Alpha:          {lora['alpha']}")
        if lora["target_modules"]:
            print(f"  Target Modules: {', '.join(lora['target_modules'])}")

    res = info["resources"]
    print(f"\nResource Requirements:")
    print(f"  Min GPU Memory: {res['min_gpu_memory_gb']} GB")
    print(f"  CPU Compatible: {'Yes' if res['cpu_compatible'] else 'No'}")
    print(f"  Est. Time/Epoch: {res['estimated_time_per_epoch']}")

    df = info["data_format"]
    print(f"\nData Format:")
    print(f"  Input Field:  {df['input_field']}")
    print(f"  Output Field: {df['output_field']}")
    print(f"\n  Example:")
    for k, v in df["example"].items():
        print(f"    {k}: {v[:50]}{'...' if len(v) > 50 else ''}")

    meta = info["metadata"]
    if meta["use_cases"]:
        print(f"\nUse Cases:")
        for uc in meta["use_cases"]:
            print(f"  - {uc}")

    if meta["tags"]:
        print(f"\nTags: {', '.join(meta['tags'])}")

    if meta["model_url"]:
        print(f"\nModel URL: {meta['model_url']}")

    print("\n" + "=" * 70)
    print("Quick Start:")
    print(f"  python -m model_zoo.cli train {args.model} --data your_data.jsonl")
    print("=" * 70)

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train a model using a zoo configuration."""
    try:
        config = load_model_config(args.model)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Check data file
    data_path = Path(args.data) if args.data else None
    if data_path and not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    # Use sample data if none provided
    if not data_path:
        zoo_dir = Path(__file__).parent
        sample_files = {
            TaskType.QUESTION_ANSWERING: "qa_samples.jsonl",
            TaskType.SUMMARIZATION: "summarization_samples.jsonl",
            TaskType.CLASSIFICATION: "classification_samples.jsonl",
            TaskType.SENTIMENT: "classification_samples.jsonl",
            TaskType.CODE_GENERATION: "code_samples.jsonl",
            TaskType.CONVERSATION: "chat_samples.jsonl",
        }
        sample_name = sample_files.get(config.task_type)
        if sample_name:
            data_path = zoo_dir / "datasets" / sample_name
            if data_path.exists():
                print(f"Using sample dataset: {data_path}")
            else:
                data_path = None

    if not data_path:
        print("Error: No data file specified. Use --data <file>")
        return 1

    # Build training config
    training_config = config.to_training_config()

    # Override with CLI args
    if args.epochs:
        training_config["num_epochs"] = args.epochs
    if args.batch_size:
        training_config["batch_size"] = args.batch_size
    if args.learning_rate:
        training_config["learning_rate"] = args.learning_rate
    if args.output:
        training_config["output_dir"] = args.output
    else:
        training_config["output_dir"] = f"./output/{config.name}"

    # LoRA settings
    use_lora = args.lora if args.lora is not None else config.lora_recommended
    if use_lora:
        training_config["use_lora"] = True
        training_config["lora_rank"] = config.lora_rank
        training_config["lora_alpha"] = config.lora_alpha
        if config.lora_target_modules:
            training_config["lora_target_modules"] = config.lora_target_modules

    print("=" * 60)
    print(f"Training: {config.display_name}")
    print("=" * 60)
    print(f"\nModel:     {config.base_model}")
    print(f"Data:      {data_path}")
    print(f"Output:    {training_config['output_dir']}")
    print(f"Epochs:    {training_config['num_epochs']}")
    print(f"Batch:     {training_config['batch_size']}")
    print(f"LR:        {training_config['learning_rate']}")
    print(f"LoRA:      {'Yes' if use_lora else 'No'}")

    if args.dry_run:
        print("\n[DRY RUN - No training performed]")
        print("\nTraining config:")
        print(json.dumps(training_config, indent=2))
        return 0

    # Import and run training
    print("\nInitializing trainer...")

    try:
        from backend.training.real_trainer import RealTrainer, TrainingConfig

        trainer_config = TrainingConfig(
            model_name=training_config["model_name"],
            output_dir=training_config["output_dir"],
            num_epochs=training_config["num_epochs"],
            batch_size=training_config["batch_size"],
            learning_rate=training_config["learning_rate"],
            max_length=training_config.get("max_length", 512),
        )

        trainer = RealTrainer(trainer_config)
        print(f"Device: {trainer.device}")

        print("\nStarting training...")
        print("-" * 60)
        trainer.train(str(data_path))
        print("-" * 60)

        print(f"\nTraining complete! Model saved to: {training_config['output_dir']}")
        return 0

    except ImportError as e:
        print(f"\nError: Training dependencies not installed")
        print("Install with: pip install -e '.[training]'")
        print(f"\nDetails: {e}")
        return 1
    except Exception as e:
        print(f"\nTraining failed: {e}")
        return 1


def cmd_datasets(args: argparse.Namespace) -> int:
    """List available sample datasets."""
    zoo_dir = Path(__file__).parent
    datasets_dir = zoo_dir / "datasets"

    if not datasets_dir.exists():
        print("No datasets directory found.")
        return 1

    print("=" * 60)
    print("Available Sample Datasets")
    print("=" * 60)

    for f in sorted(datasets_dir.glob("*.jsonl")):
        # Count lines
        with open(f) as fh:
            lines = sum(1 for _ in fh)
        print(f"\n{f.name}")
        print(f"  Path:    {f}")
        print(f"  Samples: {lines}")

    print("\n" + "=" * 60)
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="model_zoo",
        description="TinyForgeAI Model Zoo - Pre-configured models for NLP tasks",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--task", "-t", help="Filter by task type")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    list_parser.add_argument("--verbose", "-v", action="store_true", help="Show details")
    list_parser.set_defaults(func=cmd_list)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("model", help="Model name")
    info_parser.add_argument("--json", action="store_true", help="Output as JSON")
    info_parser.set_defaults(func=cmd_info)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train with a model config")
    train_parser.add_argument("model", help="Model name from zoo")
    train_parser.add_argument("--data", "-d", help="Training data file (JSONL)")
    train_parser.add_argument("--output", "-o", help="Output directory")
    train_parser.add_argument("--epochs", "-e", type=int, help="Number of epochs")
    train_parser.add_argument("--batch-size", "-b", type=int, help="Batch size")
    train_parser.add_argument("--learning-rate", "-lr", type=float, help="Learning rate")
    train_parser.add_argument("--lora", action="store_true", default=None, help="Enable LoRA")
    train_parser.add_argument("--no-lora", action="store_false", dest="lora", help="Disable LoRA")
    train_parser.add_argument("--dry-run", action="store_true", help="Show config without training")
    train_parser.set_defaults(func=cmd_train)

    # Datasets command
    datasets_parser = subparsers.add_parser("datasets", help="List sample datasets")
    datasets_parser.set_defaults(func=cmd_datasets)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
