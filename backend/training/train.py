"""
TinyForgeAI Training Script.

Provides a command-line interface for training models on JSONL datasets.
Supports dry-run mode for validation without actual training.
Supports optional LoRA adapter application via --use-lora flag.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path when run as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from backend.training.dataset import load_jsonl, summarize_dataset
from backend.training.peft_adapter import apply_lora
from backend.training.utils import ensure_dir, iso_now_utc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_training(
    data_path: str,
    output_dir: str,
    dry_run: bool = False,
    use_lora: bool = False,
) -> None:
    """
    Run the training pipeline.

    Args:
        data_path: Path to the JSONL training data file.
        output_dir: Directory to write model artifacts.
        dry_run: If True, only validate data and create stub artifact.
        use_lora: If True, apply LoRA adapter to the model.
    """
    # Load and validate dataset
    logger.info(f"Loading dataset from: {data_path}")
    records = load_jsonl(data_path)

    # Compute and log summary
    summary = summarize_dataset(records)
    logger.info(f"Dataset summary: {summary['n_records']} records")
    logger.info(f"  Average input length: {summary['avg_input_len']:.2f} tokens")
    logger.info(f"  Average output length: {summary['avg_output_len']:.2f} tokens")

    # Ensure output directory exists
    ensure_dir(output_dir)

    # Create stub model artifact
    artifact = {
        "model_type": "tinyforge_stub",
        "n_records": summary["n_records"],
        "created_time": iso_now_utc(),
        "notes": "dry-run artifact" if dry_run else "training artifact",
    }

    # Apply LoRA adapter if requested
    if use_lora:
        logger.info("Applying LoRA adapter to model...")
        artifact = apply_lora(artifact)
        logger.info(f"LoRA applied with config: {artifact['lora_config']}")

    artifact_path = Path(output_dir) / "model_stub.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    logger.info(f"Model artifact written to: {artifact_path}")

    if dry_run:
        logger.info("Dry-run complete. No actual training performed.")
    else:
        logger.info("Training complete (stub mode - no actual training performed).")


def main() -> int:
    """CLI entry point for the trainer."""
    parser = argparse.ArgumentParser(
        description="Train a TinyForgeAI model on JSONL data."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the JSONL training data file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Directory to write model artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and create stub artifact without training.",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Apply LoRA adapter to the model.",
    )

    args = parser.parse_args()

    try:
        run_training(
            data_path=args.data,
            output_dir=args.out,
            dry_run=args.dry_run,
            use_lora=args.use_lora,
        )
        return 0
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
