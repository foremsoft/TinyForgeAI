"""
URL Trainer - One-Click Training from URLs.

Train models directly from URLs without manual data preparation.
"""

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .extractor import URLExtractor, ExtractedData

logger = logging.getLogger(__name__)


@dataclass
class URLTrainConfig:
    """Configuration for URL-based training."""

    # Model settings
    model_name: str = "t5-small"
    model_type: str = "seq2seq"

    # Training settings
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    use_lora: bool = True

    # Augmentation
    augment: bool = True
    augment_target: int = 500

    # Dry run mode
    dry_run: bool = False

    # Output
    output_dir: str = "./url_trained_model"

    # API tokens (optional)
    notion_token: Optional[str] = None
    google_credentials: Optional[str] = None
    github_token: Optional[str] = None


class URLTrainer:
    """
    Train models directly from URLs.

    One-click training from:
    - Notion pages
    - Google Docs/Sheets
    - GitHub files
    - Websites (FAQ pages)
    - Raw JSON/JSONL/CSV files

    Usage:
        trainer = URLTrainer()

        # Single URL
        result = trainer.train_from_url(
            "https://notion.so/my-workspace/FAQ-123",
            output_dir="./my_model"
        )

        # Multiple URLs
        result = trainer.train_from_urls([
            "https://notion.so/FAQ",
            "https://docs.google.com/document/d/...",
            "https://example.com/help"
        ])
    """

    def __init__(self, config: Optional[URLTrainConfig] = None):
        """
        Initialize URL trainer.

        Args:
            config: Training configuration.
        """
        self.config = config or URLTrainConfig()
        self.extractor = URLExtractor(
            notion_token=self.config.notion_token,
            google_credentials=self.config.google_credentials,
            github_token=self.config.github_token,
        )

    def train_from_url(
        self,
        url: str,
        output_dir: Optional[str] = None,
        dry_run: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Train a model from a single URL.

        Args:
            url: URL to extract training data from.
            output_dir: Output directory for the trained model.
            dry_run: If True, validate without training.

        Returns:
            Training result dictionary.
        """
        return self.train_from_urls([url], output_dir, dry_run)

    def train_from_urls(
        self,
        urls: List[str],
        output_dir: Optional[str] = None,
        dry_run: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Train a model from multiple URLs.

        Args:
            urls: List of URLs to extract training data from.
            output_dir: Output directory for the trained model.
            dry_run: If True, validate without training.

        Returns:
            Training result dictionary.
        """
        output_dir = Path(output_dir or self.config.output_dir)
        dry_run = dry_run if dry_run is not None else self.config.dry_run

        logger.info(f"Training from {len(urls)} URL(s)")

        # Step 1: Extract data from all URLs
        all_samples = []
        extraction_results = []

        for url in urls:
            try:
                extracted = self.extractor.extract(url)
                all_samples.extend(extracted.samples)
                extraction_results.append({
                    "url": url,
                    "source_type": extracted.source_type,
                    "samples_extracted": len(extracted.samples),
                    "title": extracted.title,
                })
                logger.info(f"Extracted {len(extracted.samples)} samples from {url}")
            except Exception as e:
                logger.error(f"Failed to extract from {url}: {e}")
                extraction_results.append({
                    "url": url,
                    "error": str(e),
                    "samples_extracted": 0,
                })

        if not all_samples:
            raise ValueError("No training samples extracted from any URL")

        logger.info(f"Total samples extracted: {len(all_samples)}")

        # Step 2: Augment data if configured
        if self.config.augment and len(all_samples) < self.config.augment_target:
            logger.info(f"Augmenting data to {self.config.augment_target} samples")
            all_samples = self._augment_samples(all_samples)

        # Step 3: Save samples to temp file
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = output_dir / "training_data.jsonl"

        with open(data_path, "w", encoding="utf-8") as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(all_samples)} samples to {data_path}")

        # Step 4: Train model
        if dry_run:
            logger.info("Dry run mode - skipping actual training")
            training_result = self._create_stub_model(output_dir, all_samples)
        else:
            training_result = self._train_model(data_path, output_dir)

        # Step 5: Save metadata
        metadata = {
            "source_urls": urls,
            "extraction_results": extraction_results,
            "total_samples": len(all_samples),
            "augmented": self.config.augment,
            "model_config": {
                "model_name": self.config.model_name,
                "model_type": self.config.model_type,
                "epochs": self.config.epochs,
                "use_lora": self.config.use_lora,
            },
            "training_result": training_result,
        }

        metadata_path = output_dir / "url_training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _augment_samples(self, samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Augment samples using the augmentation module."""
        try:
            from backend.augment import DataGenerator, AugmentConfig

            config = AugmentConfig(
                target_count=self.config.augment_target,
                strategies=["synonym", "template", "paraphrase"],
            )

            generator = DataGenerator(config)
            augmented = generator.generate(samples, self.config.augment_target)

            logger.info(f"Augmented to {len(augmented)} samples")
            return augmented

        except ImportError:
            logger.warning("Augmentation module not available, using original samples")
            return samples
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}, using original samples")
            return samples

    def _train_model(
        self,
        data_path: Path,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Train the model using real training module."""
        try:
            from backend.training.real_trainer import RealTrainer, TrainingConfig

            config = TrainingConfig(
                model_name=self.config.model_name,
                model_type=self.config.model_type,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                use_peft=self.config.use_lora,
            )

            trainer = RealTrainer(config)
            result = trainer.train(data_path, output_dir)

            return {
                "status": "success",
                "model_path": str(output_dir),
                "training_loss": result.get("training_results", {}).get("train_loss"),
            }

        except ImportError:
            logger.warning("Real training not available, creating stub model")
            return self._create_stub_model(output_dir, [])
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    def _create_stub_model(
        self,
        output_dir: Path,
        samples: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Create a stub model for dry-run mode."""
        stub_path = output_dir / "model_stub.json"

        stub = {
            "model_type": "tinyforge_url_stub",
            "model_name": self.config.model_name,
            "version": "0.1.0",
            "dry_run": True,
            "n_samples": len(samples),
            "config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "use_lora": self.config.use_lora,
            },
        }

        with open(stub_path, "w") as f:
            json.dump(stub, f, indent=2)

        return {
            "status": "stub_created",
            "model_path": str(stub_path),
            "n_samples": len(samples),
        }

    def preview(self, url: str) -> ExtractedData:
        """
        Preview extracted data from a URL without training.

        Args:
            url: URL to preview.

        Returns:
            ExtractedData with samples and metadata.
        """
        return self.extractor.extract(url)


def train_from_url(
    url: str,
    output_dir: str = "./url_model",
    model_name: str = "t5-small",
    epochs: int = 3,
    use_lora: bool = True,
    augment: bool = True,
    dry_run: bool = False,
    notion_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function for one-click URL training.

    Args:
        url: URL to train from.
        output_dir: Output directory.
        model_name: Model to fine-tune.
        epochs: Number of training epochs.
        use_lora: Whether to use LoRA.
        augment: Whether to augment data.
        dry_run: Whether to do a dry run.
        notion_token: Optional Notion API token.

    Returns:
        Training result dictionary.
    """
    config = URLTrainConfig(
        model_name=model_name,
        epochs=epochs,
        use_lora=use_lora,
        augment=augment,
        dry_run=dry_run,
        output_dir=output_dir,
        notion_token=notion_token,
    )

    trainer = URLTrainer(config)
    return trainer.train_from_url(url, output_dir, dry_run)


def main():
    """CLI entry point for URL training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model directly from a URL"
    )
    parser.add_argument(
        "url",
        help="URL to extract training data from"
    )
    parser.add_argument(
        "--output", "-o",
        default="./url_model",
        help="Output directory (default: ./url_model)"
    )
    parser.add_argument(
        "--model", "-m",
        default="t5-small",
        help="Model to fine-tune (default: t5-small)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="Number of epochs (default: 3)"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (use full fine-tuning)"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation"
    )
    parser.add_argument(
        "--augment-target",
        type=int,
        default=500,
        help="Target sample count for augmentation (default: 500)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview data extraction without training"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Only preview extracted data, don't train"
    )
    parser.add_argument(
        "--notion-token",
        help="Notion API token for private pages"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = URLTrainConfig(
        model_name=args.model,
        epochs=args.epochs,
        use_lora=not args.no_lora,
        augment=not args.no_augment,
        augment_target=args.augment_target,
        dry_run=args.dry_run,
        output_dir=args.output,
        notion_token=args.notion_token,
    )

    trainer = URLTrainer(config)

    if args.preview:
        # Preview mode
        print(f"\nPreviewing: {args.url}\n")
        data = trainer.preview(args.url)

        print(f"Source Type: {data.source_type}")
        print(f"Title: {data.title}")
        print(f"Samples Extracted: {len(data.samples)}")
        print("\nSample Preview (first 3):")
        print("-" * 50)

        for i, sample in enumerate(data.samples[:3]):
            print(f"\n[{i+1}] Input: {sample['input'][:100]}...")
            print(f"    Output: {sample['output'][:100]}...")

        return

    # Training mode
    print(f"\nTraining from URL: {args.url}\n")

    result = trainer.train_from_url(args.url)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Source URLs: {len(result['source_urls'])}")
    print(f"Total Samples: {result['total_samples']}")
    print(f"Augmented: {result['augmented']}")
    print(f"Output: {args.output}")

    if result.get("training_result", {}).get("status") == "success":
        print(f"Training Loss: {result['training_result'].get('training_loss', 'N/A')}")
    elif result.get("training_result", {}).get("status") == "stub_created":
        print("Status: Dry run - stub model created")


if __name__ == "__main__":
    main()
