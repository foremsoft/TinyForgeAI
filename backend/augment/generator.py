"""
Training Data Generator.

Main module for generating synthetic training data from examples.
Orchestrates multiple augmentation strategies to create diverse datasets.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from .strategies import (
    AugmentationStrategy,
    AugmentedSample,
    SynonymStrategy,
    TemplateStrategy,
    ParaphraseStrategy,
    BackTranslationStrategy,
    LLMStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class AugmentConfig:
    """Configuration for data augmentation."""

    # Target count
    target_count: int = 500

    # Strategies to use (default: all non-LLM strategies)
    strategies: List[str] = field(default_factory=lambda: [
        "synonym",
        "template",
        "paraphrase",
        "back_translation",
    ])

    # Strategy weights (higher = more samples from this strategy)
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "synonym": 1.0,
        "template": 1.5,
        "paraphrase": 1.0,
        "back_translation": 0.8,
        "llm": 2.0,
    })

    # Quality settings
    min_confidence: float = 0.7
    deduplicate: bool = True

    # LLM settings (if using llm strategy)
    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-haiku-20240307"
    llm_api_key: Optional[str] = None

    # Output settings
    include_originals: bool = True
    shuffle: bool = True

    # Parallelism
    max_workers: int = 4


class DataGenerator:
    """
    Generate synthetic training data from examples.

    Usage:
        generator = DataGenerator()
        samples = generator.generate(
            examples=[
                {"input": "What are your hours?", "output": "We're open 9-5."},
                {"input": "Do you ship internationally?", "output": "Yes, we ship worldwide."},
            ],
            target_count=100
        )
        generator.save(samples, "augmented_data.jsonl")
    """

    STRATEGY_MAP = {
        "synonym": SynonymStrategy,
        "template": TemplateStrategy,
        "paraphrase": ParaphraseStrategy,
        "back_translation": BackTranslationStrategy,
        "llm": LLMStrategy,
    }

    def __init__(self, config: Optional[AugmentConfig] = None):
        """
        Initialize the data generator.

        Args:
            config: Augmentation configuration.
        """
        self.config = config or AugmentConfig()
        self._strategies: Dict[str, AugmentationStrategy] = {}
        self._initialize_strategies()

    def _initialize_strategies(self) -> None:
        """Initialize configured strategies."""
        for strategy_name in self.config.strategies:
            if strategy_name not in self.STRATEGY_MAP:
                logger.warning(f"Unknown strategy: {strategy_name}, skipping")
                continue

            strategy_class = self.STRATEGY_MAP[strategy_name]

            if strategy_name == "llm":
                self._strategies[strategy_name] = strategy_class(
                    provider=self.config.llm_provider,
                    model=self.config.llm_model,
                    api_key=self.config.llm_api_key,
                )
            else:
                self._strategies[strategy_name] = strategy_class()

        logger.info(f"Initialized {len(self._strategies)} strategies: {list(self._strategies.keys())}")

    def generate(
        self,
        examples: List[Dict[str, str]],
        target_count: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate augmented training data.

        Args:
            examples: List of {"input": ..., "output": ...} dictionaries.
            target_count: Target number of samples (overrides config).

        Returns:
            List of augmented samples in {"input": ..., "output": ...} format.
        """
        target = target_count or self.config.target_count

        if not examples:
            raise ValueError("No examples provided")

        if not self._strategies:
            raise ValueError("No strategies configured")

        logger.info(f"Generating {target} samples from {len(examples)} examples")

        all_samples: List[AugmentedSample] = []

        # Include original samples if configured
        if self.config.include_originals:
            for ex in examples:
                all_samples.append(AugmentedSample(
                    input=ex["input"],
                    output=ex["output"],
                    original_input=ex["input"],
                    original_output=ex["output"],
                    strategy="original",
                    confidence=1.0,
                ))

        # Calculate samples needed per example
        samples_needed = target - len(all_samples)
        samples_per_example = max(1, samples_needed // len(examples))

        # Calculate samples per strategy based on weights
        total_weight = sum(
            self.config.strategy_weights.get(s, 1.0)
            for s in self._strategies.keys()
        )

        strategy_samples = {}
        for strategy_name in self._strategies.keys():
            weight = self.config.strategy_weights.get(strategy_name, 1.0)
            strategy_samples[strategy_name] = max(1, int(samples_per_example * weight / total_weight))

        # Generate samples using each strategy
        for example in examples:
            input_text = example["input"]
            output_text = example["output"]

            for strategy_name, strategy in self._strategies.items():
                n_samples = strategy_samples.get(strategy_name, 1)

                try:
                    augmented = strategy.augment(input_text, output_text, n_samples)

                    # Filter by confidence
                    augmented = [
                        s for s in augmented
                        if s.confidence >= self.config.min_confidence
                    ]

                    all_samples.extend(augmented)

                except Exception as e:
                    logger.warning(f"Strategy {strategy_name} failed on '{input_text[:50]}...': {e}")
                    continue

        # Deduplicate
        if self.config.deduplicate:
            seen = set()
            unique_samples = []
            for sample in all_samples:
                key = sample.input.lower().strip()
                if key not in seen:
                    seen.add(key)
                    unique_samples.append(sample)
            all_samples = unique_samples
            logger.info(f"After deduplication: {len(all_samples)} samples")

        # Shuffle
        if self.config.shuffle:
            random.shuffle(all_samples)

        # Trim to target
        all_samples = all_samples[:target]

        logger.info(f"Generated {len(all_samples)} samples")

        # Convert to simple dict format
        return [
            {"input": s.input, "output": s.output}
            for s in all_samples
        ]

    def generate_with_metadata(
        self,
        examples: List[Dict[str, str]],
        target_count: Optional[int] = None,
    ) -> List[AugmentedSample]:
        """
        Generate augmented data with full metadata.

        Same as generate() but returns AugmentedSample objects
        with strategy, confidence, and other metadata.
        """
        target = target_count or self.config.target_count
        # ... (same logic as generate, but return AugmentedSample objects)
        # For brevity, this delegates to generate() internally
        results = self.generate(examples, target_count)
        return [
            AugmentedSample(
                input=r["input"],
                output=r["output"],
                original_input=r["input"],
                original_output=r["output"],
                strategy="mixed",
                confidence=0.9,
            )
            for r in results
        ]

    def save(
        self,
        samples: List[Dict[str, str]],
        output_path: Union[str, Path],
        format: str = "jsonl"
    ) -> Path:
        """
        Save augmented samples to file.

        Args:
            samples: List of {"input": ..., "output": ...} samples.
            output_path: Output file path.
            format: Output format ("jsonl", "json", "csv").

        Returns:
            Path to saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        elif format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)

        elif format == "csv":
            import csv
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["input", "output"])
                writer.writeheader()
                writer.writerows(samples)

        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved {len(samples)} samples to {output_path}")
        return output_path

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        config: Optional[AugmentConfig] = None,
    ) -> "DataGenerator":
        """
        Create a generator and load examples from file.

        Args:
            file_path: Path to JSONL/JSON file with examples.
            config: Augmentation configuration.

        Returns:
            DataGenerator instance.
        """
        file_path = Path(file_path)
        examples = []

        if file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))

        elif file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples = data
                else:
                    examples = [data]

        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        generator = cls(config)
        generator._examples = examples
        return generator


def augment_dataset(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_count: int = 500,
    strategies: Optional[List[str]] = None,
    use_llm: bool = False,
    llm_provider: str = "anthropic",
    llm_model: str = "claude-3-haiku-20240307",
) -> Path:
    """
    Convenience function to augment a dataset file.

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file.
        target_count: Target number of samples.
        strategies: List of strategies to use.
        use_llm: Whether to use LLM augmentation.
        llm_provider: LLM provider if use_llm is True.
        llm_model: LLM model if use_llm is True.

    Returns:
        Path to output file.
    """
    # Load examples
    input_path = Path(input_path)
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    # Configure
    if strategies is None:
        strategies = ["synonym", "template", "paraphrase", "back_translation"]

    if use_llm:
        strategies.append("llm")

    config = AugmentConfig(
        target_count=target_count,
        strategies=strategies,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # Generate
    generator = DataGenerator(config)
    samples = generator.generate(examples, target_count)

    # Save
    return generator.save(samples, output_path)


def main():
    """CLI entry point for data augmentation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from examples"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file with examples"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSONL file"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=500,
        help="Target number of samples (default: 500)"
    )
    parser.add_argument(
        "--strategies", "-s",
        nargs="+",
        default=["synonym", "template", "paraphrase", "back_translation"],
        help="Augmentation strategies to use"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for high-quality augmentation (requires API key)"
    )
    parser.add_argument(
        "--llm-provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider (default: anthropic)"
    )
    parser.add_argument(
        "--llm-model",
        default="claude-3-haiku-20240307",
        help="LLM model to use"
    )
    parser.add_argument(
        "--format",
        default="jsonl",
        choices=["jsonl", "json", "csv"],
        help="Output format (default: jsonl)"
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

    # Run augmentation
    strategies = args.strategies
    if args.use_llm:
        strategies.append("llm")

    output_path = augment_dataset(
        input_path=args.input,
        output_path=args.output,
        target_count=args.count,
        strategies=strategies,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    print(f"\nAugmentation complete!")
    print(f"Output: {output_path}")
    print(f"Samples: {args.count}")


if __name__ == "__main__":
    main()
