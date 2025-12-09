"""
TinyForgeAI Model Evaluator

Main evaluation class for comprehensive model assessment.
Supports text generation, classification, and QA models.

Usage:
    from backend.evaluation import ModelEvaluator, EvaluationConfig

    config = EvaluationConfig(
        task_type="generation",
        batch_size=8,
        max_samples=1000
    )
    evaluator = ModelEvaluator(model_path="./my_model", config=config)
    results = evaluator.evaluate(test_data="test.jsonl")
    print(results.summary())
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

from backend.evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    compute_accuracy,
    compute_f1,
    compute_perplexity,
    compute_exact_match,
    compute_all_metrics,
)

logger = logging.getLogger(__name__)

# Check for dependencies
EVALUATION_AVAILABLE = False
try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
    EVALUATION_AVAILABLE = True
except ImportError:
    logger.warning(
        "Evaluation dependencies not available. "
        "Install with: pip install transformers torch"
    )


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Task settings
    task_type: str = "generation"  # "generation", "classification", "qa"

    # Input/output settings
    input_field: str = "input"
    target_field: str = "output"
    prediction_field: str = "prediction"

    # Generation settings
    max_new_tokens: int = 128
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0

    # Processing settings
    batch_size: int = 8
    max_length: int = 512
    max_samples: Optional[int] = None

    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: ["bleu", "rouge", "exact_match"])

    # Output settings
    save_predictions: bool = True
    output_dir: Optional[str] = None


@dataclass
class EvaluationResults:
    """Results from model evaluation."""

    model_path: str
    task_type: str
    metrics: Dict[str, Any]
    num_samples: int
    evaluation_time: float
    predictions: Optional[List[Dict[str, str]]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    config: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            f"Model: {self.model_path}",
            f"Task: {self.task_type}",
            f"Samples: {self.num_samples}",
            f"Time: {self.evaluation_time:.2f}s",
            "-" * 60,
            "METRICS:",
        ]

        for metric_name, metric_value in self.metrics.items():
            if isinstance(metric_value, dict):
                lines.append(f"  {metric_name}:")
                for k, v in metric_value.items():
                    if isinstance(v, float):
                        lines.append(f"    {k}: {v:.4f}")
                    elif isinstance(v, dict):
                        lines.append(f"    {k}:")
                        for kk, vv in v.items():
                            if isinstance(vv, float):
                                lines.append(f"      {kk}: {vv:.4f}")
                    else:
                        lines.append(f"    {k}: {v}")
            elif isinstance(metric_value, float):
                lines.append(f"  {metric_name}: {metric_value:.4f}")
            else:
                lines.append(f"  {metric_name}: {metric_value}")

        lines.append("=" * 60)

        if self.errors:
            lines.append("ERRORS:")
            for error in self.errors:
                lines.append(f"  - {error}")
            lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_path": self.model_path,
            "task_type": self.task_type,
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "evaluation_time": self.evaluation_time,
            "timestamp": self.timestamp,
            "config": self.config,
            "errors": self.errors,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        if self.predictions:
            data["predictions"] = self.predictions

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {path}")


class ModelEvaluator:
    """
    Comprehensive model evaluator for TinyForgeAI models.

    Supports:
    - Text generation models (T5, BART, GPT-2, etc.)
    - Classification models
    - Question answering models
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Any = None,
        tokenizer: Any = None,
        config: Optional[EvaluationConfig] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to model directory or HuggingFace model name
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            config: Evaluation configuration
        """
        self.model_path = model_path
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EvaluationConfig()
        self.device = None

        if not EVALUATION_AVAILABLE:
            logger.warning("Evaluation dependencies not available")

    def load_model(self) -> None:
        """Load model and tokenizer."""
        if not EVALUATION_AVAILABLE:
            raise RuntimeError("Evaluation dependencies not installed")

        if self.model is not None:
            logger.info("Using pre-loaded model")
            return

        if self.model_path is None:
            raise ValueError("model_path required when model not provided")

        logger.info(f"Loading model from {self.model_path}")

        # Determine device
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Load model based on task type
        if self.config.task_type in ["generation", "qa"]:
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        else:
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded on {self.device}")

    def load_test_data(
        self,
        data: Union[str, Path, List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        """
        Load test data from file or list.

        Args:
            data: Path to JSONL file or list of dicts

        Returns:
            List of test samples
        """
        if isinstance(data, list):
            samples = data
        else:
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Test data not found: {path}")

            samples = []
            if path.suffix == ".jsonl":
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
            elif path.suffix == ".json":
                with open(path) as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        samples = loaded
                    else:
                        samples = [loaded]
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        # Limit samples if specified
        if self.config.max_samples and len(samples) > self.config.max_samples:
            samples = samples[:self.config.max_samples]

        logger.info(f"Loaded {len(samples)} test samples")
        return samples

    def generate_predictions(
        self,
        samples: List[Dict[str, str]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """
        Generate predictions for test samples.

        Args:
            samples: List of test samples
            progress_callback: Optional callback for progress updates

        Returns:
            List of generated predictions
        """
        if not EVALUATION_AVAILABLE:
            raise RuntimeError("Evaluation dependencies not installed")

        import torch

        predictions = []
        input_field = self.config.input_field

        for i in range(0, len(samples), self.config.batch_size):
            batch = samples[i:i + self.config.batch_size]
            inputs = [s.get(input_field, "") for s in batch]

            # Tokenize
            encodings = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )

            input_ids = encodings.input_ids.to(self.device)
            attention_mask = encodings.attention_mask.to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    num_beams=self.config.num_beams,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode
            batch_predictions = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            predictions.extend(batch_predictions)

            if progress_callback:
                progress_callback(min(i + self.config.batch_size, len(samples)), len(samples))

        return predictions

    def evaluate(
        self,
        test_data: Union[str, Path, List[Dict[str, str]]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> EvaluationResults:
        """
        Run full evaluation on test data.

        Args:
            test_data: Path to test file or list of samples
            progress_callback: Optional callback for progress updates

        Returns:
            EvaluationResults with all computed metrics
        """
        start_time = time.time()
        errors = []

        # Load model if needed
        if self.model is None:
            self.load_model()

        # Load test data
        samples = self.load_test_data(test_data)

        if len(samples) == 0:
            return EvaluationResults(
                model_path=self.model_path or "unknown",
                task_type=self.config.task_type,
                metrics={},
                num_samples=0,
                evaluation_time=0.0,
                errors=["No test samples provided"],
            )

        # Get references
        target_field = self.config.target_field
        references = [s.get(target_field, "") for s in samples]

        # Generate predictions or use existing
        if self.config.prediction_field in samples[0]:
            predictions = [s[self.config.prediction_field] for s in samples]
            logger.info("Using existing predictions from data")
        else:
            logger.info("Generating predictions...")
            try:
                predictions = self.generate_predictions(samples, progress_callback)
            except Exception as e:
                logger.error(f"Error generating predictions: {e}")
                errors.append(f"Prediction error: {str(e)}")
                predictions = [""] * len(samples)

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = {}

        try:
            if "bleu" in self.config.metrics:
                metrics["bleu"] = compute_bleu(predictions, references)
        except Exception as e:
            errors.append(f"BLEU error: {str(e)}")

        try:
            if "rouge" in self.config.metrics:
                metrics["rouge"] = compute_rouge(predictions, references)
        except Exception as e:
            errors.append(f"ROUGE error: {str(e)}")

        try:
            if "exact_match" in self.config.metrics:
                metrics["exact_match"] = compute_exact_match(predictions, references)
        except Exception as e:
            errors.append(f"Exact match error: {str(e)}")

        try:
            if "accuracy" in self.config.metrics:
                metrics["accuracy"] = compute_accuracy(predictions, references)
        except Exception as e:
            errors.append(f"Accuracy error: {str(e)}")

        try:
            if "f1" in self.config.metrics:
                metrics["f1"] = compute_f1(predictions, references)
        except Exception as e:
            errors.append(f"F1 error: {str(e)}")

        try:
            if "perplexity" in self.config.metrics and self.model is not None:
                texts = [s.get(target_field, "") for s in samples]
                metrics["perplexity"] = compute_perplexity(
                    self.model, texts, self.tokenizer
                )
        except Exception as e:
            errors.append(f"Perplexity error: {str(e)}")

        # Build predictions list if saving
        predictions_data = None
        if self.config.save_predictions:
            predictions_data = [
                {
                    "input": s.get(self.config.input_field, ""),
                    "reference": s.get(target_field, ""),
                    "prediction": p,
                }
                for s, p in zip(samples, predictions)
            ]

        evaluation_time = time.time() - start_time

        results = EvaluationResults(
            model_path=self.model_path or "unknown",
            task_type=self.config.task_type,
            metrics=metrics,
            num_samples=len(samples),
            evaluation_time=evaluation_time,
            predictions=predictions_data,
            config=self.config.__dict__,
            errors=errors,
        )

        # Save results if output_dir specified
        if self.config.output_dir:
            output_path = Path(self.config.output_dir) / "evaluation_results.json"
            results.save(output_path)

        return results

    def evaluate_from_predictions(
        self,
        predictions: List[str],
        references: List[str],
    ) -> EvaluationResults:
        """
        Evaluate pre-generated predictions against references.

        Args:
            predictions: List of predicted outputs
            references: List of reference outputs

        Returns:
            EvaluationResults with computed metrics
        """
        start_time = time.time()

        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        metrics = compute_all_metrics(
            predictions, references, self.config.task_type
        )

        evaluation_time = time.time() - start_time

        return EvaluationResults(
            model_path=self.model_path or "precomputed",
            task_type=self.config.task_type,
            metrics=metrics,
            num_samples=len(predictions),
            evaluation_time=evaluation_time,
        )
