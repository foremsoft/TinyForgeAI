"""
TinyForgeAI Benchmark Runner

Provides benchmarking tools to compare models, track performance over time,
and run standardized evaluation suites.

Usage:
    from backend.evaluation import BenchmarkRunner, BenchmarkConfig

    config = BenchmarkConfig(
        benchmark_name="my_benchmark",
        datasets=["test_set_1.jsonl", "test_set_2.jsonl"],
    )
    runner = BenchmarkRunner(config)
    results = runner.run(models=["./model_v1", "./model_v2"])
    print(results.comparison_table())
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.evaluation.evaluator import (
    ModelEvaluator,
    EvaluationConfig,
    EvaluationResults,
    EVALUATION_AVAILABLE,
)
from backend.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Benchmark identity
    benchmark_name: str = "tinyforge_benchmark"
    description: str = ""

    # Datasets to evaluate on
    datasets: List[str] = field(default_factory=list)

    # Evaluation settings
    task_type: str = "generation"
    metrics: List[str] = field(default_factory=lambda: ["bleu", "rouge", "exact_match"])
    batch_size: int = 8
    max_samples_per_dataset: Optional[int] = None

    # Output settings
    output_dir: str = "./benchmark_results"
    save_predictions: bool = False

    # Comparison settings
    baseline_model: Optional[str] = None


@dataclass
class ModelBenchmarkResult:
    """Results for a single model on the benchmark."""

    model_name: str
    model_path: str
    dataset_results: Dict[str, EvaluationResults]
    aggregate_metrics: Dict[str, float]
    total_time: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "dataset_results": {
                name: result.to_dict()
                for name, result in self.dataset_results.items()
            },
            "aggregate_metrics": self.aggregate_metrics,
            "total_time": self.total_time,
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkResult:
    """Overall benchmark results comparing multiple models."""

    benchmark_name: str
    config: BenchmarkConfig
    model_results: Dict[str, ModelBenchmarkResult]
    comparison: Dict[str, Dict[str, float]]
    best_model: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def comparison_table(self, metric: str = "bleu") -> str:
        """
        Generate comparison table for models.

        Args:
            metric: Metric to compare on

        Returns:
            Formatted table string
        """
        lines = [
            "=" * 80,
            f"BENCHMARK: {self.benchmark_name}",
            f"Comparing on: {metric}",
            "=" * 80,
        ]

        # Header
        datasets = list(next(iter(self.model_results.values())).dataset_results.keys())
        header = ["Model"] + datasets + ["Average"]
        col_widths = [max(20, len(h) + 2) for h in header]

        header_line = "".join(h.ljust(w) for h, w in zip(header, col_widths))
        lines.append(header_line)
        lines.append("-" * len(header_line))

        # Model rows
        for model_name, result in self.model_results.items():
            row = [model_name[:18]]

            for dataset in datasets:
                if dataset in result.dataset_results:
                    ds_result = result.dataset_results[dataset]
                    score = self._extract_metric(ds_result.metrics, metric)
                    row.append(f"{score:.4f}")
                else:
                    row.append("N/A")

            # Average
            avg = result.aggregate_metrics.get(metric, 0.0)
            row.append(f"{avg:.4f}")

            row_line = "".join(str(v).ljust(w) for v, w in zip(row, col_widths))
            if model_name == self.best_model:
                row_line += " *BEST*"
            lines.append(row_line)

        lines.append("=" * 80)

        return "\n".join(lines)

    def _extract_metric(self, metrics: Dict[str, Any], metric_name: str) -> float:
        """Extract a metric value from nested metrics dict."""
        if metric_name in metrics:
            value = metrics[metric_name]
            if isinstance(value, dict):
                # Return main score (e.g., bleu from bleu dict)
                return value.get(metric_name, value.get("fmeasure", 0.0))
            return float(value)

        # Try nested access
        for key, value in metrics.items():
            if isinstance(value, dict) and metric_name in value:
                v = value[metric_name]
                if isinstance(v, dict):
                    return v.get("fmeasure", 0.0)
                return float(v)

        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "config": self.config.__dict__,
            "model_results": {
                name: result.to_dict()
                for name, result in self.model_results.items()
            },
            "comparison": self.comparison,
            "best_model": self.best_model,
            "timestamp": self.timestamp,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save benchmark results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Benchmark results saved to {path}")


class BenchmarkRunner:
    """
    Runs benchmarks across multiple models and datasets.

    Provides:
    - Multi-model comparison
    - Multi-dataset evaluation
    - Aggregate metrics computation
    - Performance regression detection
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.eval_config = EvaluationConfig(
            task_type=self.config.task_type,
            metrics=self.config.metrics,
            batch_size=self.config.batch_size,
            max_samples=self.config.max_samples_per_dataset,
            save_predictions=self.config.save_predictions,
        )

    def run(
        self,
        models: List[str],
        datasets: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        """
        Run benchmark on multiple models.

        Args:
            models: List of model paths to evaluate
            datasets: List of dataset paths (overrides config)

        Returns:
            BenchmarkResult with all comparisons
        """
        datasets = datasets or self.config.datasets
        if not datasets:
            raise ValueError("No datasets provided for benchmark")

        start_time = time.time()
        model_results: Dict[str, ModelBenchmarkResult] = {}

        for model_path in models:
            logger.info(f"Evaluating model: {model_path}")
            model_result = self._evaluate_model(model_path, datasets)
            model_name = Path(model_path).name
            model_results[model_name] = model_result

        # Compute comparisons
        comparison = self._compute_comparison(model_results)

        # Find best model
        best_model = self._find_best_model(model_results)

        result = BenchmarkResult(
            benchmark_name=self.config.benchmark_name,
            config=self.config,
            model_results=model_results,
            comparison=comparison,
            best_model=best_model,
        )

        # Save results
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.save(output_dir / f"{self.config.benchmark_name}_results.json")

        total_time = time.time() - start_time
        logger.info(f"Benchmark completed in {total_time:.2f}s")

        return result

    def _evaluate_model(
        self,
        model_path: str,
        datasets: List[str],
    ) -> ModelBenchmarkResult:
        """Evaluate a single model on all datasets."""
        start_time = time.time()
        dataset_results: Dict[str, EvaluationResults] = {}

        evaluator = ModelEvaluator(
            model_path=model_path,
            config=self.eval_config,
        )

        for dataset_path in datasets:
            dataset_name = Path(dataset_path).stem
            logger.info(f"  Evaluating on: {dataset_name}")

            try:
                result = evaluator.evaluate(dataset_path)
                dataset_results[dataset_name] = result
            except Exception as e:
                logger.error(f"  Error evaluating {dataset_name}: {e}")
                dataset_results[dataset_name] = EvaluationResults(
                    model_path=model_path,
                    task_type=self.config.task_type,
                    metrics={},
                    num_samples=0,
                    evaluation_time=0.0,
                    errors=[str(e)],
                )

        # Compute aggregate metrics
        aggregate_metrics = self._compute_aggregate_metrics(dataset_results)

        total_time = time.time() - start_time

        return ModelBenchmarkResult(
            model_name=Path(model_path).name,
            model_path=model_path,
            dataset_results=dataset_results,
            aggregate_metrics=aggregate_metrics,
            total_time=total_time,
        )

    def _compute_aggregate_metrics(
        self,
        dataset_results: Dict[str, EvaluationResults],
    ) -> Dict[str, float]:
        """Compute average metrics across all datasets."""
        aggregate: Dict[str, List[float]] = {}

        for result in dataset_results.values():
            for metric_name, metric_value in result.metrics.items():
                if isinstance(metric_value, dict):
                    # Handle nested metrics (e.g., bleu -> bleu_4)
                    for sub_name, sub_value in metric_value.items():
                        if isinstance(sub_value, (int, float)):
                            key = f"{metric_name}_{sub_name}" if sub_name != metric_name else metric_name
                            aggregate.setdefault(key, []).append(float(sub_value))
                        elif isinstance(sub_value, dict) and "fmeasure" in sub_value:
                            key = f"{metric_name}_{sub_name}"
                            aggregate.setdefault(key, []).append(float(sub_value["fmeasure"]))
                elif isinstance(metric_value, (int, float)):
                    aggregate.setdefault(metric_name, []).append(float(metric_value))

        # Compute averages
        return {
            name: sum(values) / len(values)
            for name, values in aggregate.items()
            if values
        }

    def _compute_comparison(
        self,
        model_results: Dict[str, ModelBenchmarkResult],
    ) -> Dict[str, Dict[str, float]]:
        """Compute comparison metrics between models."""
        comparison: Dict[str, Dict[str, float]] = {}

        for model_name, result in model_results.items():
            comparison[model_name] = result.aggregate_metrics.copy()

        # If baseline specified, compute deltas
        if self.config.baseline_model and self.config.baseline_model in comparison:
            baseline = comparison[self.config.baseline_model]
            for model_name in comparison:
                if model_name != self.config.baseline_model:
                    deltas = {}
                    for metric, value in comparison[model_name].items():
                        if metric in baseline:
                            deltas[f"{metric}_delta"] = value - baseline[metric]
                    comparison[model_name].update(deltas)

        return comparison

    def _find_best_model(
        self,
        model_results: Dict[str, ModelBenchmarkResult],
        primary_metric: str = "bleu",
    ) -> Optional[str]:
        """Find the best performing model."""
        best_model = None
        best_score = -float("inf")

        for model_name, result in model_results.items():
            score = result.aggregate_metrics.get(primary_metric, 0.0)
            if score > best_score:
                best_score = score
                best_model = model_name

        return best_model

    def run_single(
        self,
        model_path: str,
        test_data: Union[str, Path, List[Dict[str, str]]],
    ) -> EvaluationResults:
        """
        Run evaluation on a single model and dataset.

        Convenience method for quick evaluations.

        Args:
            model_path: Path to model
            test_data: Test data path or list

        Returns:
            EvaluationResults for the model
        """
        evaluator = ModelEvaluator(
            model_path=model_path,
            config=self.eval_config,
        )
        return evaluator.evaluate(test_data)


# ============================================
# Built-in Benchmark Datasets
# ============================================

BUILTIN_BENCHMARKS = {
    "tinyforge_qa": {
        "description": "Question answering benchmark for TinyForgeAI",
        "task_type": "qa",
        "metrics": ["exact_match", "f1"],
        "datasets": [],  # Would be populated with actual datasets
    },
    "tinyforge_generation": {
        "description": "Text generation benchmark for TinyForgeAI",
        "task_type": "generation",
        "metrics": ["bleu", "rouge"],
        "datasets": [],
    },
    "tinyforge_summarization": {
        "description": "Summarization benchmark for TinyForgeAI",
        "task_type": "generation",
        "metrics": ["rouge"],
        "datasets": [],
    },
}


def create_benchmark_dataset(
    name: str,
    samples: List[Dict[str, str]],
    output_dir: str = "./benchmark_data",
) -> str:
    """
    Create a benchmark dataset file.

    Args:
        name: Dataset name
        samples: List of samples with input/output fields
        output_dir: Output directory

    Returns:
        Path to created dataset file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{name}.jsonl"

    with open(file_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Created benchmark dataset: {file_path} ({len(samples)} samples)")
    return str(file_path)


def load_benchmark_dataset(path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Load a benchmark dataset.

    Args:
        path: Path to JSONL dataset file

    Returns:
        List of samples
    """
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples
