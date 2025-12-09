"""
TinyForgeAI Model Evaluation Module

Provides comprehensive evaluation metrics and benchmarking tools for
trained models including text generation, classification, and embedding models.

Usage:
    from backend.evaluation import ModelEvaluator, EvaluationConfig

    evaluator = ModelEvaluator(model_path="./my_model")
    results = evaluator.evaluate(test_data="test.jsonl")
    print(results.summary())
"""

from backend.evaluation.evaluator import (
    ModelEvaluator,
    EvaluationConfig,
    EvaluationResults,
    EVALUATION_AVAILABLE,
)
from backend.evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    compute_accuracy,
    compute_f1,
    compute_perplexity,
    compute_exact_match,
)
from backend.evaluation.benchmark import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkResult,
)

__all__ = [
    # Evaluator
    "ModelEvaluator",
    "EvaluationConfig",
    "EvaluationResults",
    "EVALUATION_AVAILABLE",
    # Metrics
    "compute_bleu",
    "compute_rouge",
    "compute_accuracy",
    "compute_f1",
    "compute_perplexity",
    "compute_exact_match",
    # Benchmark
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
]
