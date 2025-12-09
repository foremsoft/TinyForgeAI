"""
Tests for TinyForgeAI Evaluation Module

Tests metrics, evaluator, and benchmark functionality.
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, List

from backend.evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    compute_accuracy,
    compute_f1,
    compute_exact_match,
    compute_all_metrics,
    _tokenize,
    _get_ngrams,
    _lcs_length,
    _normalize_answer,
)
from backend.evaluation.evaluator import (
    EvaluationConfig,
    EvaluationResults,
    ModelEvaluator,
    EVALUATION_AVAILABLE,
)
from backend.evaluation.benchmark import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkResult,
    ModelBenchmarkResult,
    create_benchmark_dataset,
    load_benchmark_dataset,
)


class TestTokenization:
    """Test tokenization utilities."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = "Hello World"
        tokens = _tokenize(text)
        assert tokens == ["hello", "world"]

    def test_tokenize_with_punctuation(self):
        """Test tokenization removes punctuation."""
        text = "Hello, World!"
        tokens = _tokenize(text)
        assert tokens == ["hello", "world"]

    def test_tokenize_preserves_apostrophes(self):
        """Test tokenization preserves apostrophes."""
        text = "It's a test"
        tokens = _tokenize(text)
        assert "it's" in tokens

    def test_tokenize_empty(self):
        """Test tokenization of empty string."""
        tokens = _tokenize("")
        assert tokens == []

    def test_get_ngrams(self):
        """Test n-gram extraction."""
        tokens = ["a", "b", "c", "d"]
        unigrams = _get_ngrams(tokens, 1)
        bigrams = _get_ngrams(tokens, 2)
        trigrams = _get_ngrams(tokens, 3)

        assert len(unigrams) == 4
        assert len(bigrams) == 3
        assert len(trigrams) == 2

    def test_lcs_length(self):
        """Test longest common subsequence."""
        seq1 = ["a", "b", "c", "d"]
        seq2 = ["a", "c", "d"]
        assert _lcs_length(seq1, seq2) == 3

    def test_lcs_length_empty(self):
        """Test LCS with empty sequences."""
        assert _lcs_length([], ["a", "b"]) == 0
        assert _lcs_length(["a", "b"], []) == 0

    def test_normalize_answer(self):
        """Test answer normalization."""
        assert _normalize_answer("The Answer") == "answer"
        assert _normalize_answer("a test!") == "test"
        assert _normalize_answer("  spaces  ") == "spaces"


class TestBLEU:
    """Test BLEU score computation."""

    def test_bleu_perfect_match(self):
        """Test BLEU with perfect match."""
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        scores = compute_bleu(predictions, references)

        assert "bleu" in scores
        assert scores["bleu"] > 0.9  # Should be close to 1.0

    def test_bleu_partial_match(self):
        """Test BLEU with partial match."""
        predictions = ["the cat sat"]
        references = ["the cat sat on the mat"]
        scores = compute_bleu(predictions, references)

        assert scores["bleu_1"] > 0
        assert scores["bleu"] < 1.0

    def test_bleu_no_match(self):
        """Test BLEU with no match."""
        predictions = ["xyz abc"]
        references = ["the cat sat on the mat"]
        scores = compute_bleu(predictions, references)

        assert scores["bleu"] < 0.1

    def test_bleu_empty_input(self):
        """Test BLEU with empty input."""
        scores = compute_bleu([], [])
        assert scores["bleu"] == 0.0

    def test_bleu_multiple_references(self):
        """Test BLEU with multiple references."""
        predictions = ["the cat sat on the mat"]
        references = [["the cat sat on the mat", "a cat is sitting on a mat"]]
        scores = compute_bleu(predictions, references)

        assert scores["bleu"] > 0

    def test_bleu_length_mismatch(self):
        """Test BLEU raises on length mismatch."""
        with pytest.raises(ValueError):
            compute_bleu(["a", "b"], ["a"])


class TestROUGE:
    """Test ROUGE score computation."""

    def test_rouge_perfect_match(self):
        """Test ROUGE with perfect match."""
        predictions = ["the quick brown fox"]
        references = ["the quick brown fox"]
        scores = compute_rouge(predictions, references)

        assert "rouge1" in scores
        assert scores["rouge1"]["fmeasure"] > 0.9

    def test_rouge_partial_match(self):
        """Test ROUGE with partial match."""
        predictions = ["the quick brown"]
        references = ["the quick brown fox jumps"]
        scores = compute_rouge(predictions, references)

        assert scores["rouge1"]["precision"] > 0
        assert scores["rouge1"]["recall"] < 1.0

    def test_rouge_empty_input(self):
        """Test ROUGE with empty input."""
        scores = compute_rouge([], [])
        assert scores["rouge1"]["fmeasure"] == 0.0

    def test_rouge_l_lcs(self):
        """Test ROUGE-L uses LCS."""
        predictions = ["a b c d"]
        references = ["a c d e"]
        scores = compute_rouge(predictions, references, rouge_types=["rougeL"])

        assert "rougeL" in scores
        assert scores["rougeL"]["fmeasure"] > 0

    def test_rouge_length_mismatch(self):
        """Test ROUGE raises on length mismatch."""
        with pytest.raises(ValueError):
            compute_rouge(["a", "b"], ["a"])


class TestAccuracy:
    """Test accuracy computation."""

    def test_accuracy_perfect(self):
        """Test perfect accuracy."""
        predictions = ["a", "b", "c"]
        references = ["a", "b", "c"]
        scores = compute_accuracy(predictions, references)

        assert scores["accuracy"] == 1.0
        assert scores["correct"] == 3

    def test_accuracy_partial(self):
        """Test partial accuracy."""
        predictions = ["a", "b", "x"]
        references = ["a", "b", "c"]
        scores = compute_accuracy(predictions, references)

        assert scores["accuracy"] == pytest.approx(2/3)
        assert scores["correct"] == 2

    def test_accuracy_empty(self):
        """Test empty accuracy."""
        scores = compute_accuracy([], [])
        assert scores["accuracy"] == 0.0

    def test_accuracy_length_mismatch(self):
        """Test accuracy raises on mismatch."""
        with pytest.raises(ValueError):
            compute_accuracy(["a", "b"], ["a"])


class TestF1:
    """Test F1 score computation."""

    def test_f1_perfect(self):
        """Test perfect F1."""
        predictions = ["a", "b", "c"]
        references = ["a", "b", "c"]
        scores = compute_f1(predictions, references)

        assert scores["f1"] == 1.0
        assert scores["precision"] == 1.0
        assert scores["recall"] == 1.0

    def test_f1_partial(self):
        """Test partial F1."""
        predictions = ["a", "a", "b"]
        references = ["a", "b", "b"]
        scores = compute_f1(predictions, references)

        assert 0 < scores["f1"] < 1.0

    def test_f1_empty(self):
        """Test empty F1."""
        scores = compute_f1([], [])
        assert scores["f1"] == 0.0

    def test_f1_weighted(self):
        """Test weighted F1."""
        predictions = ["a", "a", "b"]
        references = ["a", "b", "b"]
        scores = compute_f1(predictions, references, average="weighted")

        assert 0 < scores["f1"] <= 1.0

    def test_f1_micro(self):
        """Test micro F1."""
        predictions = ["a", "b", "c"]
        references = ["a", "b", "c"]
        scores = compute_f1(predictions, references, average="micro")

        assert scores["f1"] == 1.0


class TestExactMatch:
    """Test exact match computation."""

    def test_exact_match_perfect(self):
        """Test perfect exact match."""
        predictions = ["answer", "test"]
        references = ["answer", "test"]
        scores = compute_exact_match(predictions, references)

        assert scores["exact_match"] == 1.0

    def test_exact_match_normalized(self):
        """Test normalized exact match."""
        predictions = ["The Answer", "A test"]
        references = ["the answer", "a test"]
        scores = compute_exact_match(predictions, references, normalize=True)

        assert scores["exact_match"] == 1.0

    def test_exact_match_no_normalize(self):
        """Test exact match without normalization."""
        predictions = ["The Answer"]
        references = ["the answer"]
        scores = compute_exact_match(predictions, references, normalize=False)

        assert scores["exact_match"] == 0.0

    def test_exact_match_empty(self):
        """Test empty exact match."""
        scores = compute_exact_match([], [])
        assert scores["exact_match"] == 0.0


class TestComputeAllMetrics:
    """Test compute_all_metrics function."""

    def test_generation_metrics(self):
        """Test generation task metrics."""
        predictions = ["the cat sat"]
        references = ["the cat sat on the mat"]
        metrics = compute_all_metrics(predictions, references, task_type="generation")

        assert "bleu" in metrics
        assert "rouge" in metrics
        assert "exact_match" in metrics

    def test_classification_metrics(self):
        """Test classification task metrics."""
        predictions = ["a", "b", "c"]
        references = ["a", "b", "c"]
        metrics = compute_all_metrics(predictions, references, task_type="classification")

        assert "accuracy" in metrics
        assert "f1" in metrics


class TestEvaluationConfig:
    """Test EvaluationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = EvaluationConfig()
        assert config.task_type == "generation"
        assert config.batch_size == 8
        assert "bleu" in config.metrics

    def test_custom_config(self):
        """Test custom configuration."""
        config = EvaluationConfig(
            task_type="classification",
            batch_size=16,
            metrics=["accuracy", "f1"],
        )
        assert config.task_type == "classification"
        assert config.batch_size == 16


class TestEvaluationResults:
    """Test EvaluationResults dataclass."""

    def test_results_summary(self):
        """Test results summary generation."""
        results = EvaluationResults(
            model_path="./test_model",
            task_type="generation",
            metrics={"bleu": {"bleu": 0.5, "bleu_1": 0.8}},
            num_samples=100,
            evaluation_time=10.5,
        )

        summary = results.summary()
        assert "test_model" in summary
        assert "100" in summary
        assert "bleu" in summary.lower()

    def test_results_to_dict(self):
        """Test results to dictionary conversion."""
        results = EvaluationResults(
            model_path="./test_model",
            task_type="generation",
            metrics={"bleu": 0.5},
            num_samples=100,
            evaluation_time=10.5,
        )

        d = results.to_dict()
        assert d["model_path"] == "./test_model"
        assert d["metrics"]["bleu"] == 0.5

    def test_results_save_load(self):
        """Test results save and load."""
        results = EvaluationResults(
            model_path="./test_model",
            task_type="generation",
            metrics={"bleu": 0.5},
            num_samples=100,
            evaluation_time=10.5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            results.save(path)

            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)

            assert loaded["model_path"] == "./test_model"


class TestModelEvaluator:
    """Test ModelEvaluator class."""

    def test_evaluator_init(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator(model_path="./test")
        assert evaluator.model_path == "./test"
        assert evaluator.config is not None

    def test_evaluator_with_config(self):
        """Test evaluator with custom config."""
        config = EvaluationConfig(task_type="qa")
        evaluator = ModelEvaluator(model_path="./test", config=config)
        assert evaluator.config.task_type == "qa"

    def test_load_test_data_jsonl(self):
        """Test loading JSONL test data."""
        evaluator = ModelEvaluator()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test.jsonl"
            with open(data_path, "w") as f:
                f.write('{"input": "hello", "output": "world"}\n')
                f.write('{"input": "foo", "output": "bar"}\n')

            samples = evaluator.load_test_data(data_path)
            assert len(samples) == 2
            assert samples[0]["input"] == "hello"

    def test_load_test_data_json(self):
        """Test loading JSON test data."""
        evaluator = ModelEvaluator()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "test.json"
            with open(data_path, "w") as f:
                json.dump([
                    {"input": "hello", "output": "world"},
                    {"input": "foo", "output": "bar"},
                ], f)

            samples = evaluator.load_test_data(data_path)
            assert len(samples) == 2

    def test_load_test_data_list(self):
        """Test loading test data from list."""
        evaluator = ModelEvaluator()
        samples = evaluator.load_test_data([
            {"input": "hello", "output": "world"},
        ])
        assert len(samples) == 1

    def test_load_test_data_max_samples(self):
        """Test max_samples limit."""
        config = EvaluationConfig(max_samples=1)
        evaluator = ModelEvaluator(config=config)

        samples = evaluator.load_test_data([
            {"input": "a", "output": "1"},
            {"input": "b", "output": "2"},
        ])
        assert len(samples) == 1

    def test_evaluate_from_predictions(self):
        """Test evaluation from pre-generated predictions."""
        evaluator = ModelEvaluator()
        predictions = ["the cat sat", "hello world"]
        references = ["the cat sat on the mat", "hello world"]

        results = evaluator.evaluate_from_predictions(predictions, references)

        assert results.num_samples == 2
        assert "bleu" in results.metrics or "rouge" in results.metrics


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""

    def test_default_config(self):
        """Test default benchmark config."""
        config = BenchmarkConfig()
        assert config.benchmark_name == "tinyforge_benchmark"
        assert config.task_type == "generation"

    def test_custom_config(self):
        """Test custom benchmark config."""
        config = BenchmarkConfig(
            benchmark_name="my_benchmark",
            datasets=["test1.jsonl", "test2.jsonl"],
        )
        assert config.benchmark_name == "my_benchmark"
        assert len(config.datasets) == 2


class TestBenchmarkDataset:
    """Test benchmark dataset creation and loading."""

    def test_create_benchmark_dataset(self):
        """Test creating benchmark dataset."""
        samples = [
            {"input": "hello", "output": "world"},
            {"input": "foo", "output": "bar"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_benchmark_dataset(
                name="test_dataset",
                samples=samples,
                output_dir=tmpdir,
            )

            assert Path(path).exists()

            loaded = load_benchmark_dataset(path)
            assert len(loaded) == 2
            assert loaded[0]["input"] == "hello"

    def test_load_benchmark_dataset(self):
        """Test loading benchmark dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            with open(path, "w") as f:
                f.write('{"input": "a", "output": "b"}\n')
                f.write('{"input": "c", "output": "d"}\n')

            samples = load_benchmark_dataset(path)
            assert len(samples) == 2


class TestBenchmarkRunner:
    """Test BenchmarkRunner class."""

    def test_runner_init(self):
        """Test runner initialization."""
        runner = BenchmarkRunner()
        assert runner.config is not None

    def test_runner_with_config(self):
        """Test runner with custom config."""
        config = BenchmarkConfig(benchmark_name="test")
        runner = BenchmarkRunner(config)
        assert runner.config.benchmark_name == "test"


class TestModelBenchmarkResult:
    """Test ModelBenchmarkResult dataclass."""

    def test_result_to_dict(self):
        """Test result to dict conversion."""
        result = ModelBenchmarkResult(
            model_name="test_model",
            model_path="./test",
            dataset_results={},
            aggregate_metrics={"bleu": 0.5},
            total_time=10.0,
        )

        d = result.to_dict()
        assert d["model_name"] == "test_model"
        assert d["aggregate_metrics"]["bleu"] == 0.5


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_comparison_table(self):
        """Test comparison table generation."""
        eval_result = EvaluationResults(
            model_path="./model1",
            task_type="generation",
            metrics={"bleu": {"bleu": 0.5}},
            num_samples=100,
            evaluation_time=10.0,
        )

        model_result = ModelBenchmarkResult(
            model_name="model1",
            model_path="./model1",
            dataset_results={"test": eval_result},
            aggregate_metrics={"bleu": 0.5},
            total_time=10.0,
        )

        benchmark_result = BenchmarkResult(
            benchmark_name="test_benchmark",
            config=BenchmarkConfig(),
            model_results={"model1": model_result},
            comparison={"model1": {"bleu": 0.5}},
            best_model="model1",
        )

        table = benchmark_result.comparison_table()
        assert "model1" in table
        assert "BENCHMARK" in table

    def test_result_save(self):
        """Test benchmark result save."""
        result = BenchmarkResult(
            benchmark_name="test",
            config=BenchmarkConfig(),
            model_results={},
            comparison={},
            best_model=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            result.save(path)

            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["benchmark_name"] == "test"


class TestEvaluationIntegration:
    """Integration tests for evaluation module."""

    def test_full_evaluation_flow(self):
        """Test complete evaluation workflow."""
        # Create test data
        samples = [
            {"input": "translate: hello", "output": "hola"},
            {"input": "translate: goodbye", "output": "adios"},
        ]

        config = EvaluationConfig(
            task_type="generation",
            metrics=["bleu", "exact_match"],
        )

        evaluator = ModelEvaluator(config=config)

        # Test with pre-existing predictions
        predictions = ["hola", "adios"]
        references = ["hola", "adios"]

        results = evaluator.evaluate_from_predictions(predictions, references)

        assert results.num_samples == 2
        assert results.metrics["exact_match"]["exact_match"] == 1.0

    def test_metrics_consistency(self):
        """Test that metrics are consistent across methods."""
        predictions = ["the quick brown fox", "hello world"]
        references = ["the quick brown fox", "hello world"]

        # Direct metric computation
        bleu1 = compute_bleu(predictions, references)
        rouge1 = compute_rouge(predictions, references)
        em1 = compute_exact_match(predictions, references)

        # Via compute_all_metrics
        all_metrics = compute_all_metrics(predictions, references, "generation")

        assert bleu1["bleu"] == all_metrics["bleu"]["bleu"]
        assert em1["exact_match"] == all_metrics["exact_match"]["exact_match"]
