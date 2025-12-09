"""
TinyForgeAI Evaluation CLI

Command-line interface for model evaluation and benchmarking.

Usage:
    # Evaluate a model
    python -m backend.evaluation.cli evaluate --model ./my_model --data test.jsonl

    # Run benchmark
    python -m backend.evaluation.cli benchmark --models ./model1 ./model2 --datasets test1.jsonl test2.jsonl

    # Compare predictions
    python -m backend.evaluation.cli compare --predictions pred.jsonl --references ref.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from backend.evaluation.evaluator import (
    ModelEvaluator,
    EvaluationConfig,
    EvaluationResults,
    EVALUATION_AVAILABLE,
)
from backend.evaluation.benchmark import (
    BenchmarkRunner,
    BenchmarkConfig,
    create_benchmark_dataset,
)
from backend.evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    compute_exact_match,
    compute_accuracy,
    compute_f1,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run model evaluation."""
    logger.info(f"Evaluating model: {args.model}")
    logger.info(f"Test data: {args.data}")

    # Build config
    config = EvaluationConfig(
        task_type=args.task_type,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_new_tokens=args.max_tokens,
        metrics=args.metrics.split(",") if args.metrics else ["bleu", "rouge", "exact_match"],
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
    )

    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        config=config,
    )

    # Run evaluation
    try:
        results = evaluator.evaluate(args.data)
        print(results.summary())

        if args.output_dir:
            results.save(Path(args.output_dir) / "evaluation_results.json")
            logger.info(f"Results saved to {args.output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmark comparison."""
    logger.info(f"Running benchmark: {args.name}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Datasets: {args.datasets}")

    # Build config
    config = BenchmarkConfig(
        benchmark_name=args.name,
        task_type=args.task_type,
        datasets=args.datasets,
        batch_size=args.batch_size,
        max_samples_per_dataset=args.max_samples,
        output_dir=args.output_dir,
        baseline_model=args.baseline,
    )

    # Create runner
    runner = BenchmarkRunner(config)

    # Run benchmark
    try:
        results = runner.run(models=args.models)

        # Print comparison table
        print("\n" + results.comparison_table())

        if results.best_model:
            print(f"\nBest model: {results.best_model}")

        logger.info(f"Results saved to {args.output_dir}")
        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare predictions against references."""
    logger.info(f"Comparing predictions: {args.predictions}")
    logger.info(f"References: {args.references}")

    # Load predictions and references
    predictions = []
    references = []

    pred_path = Path(args.predictions)
    ref_path = Path(args.references)

    # Load predictions
    if pred_path.suffix == ".jsonl":
        with open(pred_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    predictions.append(data.get(args.pred_field, data.get("prediction", "")))
    elif pred_path.suffix == ".json":
        with open(pred_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                predictions = [d.get(args.pred_field, d.get("prediction", "")) for d in data]
            else:
                predictions = [data.get(args.pred_field, data.get("prediction", ""))]
    else:
        with open(pred_path) as f:
            predictions = [line.strip() for line in f if line.strip()]

    # Load references
    if ref_path.suffix == ".jsonl":
        with open(ref_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    references.append(data.get(args.ref_field, data.get("output", "")))
    elif ref_path.suffix == ".json":
        with open(ref_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                references = [d.get(args.ref_field, d.get("output", "")) for d in data]
            else:
                references = [data.get(args.ref_field, data.get("output", ""))]
    else:
        with open(ref_path) as f:
            references = [line.strip() for line in f if line.strip()]

    if len(predictions) != len(references):
        logger.error(f"Mismatch: {len(predictions)} predictions vs {len(references)} references")
        return 1

    logger.info(f"Comparing {len(predictions)} samples")

    # Compute metrics
    metrics = {}
    metric_list = args.metrics.split(",") if args.metrics else ["bleu", "rouge", "exact_match"]

    if "bleu" in metric_list:
        metrics["bleu"] = compute_bleu(predictions, references)

    if "rouge" in metric_list:
        metrics["rouge"] = compute_rouge(predictions, references)

    if "exact_match" in metric_list:
        metrics["exact_match"] = compute_exact_match(predictions, references)

    if "accuracy" in metric_list:
        metrics["accuracy"] = compute_accuracy(predictions, references)

    if "f1" in metric_list:
        metrics["f1"] = compute_f1(predictions, references)

    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Samples: {len(predictions)}")
    print("-" * 60)

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, dict):
            print(f"{metric_name}:")
            for k, v in metric_value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, float):
                            print(f"  {k}.{kk}: {vv:.4f}")
        elif isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")

    print("=" * 60)

    # Save results if output specified
    if args.output:
        output_data = {
            "predictions_file": str(pred_path),
            "references_file": str(ref_path),
            "num_samples": len(predictions),
            "metrics": metrics,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    return 0


def cmd_create_dataset(args: argparse.Namespace) -> int:
    """Create a benchmark dataset."""
    logger.info(f"Creating dataset: {args.name}")

    samples = []

    if args.input:
        # Load from input file
        with open(args.input) as f:
            if args.input.endswith(".jsonl"):
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            else:
                samples = json.load(f)
    else:
        # Interactive creation
        print("Enter samples (empty input to finish):")
        while True:
            inp = input("Input: ").strip()
            if not inp:
                break
            out = input("Output: ").strip()
            samples.append({"input": inp, "output": out})

    if not samples:
        logger.error("No samples provided")
        return 1

    # Create dataset
    path = create_benchmark_dataset(
        name=args.name,
        samples=samples,
        output_dir=args.output_dir,
    )

    print(f"Created dataset: {path} ({len(samples)} samples)")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tinyforge-eval",
        description="TinyForgeAI Model Evaluation and Benchmarking",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--model", "-m", required=True, help="Model path")
    eval_parser.add_argument("--data", "-d", required=True, help="Test data file (JSONL)")
    eval_parser.add_argument("--task-type", "-t", default="generation",
                             choices=["generation", "classification", "qa"],
                             help="Task type")
    eval_parser.add_argument("--metrics", default="bleu,rouge,exact_match",
                             help="Comma-separated metrics to compute")
    eval_parser.add_argument("--batch-size", "-b", type=int, default=8,
                             help="Batch size")
    eval_parser.add_argument("--max-samples", type=int, default=None,
                             help="Maximum samples to evaluate")
    eval_parser.add_argument("--max-tokens", type=int, default=128,
                             help="Maximum tokens to generate")
    eval_parser.add_argument("--output-dir", "-o", default="./eval_results",
                             help="Output directory")
    eval_parser.add_argument("--save-predictions", action="store_true",
                             help="Save predictions to output")
    eval_parser.set_defaults(func=cmd_evaluate)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark comparison")
    bench_parser.add_argument("--name", "-n", default="benchmark",
                              help="Benchmark name")
    bench_parser.add_argument("--models", "-m", nargs="+", required=True,
                              help="Model paths to compare")
    bench_parser.add_argument("--datasets", "-d", nargs="+", required=True,
                              help="Dataset files to evaluate on")
    bench_parser.add_argument("--task-type", "-t", default="generation",
                              choices=["generation", "classification", "qa"],
                              help="Task type")
    bench_parser.add_argument("--batch-size", "-b", type=int, default=8,
                              help="Batch size")
    bench_parser.add_argument("--max-samples", type=int, default=None,
                              help="Maximum samples per dataset")
    bench_parser.add_argument("--output-dir", "-o", default="./benchmark_results",
                              help="Output directory")
    bench_parser.add_argument("--baseline", help="Baseline model for comparison")
    bench_parser.set_defaults(func=cmd_benchmark)

    # Compare command
    comp_parser = subparsers.add_parser("compare", help="Compare predictions to references")
    comp_parser.add_argument("--predictions", "-p", required=True,
                             help="Predictions file")
    comp_parser.add_argument("--references", "-r", required=True,
                             help="References file")
    comp_parser.add_argument("--pred-field", default="prediction",
                             help="Field name for predictions in JSON")
    comp_parser.add_argument("--ref-field", default="output",
                             help="Field name for references in JSON")
    comp_parser.add_argument("--metrics", default="bleu,rouge,exact_match",
                             help="Comma-separated metrics to compute")
    comp_parser.add_argument("--output", "-o", help="Output file for results")
    comp_parser.set_defaults(func=cmd_compare)

    # Create dataset command
    create_parser = subparsers.add_parser("create-dataset", help="Create a benchmark dataset")
    create_parser.add_argument("--name", "-n", required=True, help="Dataset name")
    create_parser.add_argument("--input", "-i", help="Input file (optional)")
    create_parser.add_argument("--output-dir", "-o", default="./benchmark_data",
                               help="Output directory")
    create_parser.set_defaults(func=cmd_create_dataset)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
