#!/usr/bin/env python3
"""
TinyForgeAI Model Evaluation Example

This script demonstrates how to evaluate trained models using various metrics
and techniques.

Prerequisites:
    pip install -e ".[training]"

Usage:
    python examples/training/evaluation.py
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_test_data(output_path: Path):
    """Create test data for evaluation."""
    data = [
        {"input": "What is Python?", "output": "Python is a programming language."},
        {"input": "What is machine learning?", "output": "Machine learning is a type of AI."},
        {"input": "What is deep learning?", "output": "Deep learning uses neural networks."},
        {"input": "What is TinyForgeAI?", "output": "TinyForgeAI is a model training platform."},
        {"input": "What is LoRA?", "output": "LoRA is an efficient fine-tuning method."},
    ]

    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return data


def evaluate_mock_model():
    """Demonstrate evaluation with mock model."""
    print("\n1. Mock Model Evaluation")
    print("-" * 40)

    from backend.training.trainer import MockTrainer, TrainingConfig as MockConfig

    config = MockConfig(model_name="mock-eval-model")
    trainer = MockTrainer(config)

    # Create test data
    test_data = [
        {"input": "test input 1", "expected": "output 1"},
        {"input": "test input 2", "expected": "output 2"},
        {"input": "test input 3", "expected": "output 3"},
    ]

    print(f"  Test examples: {len(test_data)}")

    # Mock predictions
    predictions = []
    for item in test_data:
        # Mock model reverses the input
        pred = item["input"][::-1]
        predictions.append({
            "input": item["input"],
            "expected": item["expected"],
            "predicted": pred,
        })

    print("\n  Sample predictions:")
    for p in predictions[:2]:
        print(f"    Input: {p['input']}")
        print(f"    Expected: {p['expected']}")
        print(f"    Predicted: {p['predicted']}")
        print()


def calculate_metrics(predictions: list, references: list):
    """Calculate evaluation metrics."""
    print("\n2. Evaluation Metrics")
    print("-" * 40)

    # Exact match
    exact_matches = sum(1 for p, r in zip(predictions, references) if p == r)
    exact_match_rate = exact_matches / len(predictions) * 100

    print(f"  Exact Match Rate: {exact_match_rate:.1f}%")

    # Character-level accuracy (simple)
    char_correct = 0
    char_total = 0
    for pred, ref in zip(predictions, references):
        for i, (p_char, r_char) in enumerate(zip(pred, ref)):
            if p_char == r_char:
                char_correct += 1
            char_total += 1

    char_accuracy = char_correct / char_total * 100 if char_total > 0 else 0
    print(f"  Character Accuracy: {char_accuracy:.1f}%")

    # Length ratio
    avg_pred_len = sum(len(p) for p in predictions) / len(predictions)
    avg_ref_len = sum(len(r) for r in references) / len(references)
    length_ratio = avg_pred_len / avg_ref_len if avg_ref_len > 0 else 0

    print(f"  Avg Prediction Length: {avg_pred_len:.1f}")
    print(f"  Avg Reference Length: {avg_ref_len:.1f}")
    print(f"  Length Ratio: {length_ratio:.2f}")

    return {
        "exact_match": exact_match_rate,
        "char_accuracy": char_accuracy,
        "length_ratio": length_ratio,
    }


def demonstrate_evaluation_workflow():
    """Show complete evaluation workflow."""
    print("\n3. Complete Evaluation Workflow")
    print("-" * 40)

    print("""
    # Step 1: Load your trained model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained("./output/model")
    tokenizer = AutoTokenizer.from_pretrained("./output/model")

    # Step 2: Prepare test data
    test_data = load_test_data("test.jsonl")

    # Step 3: Generate predictions
    predictions = []
    for item in test_data:
        inputs = tokenizer(item["input"], return_tensors="pt")
        outputs = model(**inputs)
        pred = tokenizer.decode(outputs.logits.argmax(-1)[0])
        predictions.append(pred)

    # Step 4: Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    accuracy = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average="weighted")

    print(f"Accuracy: {accuracy:.2%}")
    print(f"F1 Score: {f1:.2f}")
    print(classification_report(references, predictions))
    """)


def demonstrate_cross_validation():
    """Show cross-validation approach."""
    print("\n4. Cross-Validation")
    print("-" * 40)

    print("""
    # K-Fold Cross-Validation for robust evaluation

    from sklearn.model_selection import KFold

    def cross_validate(data, k=5):
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
            print(f"Fold {fold + 1}/{k}")

            # Split data
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]

            # Train model
            trainer = RealTrainer(config)
            trainer.train(train_data)

            # Evaluate
            score = evaluate(trainer.model, val_data)
            scores.append(score)

            print(f"  Validation score: {score:.2f}")

        print(f"\\nMean score: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
        return scores
    """)


def demonstrate_error_analysis():
    """Show error analysis techniques."""
    print("\n5. Error Analysis")
    print("-" * 40)

    print("""
    # Analyze model errors to improve training

    def analyze_errors(predictions, references, inputs):
        errors = []

        for pred, ref, inp in zip(predictions, references, inputs):
            if pred != ref:
                errors.append({
                    "input": inp,
                    "expected": ref,
                    "predicted": pred,
                    "error_type": classify_error(pred, ref)
                })

        # Group by error type
        error_types = {}
        for e in errors:
            error_type = e["error_type"]
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(e)

        print("Error Distribution:")
        for error_type, examples in error_types.items():
            print(f"  {error_type}: {len(examples)} ({len(examples)/len(errors)*100:.1f}%)")

        return errors

    def classify_error(pred, ref):
        if len(pred) < len(ref) * 0.5:
            return "too_short"
        elif len(pred) > len(ref) * 1.5:
            return "too_long"
        elif pred.lower() == ref.lower():
            return "case_mismatch"
        else:
            return "content_error"
    """)


def main():
    """Run the evaluation example."""
    print("=" * 60)
    print("TinyForgeAI Model Evaluation Example")
    print("=" * 60)

    # Create test data
    output_dir = project_root / "examples" / "training"
    test_path = output_dir / "test_data.jsonl"
    test_data = create_test_data(test_path)

    print(f"\nCreated test data: {test_path}")

    # Mock model evaluation
    evaluate_mock_model()

    # Calculate metrics example
    predictions = ["Python is a language.", "ML is AI.", "DL uses networks."]
    references = ["Python is a programming language.", "Machine learning is a type of AI.", "Deep learning uses neural networks."]
    calculate_metrics(predictions, references)

    # Show workflows
    demonstrate_evaluation_workflow()
    demonstrate_cross_validation()
    demonstrate_error_analysis()

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Best Practices")
    print("=" * 60)

    print("""
    1. Always use a held-out test set
       - Never evaluate on training data
       - Split data: 80% train, 10% val, 10% test

    2. Choose appropriate metrics
       - Classification: Accuracy, F1, Precision, Recall
       - Generation: BLEU, ROUGE, Perplexity
       - Similarity: Cosine similarity, Edit distance

    3. Use cross-validation for small datasets
       - More reliable estimates
       - Identifies variance in model performance

    4. Perform error analysis
       - Understand failure modes
       - Guide data collection and model improvement

    5. Compare against baselines
       - Random baseline
       - Previous model versions
       - Simple heuristic approaches
    """)

    print("\nNext steps:")
    print("  1. Train a model: python examples/training/quick_start.py")
    print("  2. Run evaluation on your trained model")
    print("  3. Analyze errors and improve training data")


if __name__ == "__main__":
    main()
