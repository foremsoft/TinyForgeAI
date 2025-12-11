# Module 8: Test & Improve Your Model

**Time needed:** 20 minutes
**Prerequisites:** Module 7 (trained model)
**Goal:** Evaluate your model and make it better

---

## The Testing Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                  Model Improvement Cycle                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Train → Test → Analyze Errors → Improve → Repeat         │
│                                                             │
│   1. Test with new questions                                │
│   2. Find where it fails                                    │
│   3. Add more training data for failures                    │
│   4. Retrain and test again                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Load Your Trained Model

```python
# load_model.py - Load and use your trained model

"""
Load your trained model and make predictions.
"""

import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TrainedFAQBot:
    """Use your trained model for predictions."""

    def __init__(self, model_path: str):
        """
        Load a trained model.

        Args:
            model_path: Path to saved model directory
        """
        print(f"Loading model from {model_path}...")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Load label mapping
        with open(f"{model_path}/label_mapping.json", 'r') as f:
            mapping = json.load(f)
            self.id2label = {int(k): v for k, v in mapping['id2label'].items()}
            self.label2id = mapping['label2id']

        # Set to evaluation mode
        self.model.eval()

        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        print(f"  ✓ Model loaded ({len(self.id2label)} classes)")
        print(f"  ✓ Running on: {self.device}")

    def predict(self, question: str, return_all: bool = False) -> dict:
        """
        Get prediction for a question.

        Args:
            question: The user's question
            return_all: If True, return all predictions with scores

        Returns:
            Dictionary with answer and confidence
        """
        # Tokenize
        inputs = self.tokenizer(
            question,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # Get top prediction
        top_prob, top_idx = torch.max(probs, dim=-1)
        top_label = self.id2label[top_idx.item()]

        result = {
            "answer": top_label,
            "confidence": top_prob.item()
        }

        # Optionally return all predictions
        if return_all:
            all_preds = []
            for idx, prob in enumerate(probs[0]):
                all_preds.append({
                    "answer": self.id2label[idx],
                    "confidence": prob.item()
                })
            # Sort by confidence
            all_preds.sort(key=lambda x: x['confidence'], reverse=True)
            result["all_predictions"] = all_preds[:5]  # Top 5

        return result

    def chat(self):
        """Interactive chat mode."""
        print("\n" + "=" * 60)
        print("  Trained Model - Chat Mode")
        print("  Type 'quit' to exit, 'debug' for detailed output")
        print("=" * 60 + "\n")

        debug_mode = False

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            if user_input.lower() == 'debug':
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                continue

            if not user_input:
                continue

            # Get prediction
            result = self.predict(user_input, return_all=debug_mode)

            # Show result
            confidence = result['confidence'] * 100
            print(f"Bot: {result['answer']}")
            print(f"     (Confidence: {confidence:.1f}%)")

            if debug_mode and 'all_predictions' in result:
                print("     Top alternatives:")
                for pred in result['all_predictions'][1:4]:  # Skip first (already shown)
                    print(f"       - {pred['confidence']*100:.1f}%: {pred['answer'][:50]}...")

            print()


# Usage
if __name__ == "__main__":
    bot = TrainedFAQBot("./my_faq_model")
    bot.chat()
```

---

## Step 2: Evaluate on Test Data

```python
# evaluate_model.py - Comprehensive model evaluation

"""
Evaluate your model's performance on a test set.
"""

import json
from collections import defaultdict


def evaluate_model(bot, test_file: str) -> dict:
    """
    Evaluate model on test data.

    Args:
        bot: TrainedFAQBot instance
        test_file: Path to test JSONL file

    Returns:
        Evaluation metrics
    """
    print(f"\nEvaluating on {test_file}...")
    print("-" * 50)

    # Load test data
    test_examples = []
    with open(test_file, 'r') as f:
        for line in f:
            if line.strip():
                test_examples.append(json.loads(line))

    print(f"Test examples: {len(test_examples)}")

    # Track results
    correct = 0
    total = 0
    errors = []
    confidence_correct = []
    confidence_wrong = []

    for example in test_examples:
        question = example['input']
        expected = example['output']

        # Get prediction
        result = bot.predict(question)
        predicted = result['answer']
        confidence = result['confidence']

        total += 1

        if predicted == expected:
            correct += 1
            confidence_correct.append(confidence)
        else:
            confidence_wrong.append(confidence)
            errors.append({
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence
            })

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    avg_conf_correct = sum(confidence_correct) / len(confidence_correct) if confidence_correct else 0
    avg_conf_wrong = sum(confidence_wrong) / len(confidence_wrong) if confidence_wrong else 0

    results = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_confidence_correct": avg_conf_correct,
        "avg_confidence_wrong": avg_conf_wrong,
        "errors": errors
    }

    # Print results
    print(f"\n📊 Results:")
    print(f"   Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
    print(f"   Avg confidence (correct): {avg_conf_correct*100:.1f}%")
    print(f"   Avg confidence (wrong):   {avg_conf_wrong*100:.1f}%")

    if errors:
        print(f"\n❌ Errors ({len(errors)}):")
        for err in errors[:5]:  # Show first 5 errors
            print(f"   Q: {err['question']}")
            print(f"   Expected: {err['expected'][:50]}...")
            print(f"   Got: {err['predicted'][:50]}... ({err['confidence']*100:.0f}%)")
            print()

    return results


def detailed_analysis(errors: list) -> dict:
    """Analyze error patterns."""
    print("\n🔍 Error Analysis")
    print("-" * 50)

    # Group errors by expected answer
    by_expected = defaultdict(list)
    for err in errors:
        by_expected[err['expected'][:30]].append(err)

    print("\nMost confused answers:")
    sorted_groups = sorted(by_expected.items(), key=lambda x: len(x[1]), reverse=True)
    for expected, errs in sorted_groups[:5]:
        print(f"  '{expected}...' - {len(errs)} errors")

    # Analyze confidence distribution
    low_conf_errors = [e for e in errors if e['confidence'] < 0.5]
    high_conf_errors = [e for e in errors if e['confidence'] >= 0.5]

    print(f"\nConfidence analysis:")
    print(f"  Low confidence errors (<50%): {len(low_conf_errors)}")
    print(f"  High confidence errors (≥50%): {len(high_conf_errors)}")

    if high_conf_errors:
        print("\n  ⚠️  High-confidence errors need attention!")
        print("     These are cases where the model is confident but wrong.")

    return {
        "by_expected": dict(by_expected),
        "low_conf_errors": len(low_conf_errors),
        "high_conf_errors": len(high_conf_errors)
    }


# Usage
if __name__ == "__main__":
    from load_model import TrainedFAQBot

    bot = TrainedFAQBot("./my_faq_model")
    results = evaluate_model(bot, "test_data.jsonl")

    if results['errors']:
        detailed_analysis(results['errors'])
```

---

## Step 3: Interactive Testing

```python
# interactive_test.py - Test and record results interactively

"""
Interactive testing tool.
Test your model and save cases it gets wrong for improvement.
"""

import json
from datetime import datetime


class InteractiveTester:
    """Test model interactively and record failures."""

    def __init__(self, bot, failures_file: str = "failures.jsonl"):
        self.bot = bot
        self.failures_file = failures_file
        self.session_stats = {
            "tested": 0,
            "correct": 0,
            "wrong": 0,
            "unsure": 0
        }

    def test_single(self, question: str) -> dict:
        """Test a single question."""
        result = self.bot.predict(question, return_all=True)
        return result

    def record_failure(self, question: str, wrong_answer: str, correct_answer: str):
        """Record a failure for later training."""
        failure = {
            "input": question,
            "output": correct_answer,
            "wrong_prediction": wrong_answer,
            "timestamp": datetime.now().isoformat()
        }

        with open(self.failures_file, 'a') as f:
            f.write(json.dumps(failure) + '\n')

        print(f"  ✓ Recorded for retraining")

    def run_session(self):
        """Run interactive testing session."""
        print("\n" + "=" * 60)
        print("  Interactive Model Testing")
        print("=" * 60)
        print("\nCommands:")
        print("  Type a question to test")
        print("  After each answer, rate: 'y' (correct), 'n' (wrong), 's' (skip)")
        print("  Type 'stats' to see session statistics")
        print("  Type 'quit' to exit")
        print()

        while True:
            question = input("\nQuestion: ").strip()

            if question.lower() == 'quit':
                break

            if question.lower() == 'stats':
                self._print_stats()
                continue

            if not question:
                continue

            # Get prediction
            result = self.test_single(question)
            answer = result['answer']
            confidence = result['confidence']

            print(f"\n🤖 Answer: {answer}")
            print(f"   Confidence: {confidence*100:.1f}%")

            # Show alternatives if low confidence
            if confidence < 0.7 and 'all_predictions' in result:
                print("   Alternatives:")
                for alt in result['all_predictions'][1:3]:
                    print(f"     - {alt['confidence']*100:.0f}%: {alt['answer'][:40]}...")

            # Get feedback
            feedback = input("\nCorrect? (y/n/s): ").strip().lower()

            self.session_stats['tested'] += 1

            if feedback == 'y':
                self.session_stats['correct'] += 1
                print("  ✓ Great!")

            elif feedback == 'n':
                self.session_stats['wrong'] += 1
                correct = input("What should the answer be? ").strip()
                if correct:
                    self.record_failure(question, answer, correct)
                else:
                    print("  (Skipped recording)")

            else:
                self.session_stats['unsure'] += 1
                print("  (Skipped)")

        self._print_stats()
        print("\nGoodbye!")

    def _print_stats(self):
        """Print session statistics."""
        total = self.session_stats['tested']
        correct = self.session_stats['correct']
        wrong = self.session_stats['wrong']

        print("\n📊 Session Statistics:")
        print(f"   Tested: {total}")
        print(f"   Correct: {correct}")
        print(f"   Wrong: {wrong}")

        if total > 0:
            accuracy = correct / total * 100
            print(f"   Accuracy: {accuracy:.1f}%")

        # Count total failures recorded
        try:
            with open(self.failures_file, 'r') as f:
                failures = sum(1 for _ in f)
            print(f"   Total failures recorded: {failures}")
        except FileNotFoundError:
            pass


# Usage
if __name__ == "__main__":
    from load_model import TrainedFAQBot

    bot = TrainedFAQBot("./my_faq_model")
    tester = InteractiveTester(bot)
    tester.run_session()
```

---

## Step 4: Improve with More Data

```python
# improve_model.py - Add failures to training data and retrain

"""
Improve your model by adding failure cases to training data.
"""

import json
import shutil
from pathlib import Path


def merge_failures_to_training(
    original_train_file: str,
    failures_file: str,
    output_file: str
) -> int:
    """
    Merge failure cases into training data.

    Args:
        original_train_file: Original training data
        failures_file: Recorded failures
        output_file: Output combined file

    Returns:
        Number of new examples added
    """
    # Load original training data
    original = []
    with open(original_train_file, 'r') as f:
        for line in f:
            if line.strip():
                original.append(json.loads(line))

    print(f"Original training examples: {len(original)}")

    # Load failures
    failures = []
    try:
        with open(failures_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Convert to training format
                    failures.append({
                        "input": data['input'],
                        "output": data['output']
                    })
    except FileNotFoundError:
        print(f"No failures file found: {failures_file}")
        return 0

    print(f"Failures to add: {len(failures)}")

    # Check for duplicates
    existing_inputs = {ex['input'].lower() for ex in original}
    new_examples = []

    for fail in failures:
        if fail['input'].lower() not in existing_inputs:
            new_examples.append(fail)
            existing_inputs.add(fail['input'].lower())

    print(f"New unique examples: {len(new_examples)}")

    # Combine and save
    combined = original + new_examples

    with open(output_file, 'w') as f:
        for ex in combined:
            f.write(json.dumps(ex) + '\n')

    print(f"✅ Saved {len(combined)} examples to {output_file}")

    return len(new_examples)


def create_augmented_data(training_file: str, output_file: str) -> int:
    """
    Create augmented training data with variations.

    Augmentation strategies:
    - Lowercase versions
    - Remove punctuation variations
    - Add common typos (optional)
    """
    print(f"\nAugmenting {training_file}...")

    # Load original
    original = []
    with open(training_file, 'r') as f:
        for line in f:
            if line.strip():
                original.append(json.loads(line))

    augmented = list(original)  # Start with original

    for ex in original:
        inp = ex['input']
        out = ex['output']

        # Variation 1: Lowercase
        if inp.lower() != inp:
            augmented.append({"input": inp.lower(), "output": out})

        # Variation 2: Without question mark
        if inp.endswith('?'):
            augmented.append({"input": inp[:-1], "output": out})

        # Variation 3: With "please" added
        if not inp.lower().startswith('please'):
            augmented.append({"input": f"Please {inp.lower()}", "output": out})

    # Remove duplicates
    seen = set()
    unique = []
    for ex in augmented:
        key = ex['input'].lower()
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    # Save
    with open(output_file, 'w') as f:
        for ex in unique:
            f.write(json.dumps(ex) + '\n')

    added = len(unique) - len(original)
    print(f"  Original: {len(original)}")
    print(f"  Augmented: {len(unique)}")
    print(f"  Added: {added} variations")
    print(f"✅ Saved to {output_file}")

    return added


def improvement_workflow():
    """Complete improvement workflow."""
    print("=" * 60)
    print("  Model Improvement Workflow")
    print("=" * 60)

    # Step 1: Merge failures
    print("\n[Step 1] Merging failure cases...")
    merge_failures_to_training(
        "train_data.jsonl",
        "failures.jsonl",
        "train_data_improved.jsonl"
    )

    # Step 2: Augment data
    print("\n[Step 2] Creating augmented variations...")
    create_augmented_data(
        "train_data_improved.jsonl",
        "train_data_augmented.jsonl"
    )

    # Step 3: Instructions for retraining
    print("\n[Step 3] Ready to retrain!")
    print("-" * 50)
    print("Run the training script with the new data:")
    print()
    print("  python train_detailed.py --train_file train_data_augmented.jsonl")
    print()
    print("Or modify train_detailed.py to use:")
    print("  TRAIN_FILE = 'train_data_augmented.jsonl'")
    print()
    print("After training, test again to see improvement!")


# Usage
if __name__ == "__main__":
    improvement_workflow()
```

---

## Step 5: Compare Versions

```python
# compare_models.py - Compare different model versions

"""
Compare performance of different model versions.
"""

import json


def compare_models(models: dict, test_file: str):
    """
    Compare multiple models on the same test set.

    Args:
        models: Dict of {name: model_path}
        test_file: Test data file
    """
    from load_model import TrainedFAQBot

    # Load test data
    test_examples = []
    with open(test_file, 'r') as f:
        for line in f:
            if line.strip():
                test_examples.append(json.loads(line))

    print(f"\nComparing {len(models)} models on {len(test_examples)} test examples")
    print("=" * 70)

    results = {}

    for name, path in models.items():
        print(f"\n📊 {name}")
        print("-" * 40)

        try:
            bot = TrainedFAQBot(path)

            correct = 0
            total_conf = 0

            for ex in test_examples:
                result = bot.predict(ex['input'])
                if result['answer'] == ex['output']:
                    correct += 1
                total_conf += result['confidence']

            accuracy = correct / len(test_examples) * 100
            avg_conf = total_conf / len(test_examples) * 100

            results[name] = {
                "accuracy": accuracy,
                "avg_confidence": avg_conf
            }

            print(f"   Accuracy: {accuracy:.1f}%")
            print(f"   Avg Confidence: {avg_conf:.1f}%")

        except Exception as e:
            print(f"   Error: {e}")
            results[name] = {"error": str(e)}

    # Summary comparison
    print("\n" + "=" * 70)
    print("Summary Comparison")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':>12} {'Confidence':>12}")
    print("-" * 44)

    for name, res in results.items():
        if 'error' in res:
            print(f"{name:<20} {'Error':>12}")
        else:
            print(f"{name:<20} {res['accuracy']:>11.1f}% {res['avg_confidence']:>11.1f}%")

    # Recommendation
    print("\n💡 Recommendation:")
    valid_results = {k: v for k, v in results.items() if 'accuracy' in v}
    if valid_results:
        best = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"   Best model: {best[0]} ({best[1]['accuracy']:.1f}% accuracy)")


# Usage
if __name__ == "__main__":
    models_to_compare = {
        "v1_original": "./my_faq_model",
        "v2_improved": "./my_faq_model_v2",
        # "v3_augmented": "./my_faq_model_v3",
    }

    compare_models(models_to_compare, "test_data.jsonl")
```

---

## Improvement Strategies Summary

```
┌─────────────────────────────────────────────────────────────┐
│              Model Improvement Strategies                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📈 Improve Accuracy                                        │
│  ├── Add more training examples                             │
│  ├── Include failure cases                                  │
│  ├── Balance classes (equal examples per answer)            │
│  └── Train for more epochs                                  │
│                                                             │
│  🎯 Improve Confidence                                       │
│  ├── Add variations of the same question                    │
│  ├── Use data augmentation                                  │
│  └── Fine-tune on similar domains first                     │
│                                                             │
│  🔧 Fix Specific Errors                                      │
│  ├── Add exact examples the model got wrong                 │
│  ├── Add synonyms and rephrasing                            │
│  └── Remove conflicting examples                            │
│                                                             │
│  ⚠️  Avoid Overfitting                                       │
│  ├── Keep a test set separate                               │
│  ├── Don't train too many epochs                            │
│  └── Monitor test accuracy, not just training loss          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Checkpoint Quiz

**1. Why do we keep a separate test set?**
<details>
<summary>Click for answer</summary>

To evaluate how well the model performs on data it hasn't seen during training. This tells us if the model generalizes well or just memorized the training data.

</details>

**2. What is a "high-confidence error"?**
<details>
<summary>Click for answer</summary>

When the model is very confident (e.g., 90%) but wrong. These are dangerous because the model seems sure. They often indicate conflicting training data or missing examples.

</details>

**3. What is data augmentation?**
<details>
<summary>Click for answer</summary>

Creating variations of training examples (lowercase, removing punctuation, adding "please", etc.) to teach the model that these variations mean the same thing.

</details>

---

## What's Next?

In **Module 9: Deploy & Share**, you'll:
- Package your model for production
- Create a web API
- Deploy to the cloud
- Share with the world!

**Your model is improving! Let's put it online.**

---

[← Back to Module 7](07-train-your-model.md) | [Continue to Module 9 →](09-deploy-and-share.md)
