#!/usr/bin/env python3
"""
TinyForgeAI Custom Dataset Example

This script demonstrates how to prepare and use different data formats
for training: JSONL, JSON, and CSV files.

Prerequisites:
    pip install -e ".[training]"

Usage:
    python examples/training/custom_dataset.py
"""

import os
import sys
import json
import csv
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_jsonl_data(output_path: Path):
    """Create sample data in JSONL format (recommended)."""
    data = [
        {"input": "Translate to French: Hello", "output": "Bonjour"},
        {"input": "Translate to French: Goodbye", "output": "Au revoir"},
        {"input": "Translate to French: Thank you", "output": "Merci"},
        {"input": "Translate to French: Please", "output": "S'il vous plaît"},
        {"input": "Translate to French: Yes", "output": "Oui"},
        {"input": "Translate to French: No", "output": "Non"},
        {"input": "Translate to French: Good morning", "output": "Bonjour"},
        {"input": "Translate to French: Good night", "output": "Bonne nuit"},
        {"input": "Translate to French: How are you?", "output": "Comment allez-vous?"},
        {"input": "Translate to French: I love you", "output": "Je t'aime"},
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Created JSONL file: {output_path}")
    print(f"  Number of examples: {len(data)}")
    return output_path


def create_json_data(output_path: Path):
    """Create sample data in JSON format (array of objects)."""
    data = [
        {"input": "Summarize: The quick brown fox jumps over the lazy dog. This is a common pangram used for typing practice.", "output": "A pangram about a fox and dog used for typing."},
        {"input": "Summarize: Machine learning is a subset of AI that enables computers to learn from data.", "output": "ML is AI that learns from data."},
        {"input": "Summarize: Python is a popular programming language known for its simplicity and readability.", "output": "Python is a simple, readable language."},
        {"input": "Summarize: TinyForgeAI helps developers train and deploy small language models efficiently.", "output": "TinyForgeAI trains and deploys small LLMs."},
        {"input": "Summarize: Docker containers package applications with their dependencies for consistent deployment.", "output": "Docker packages apps for consistent deployment."},
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Created JSON file: {output_path}")
    print(f"  Number of examples: {len(data)}")
    return output_path


def create_csv_data(output_path: Path):
    """Create sample data in CSV format."""
    data = [
        {"input": "Sentiment: I love this product!", "output": "positive"},
        {"input": "Sentiment: This is terrible.", "output": "negative"},
        {"input": "Sentiment: It's okay, nothing special.", "output": "neutral"},
        {"input": "Sentiment: Best purchase ever!", "output": "positive"},
        {"input": "Sentiment: Waste of money.", "output": "negative"},
        {"input": "Sentiment: Works as expected.", "output": "neutral"},
        {"input": "Sentiment: Amazing quality!", "output": "positive"},
        {"input": "Sentiment: Very disappointing.", "output": "negative"},
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output"])
        writer.writeheader()
        writer.writerows(data)

    print(f"  Created CSV file: {output_path}")
    print(f"  Number of examples: {len(data)}")
    return output_path


def create_conversation_data(output_path: Path):
    """Create conversation-style training data."""
    data = [
        {
            "input": "User: What's the weather like?\nAssistant:",
            "output": "I don't have access to real-time weather data. Please check a weather service for current conditions."
        },
        {
            "input": "User: Tell me a joke.\nAssistant:",
            "output": "Why did the programmer quit? Because they didn't get arrays! (a raise)"
        },
        {
            "input": "User: What is 2+2?\nAssistant:",
            "output": "2+2 equals 4."
        },
        {
            "input": "User: Who are you?\nAssistant:",
            "output": "I'm an AI assistant trained to help answer questions and have conversations."
        },
        {
            "input": "User: Can you help me code?\nAssistant:",
            "output": "I'd be happy to help with coding! What programming language and what kind of task?"
        },
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Created conversation data: {output_path}")
    print(f"  Number of examples: {len(data)}")
    return output_path


def load_and_preview_data(file_path: Path):
    """Load and preview any supported data format."""
    from backend.data_processing.dataset_loader import DatasetLoader

    print(f"\n  Loading: {file_path}")

    loader = DatasetLoader()

    try:
        # Determine format from extension
        ext = file_path.suffix.lower()

        if ext == ".jsonl":
            data = loader.load_jsonl(str(file_path))
        elif ext == ".json":
            data = loader.load_json(str(file_path))
        elif ext == ".csv":
            data = loader.load_csv(str(file_path))
        else:
            print(f"  Unsupported format: {ext}")
            return None

        print(f"  Loaded {len(data)} examples")
        print(f"  First example:")
        print(f"    Input: {data[0]['input'][:50]}...")
        print(f"    Output: {data[0]['output'][:50]}...")

        return data

    except Exception as e:
        print(f"  Error loading data: {e}")
        return None


def main():
    """Run the custom dataset example."""
    print("=" * 60)
    print("TinyForgeAI Custom Dataset Example")
    print("=" * 60)

    # Create output directory
    output_dir = project_root / "examples" / "training" / "custom_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create different data formats
    print("\n1. Creating Sample Datasets")
    print("-" * 40)

    jsonl_path = create_jsonl_data(output_dir / "translation.jsonl")
    json_path = create_json_data(output_dir / "summarization.json")
    csv_path = create_csv_data(output_dir / "sentiment.csv")
    conv_path = create_conversation_data(output_dir / "conversation.jsonl")

    # Load and preview each format
    print("\n2. Loading and Previewing Datasets")
    print("-" * 40)

    for path in [jsonl_path, json_path, csv_path, conv_path]:
        load_and_preview_data(path)

    # Show how to use with trainer
    print("\n3. Using Custom Data with Trainer")
    print("-" * 40)

    print("""
    from backend.training.real_trainer import RealTrainer, TrainingConfig

    # For translation task
    config = TrainingConfig(
        model_name="t5-small",  # Good for seq2seq tasks
        output_dir="./output/translation_model",
        num_epochs=5,
    )
    trainer = RealTrainer(config)
    trainer.train("custom_data/translation.jsonl")

    # For sentiment classification
    config = TrainingConfig(
        model_name="distilbert-base-uncased",
        output_dir="./output/sentiment_model",
        num_epochs=3,
    )
    trainer = RealTrainer(config)
    trainer.train("custom_data/sentiment.csv")
    """)

    # Best practices
    print("\n4. Data Preparation Best Practices")
    print("-" * 40)

    print("""
    1. JSONL is recommended for most use cases:
       - One example per line
       - Easy to append new data
       - Memory efficient for large datasets

    2. Data quality matters more than quantity:
       - Clean, consistent formatting
       - Diverse examples
       - Balanced classes (for classification)

    3. Recommended minimum examples:
       - Simple tasks: 50-100 examples
       - Complex tasks: 500-1000+ examples
       - Fine-tuning: 100+ high-quality examples

    4. Data format for different tasks:
       - Classification: {"input": "text", "output": "label"}
       - Generation: {"input": "prompt", "output": "completion"}
       - Translation: {"input": "source text", "output": "target text"}
       - Q&A: {"input": "question", "output": "answer"}
    """)

    print("\nFiles created in:", output_dir)
    print("\nNext steps:")
    print("  1. Customize the sample data for your use case")
    print("  2. Run training with: python examples/training/quick_start.py")
    print("  3. Try LoRA for efficient training: python examples/training/lora_training.py")


if __name__ == "__main__":
    main()
