#!/usr/bin/env python3
"""
TinyForgeAI Complete Demo

This script demonstrates all major features of TinyForgeAI in one run:
1. Data preparation
2. Model training (dry-run or real)
3. RAG document indexing
4. Model export
5. Inference serving

Usage:
    python demo.py              # Run with dry-run training
    python demo.py --real       # Run with real training (requires GPU/training deps)
    python demo.py --help       # Show all options
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_header(text: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step: int, text: str):
    """Print a formatted step."""
    print(f"\n[Step {step}] {text}")
    print("-" * 40)


def check_dependencies():
    """Check which dependencies are available."""
    deps = {
        "training": False,
        "rag": False,
        "inference": False,
    }

    try:
        import torch
        import transformers
        deps["training"] = True
    except ImportError:
        pass

    try:
        import sentence_transformers
        deps["rag"] = True
    except ImportError:
        pass

    try:
        import fastapi
        import uvicorn
        deps["inference"] = True
    except ImportError:
        pass

    return deps


def demo_data_preparation(output_dir: Path):
    """Demonstrate data preparation."""
    print_step(1, "Data Preparation")

    # Create sample training data
    training_data = [
        {"input": "What is TinyForgeAI?", "output": "TinyForgeAI is a lightweight platform for training and deploying small language models."},
        {"input": "How do I train a model?", "output": "Use the RealTrainer class with a TrainingConfig to train models on your data."},
        {"input": "What is LoRA?", "output": "LoRA (Low-Rank Adaptation) is an efficient fine-tuning technique that reduces memory usage by only training small adapter matrices."},
        {"input": "How do I deploy a model?", "output": "Export your model to ONNX format and deploy it using the inference server or Docker."},
        {"input": "What file formats are supported?", "output": "TinyForgeAI supports JSONL, JSON, CSV, PDF, DOCX, and plain text files."},
    ]

    data_path = output_dir / "demo_data.jsonl"
    with open(data_path, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"  Created training data: {data_path}")
    print(f"  Number of examples: {len(training_data)}")
    print(f"  Sample: {training_data[0]}")

    return data_path


def demo_training_dryrun(data_path: Path, output_dir: Path):
    """Demonstrate dry-run training."""
    print_step(2, "Training (Dry-Run Mode)")

    from backend.training.trainer import MockTrainer, TrainingConfig as MockConfig

    config = MockConfig(
        model_name="demo-model",
        epochs=3,
        batch_size=4,
        learning_rate=0.001,
    )

    trainer = MockTrainer(config)
    print(f"  Config: {config}")
    print(f"  Training on: {data_path}")

    # Simulate training
    result = trainer.train(str(data_path))

    # Save mock model
    model_path = output_dir / "model_stub.json"
    with open(model_path, "w") as f:
        json.dump({
            "model_name": config.model_name,
            "type": "mock",
            "training_data": str(data_path),
            "epochs": config.epochs,
        }, f, indent=2)

    print(f"  Training complete!")
    print(f"  Model saved to: {model_path}")

    return model_path


def demo_training_real(data_path: Path, output_dir: Path):
    """Demonstrate real training with HuggingFace."""
    print_step(2, "Training (Real Mode - HuggingFace)")

    from backend.training.real_trainer import RealTrainer, TrainingConfig

    config = TrainingConfig(
        model_name="distilbert-base-uncased",
        output_dir=str(output_dir / "trained_model"),
        num_epochs=1,  # Just 1 epoch for demo
        batch_size=2,
        learning_rate=2e-5,
        logging_steps=1,
        save_steps=100,
        max_length=128,
    )

    print(f"  Model: {config.model_name}")
    print(f"  Output: {config.output_dir}")
    print(f"  Epochs: {config.num_epochs}")

    trainer = RealTrainer(config)
    print(f"  Device: {trainer.device}")
    print(f"  Training...")

    try:
        trainer.train(str(data_path))
        print(f"  Training complete!")
        print(f"  Model saved to: {config.output_dir}")
        return Path(config.output_dir)
    except Exception as e:
        print(f"  Training failed: {e}")
        print(f"  Falling back to dry-run mode...")
        return demo_training_dryrun(data_path, output_dir)


def demo_rag_indexing(output_dir: Path):
    """Demonstrate RAG document indexing."""
    print_step(3, "RAG Document Indexing")

    from connectors import indexer as indexer_module

    if not indexer_module.EMBEDDINGS_AVAILABLE:
        print("  RAG dependencies not installed.")
        print("  Install with: pip install -e '.[rag]'")
        print("  Skipping RAG demo...")
        return None

    from connectors.indexer import DocumentIndexer, IndexerConfig

    # Create sample documents
    docs_dir = output_dir / "documents"
    docs_dir.mkdir(exist_ok=True)

    documents = {
        "readme.txt": "TinyForgeAI is a platform for training small language models. It supports multiple data formats and efficient fine-tuning with LoRA.",
        "training.txt": "To train a model, prepare your data in JSONL format with input and output fields. Then use RealTrainer with your configuration.",
        "deployment.txt": "Deploy models using Docker or the inference server. Export to ONNX for optimized performance in production.",
    }

    for name, content in documents.items():
        (docs_dir / name).write_text(content)

    print(f"  Created {len(documents)} sample documents")

    # Index documents
    config = IndexerConfig(
        chunk_size=100,
        chunk_overlap=10,
        index_path=str(output_dir / "rag_index"),
    )

    indexer = DocumentIndexer(config)

    for doc_file in docs_dir.glob("*.txt"):
        doc_ids = indexer.index_file(doc_file)
        print(f"  Indexed {doc_file.name}: {len(doc_ids)} chunks")

    indexer.save_index()
    print(f"  Index saved to: {config.index_path}")

    # Demo search
    print("\n  Demo Search:")
    query = "How do I deploy a model?"
    results = indexer.search(query, top_k=2)

    print(f"  Query: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"  [{i}] Score: {result.score:.3f} - {result.document.content[:50]}...")

    return indexer


def demo_export(model_path: Path, output_dir: Path):
    """Demonstrate model export."""
    print_step(4, "Model Export")

    from backend.exporter.service_exporter import export_service

    service_dir = output_dir / "service"

    print(f"  Exporting model to service...")
    print(f"  Model: {model_path}")
    print(f"  Output: {service_dir}")

    try:
        export_service(
            model_path=str(model_path),
            output_dir=str(service_dir),
            export_onnx=False,  # Skip ONNX for demo
            overwrite=True,
        )
        print(f"  Service exported to: {service_dir}")
        return service_dir
    except Exception as e:
        print(f"  Export failed: {e}")
        print(f"  This is expected if using real training output.")
        return None


def demo_inference(service_dir: Path):
    """Demonstrate inference."""
    print_step(5, "Inference Demo")

    if service_dir is None or not service_dir.exists():
        print("  No service directory available.")
        print("  Skipping inference demo...")
        return

    app_path = service_dir / "app.py"
    if not app_path.exists():
        print(f"  Service app not found: {app_path}")
        return

    print(f"  Loading service from: {app_path}")

    try:
        import importlib.util
        from fastapi.testclient import TestClient

        spec = importlib.util.spec_from_file_location("service_app", app_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        client = TestClient(mod.app)

        # Test health endpoint
        response = client.get("/health")
        print(f"  Health check: {response.json()}")

        # Test prediction
        response = client.post("/predict", json={"input": "Hello TinyForgeAI!"})
        print(f"  Prediction: {response.json()}")

    except Exception as e:
        print(f"  Inference test failed: {e}")


def demo_dashboard_api():
    """Demonstrate dashboard API."""
    print_step(6, "Dashboard API")

    try:
        from services.dashboard_api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Test endpoints
        response = client.get("/health")
        print(f"  Health: {response.json()}")

        response = client.get("/api/stats")
        print(f"  Stats: {response.json()}")

        response = client.get("/api/jobs")
        print(f"  Jobs: {response.json()}")

    except Exception as e:
        print(f"  Dashboard API test failed: {e}")


def main():
    """Run the complete demo."""
    parser = argparse.ArgumentParser(
        description="TinyForgeAI Complete Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo.py              # Dry-run mode (no GPU needed)
    python demo.py --real       # Real training (requires training deps)
    python demo.py --output ./my_demo  # Custom output directory
        """
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real training instead of dry-run"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: temp directory)"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep output files after demo"
    )

    args = parser.parse_args()

    print_header("TinyForgeAI Complete Demo")

    # Check dependencies
    print("\nChecking dependencies...")
    deps = check_dependencies()
    print(f"  Training available: {deps['training']}")
    print(f"  RAG available: {deps['rag']}")
    print(f"  Inference available: {deps['inference']}")

    if args.real and not deps["training"]:
        print("\nError: Real training requested but dependencies not installed.")
        print("Install with: pip install -e '.[training]'")
        print("Falling back to dry-run mode...")
        args.real = False

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="tinyforge_demo_"))

    print(f"\nOutput directory: {output_dir}")

    try:
        # Step 1: Data preparation
        data_path = demo_data_preparation(output_dir)

        # Step 2: Training
        if args.real:
            model_path = demo_training_real(data_path, output_dir)
        else:
            model_path = demo_training_dryrun(data_path, output_dir)

        # Step 3: RAG indexing
        if deps["rag"]:
            demo_rag_indexing(output_dir)
        else:
            print_step(3, "RAG Document Indexing")
            print("  Skipped (dependencies not installed)")

        # Step 4: Export
        service_dir = demo_export(model_path, output_dir)

        # Step 5: Inference
        if deps["inference"]:
            demo_inference(service_dir)
        else:
            print_step(5, "Inference Demo")
            print("  Skipped (dependencies not installed)")

        # Step 6: Dashboard API
        if deps["inference"]:
            demo_dashboard_api()
        else:
            print_step(6, "Dashboard API")
            print("  Skipped (dependencies not installed)")

        # Summary
        print_header("Demo Complete!")
        print(f"\nOutput files are in: {output_dir}")
        print("\nWhat was demonstrated:")
        print("  1. Data preparation (JSONL format)")
        print(f"  2. Model training ({'real' if args.real else 'dry-run'} mode)")
        print(f"  3. RAG indexing ({'completed' if deps['rag'] else 'skipped'})")
        print("  4. Service export")
        print(f"  5. Inference testing ({'completed' if deps['inference'] else 'skipped'})")
        print(f"  6. Dashboard API ({'completed' if deps['inference'] else 'skipped'})")

        print("\nNext steps:")
        print("  - Read the tutorials: docs/tutorials/")
        print("  - Try real training: python demo.py --real")
        print("  - Run the examples: python examples/training/quick_start.py")

        if not args.keep and not args.output:
            print(f"\nNote: Output files will be cleaned up.")
            print(f"      Use --keep or --output to preserve them.")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
