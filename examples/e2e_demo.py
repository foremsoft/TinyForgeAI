#!/usr/bin/env python
"""
TinyForgeAI End-to-End Demo Script (Python version).

This script demonstrates the complete dry-run workflow:
dataset -> train (dry-run) -> export microservice -> test inference

Works on all platforms (Windows, Linux, macOS) without shell dependencies.

Usage:
    python examples/e2e_demo.py           # Run demo, keep artifacts
    python examples/e2e_demo.py --cleanup # Run demo, then delete artifacts
"""

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Get project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("=" * 50)
    print(text)
    print("=" * 50)
    print()


def print_step(step_num: int, text: str) -> None:
    """Print a step description."""
    print(f"[Step {step_num}] {text}")


def load_module_from_path(module_name: str, file_path: Path):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_demo(cleanup: bool = False) -> dict:
    """
    Run the end-to-end demo.

    Args:
        cleanup: If True, delete workspace after demo.

    Returns:
        Dict with demo results including paths and response.
    """
    print_header("TinyForgeAI End-to-End Demo (Dry-Run Mode)")

    print(f"Project root: {PROJECT_ROOT}")
    print()

    # Create temporary workspace
    workdir = Path(tempfile.mkdtemp(prefix="tinyforge_demo_"))
    print(f"Created workspace: {workdir}")
    print()

    results = {
        "workdir": str(workdir),
        "model_path": None,
        "service_path": None,
        "response": None,
    }

    try:
        # Step 1: Copy sample data
        print_step(1, "Copying sample training data...")
        sample_data = PROJECT_ROOT / "examples" / "sample_qna.jsonl"
        data_path = workdir / "data.jsonl"
        shutil.copy(sample_data, data_path)
        print(f"  -> Copied to: {data_path}")
        print()

        # Step 2: Run dry-run trainer
        print_step(2, "Running dry-run trainer...")
        model_dir = workdir / "tiny_model"

        # Import and run trainer programmatically
        sys.path.insert(0, str(PROJECT_ROOT))
        from backend.training.dataset import load_jsonl
        from backend.training.train import run_training

        run_training(
            data_path=str(data_path),
            output_dir=str(model_dir),
            dry_run=True,
            use_lora=False,
        )

        model_stub_path = model_dir / "model_stub.json"
        if model_stub_path.exists():
            print(f"  -> Model artifact created: {model_stub_path}")
            results["model_path"] = str(model_stub_path)
        else:
            raise RuntimeError("model_stub.json not created!")
        print()

        # Step 3: Export to microservice
        print_step(3, "Exporting to inference microservice...")
        service_dir = workdir / "service"

        from backend.exporter.builder import build

        build(
            model_path=str(model_stub_path),
            output_dir=str(service_dir),
            overwrite=True,
            export_onnx=True,
        )

        if (service_dir / "model_metadata.json").exists() and (service_dir / "app.py").exists():
            print(f"  -> Service created at: {service_dir}")
            print("  -> Files: app.py, model_metadata.json, model_loader.py, schemas.py")
            results["service_path"] = str(service_dir)
        else:
            raise RuntimeError("Service files not created!")
        print()

        # Step 4: Run smoke test against generated service
        print_step(4, "Running smoke test against generated service...")

        # Add service directory to path for imports
        sys.path.insert(0, str(service_dir))

        # Load the app module dynamically
        app_module = load_module_from_path("service_app", service_dir / "app.py")
        app = app_module.app

        # Use FastAPI TestClient
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Test /health endpoint
        health_response = client.get("/health")
        assert health_response.status_code == 200, f"Health check failed: {health_response.status_code}"

        # Test /predict endpoint
        predict_response = client.post("/predict", json={"input": "hello"})
        assert predict_response.status_code == 200, f"Predict failed: {predict_response.status_code}"

        result = predict_response.json()
        results["response"] = result
        print("  -> Inference response:")
        print(json.dumps(result, indent=2))
        print()

        # Step 5: Summary
        print_header("Demo Complete!")
        print("Summary:")
        print(f"  Model path:     {results['model_path']}")
        print(f"  Service path:   {results['service_path']}")
        print(f"  Response:       {json.dumps(results['response'])}")
        print()

    finally:
        # Cleanup service module from sys.modules to avoid conflicts
        modules_to_remove = [
            k for k in sys.modules.keys()
            if k.startswith("service_app") or k in ("model_loader", "schemas", "app")
        ]
        for mod in modules_to_remove:
            sys.modules.pop(mod, None)

        # Remove service dir from path
        if str(service_dir) in sys.path:
            sys.path.remove(str(service_dir))

        # Cleanup or show instructions
        if cleanup:
            print("Cleaning up workspace...")
            shutil.rmtree(workdir)
            print(f"  -> Deleted: {workdir}")
            results["workdir"] = None
        else:
            print(f"Workspace preserved at: {workdir}")
            print("To delete manually:")
            if sys.platform == "win32":
                print(f"  rmdir /s /q {workdir}")
            else:
                print(f"  rm -rf {workdir}")

    print()
    print("Done!")

    return results


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TinyForgeAI End-to-End Demo (Dry-Run Mode)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete workspace after demo completes",
    )

    args = parser.parse_args()

    try:
        run_demo(cleanup=args.cleanup)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
