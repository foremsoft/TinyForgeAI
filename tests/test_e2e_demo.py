"""
End-to-end demo tests for TinyForgeAI.

Tests the complete dry-run workflow:
dataset -> train (dry-run) -> export microservice -> test inference

Uses pytest tmp_path for isolation and FastAPI TestClient for testing
the generated service without starting uvicorn.
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

# Project root for accessing modules
PROJECT_ROOT = Path(__file__).parent.parent


class TestE2EDemoWorkflow:
    """Tests for the complete end-to-end demo workflow."""

    @pytest.fixture
    def sample_data_path(self) -> Path:
        """Path to sample training data."""
        return PROJECT_ROOT / "examples" / "sample_qna.jsonl"

    def load_module_from_path(self, module_name: str, file_path: Path):
        """Dynamically load a Python module from a file path."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def test_full_e2e_workflow(self, tmp_path: Path, sample_data_path: Path):
        """Test the complete end-to-end workflow."""
        # Ensure sample data exists
        assert sample_data_path.exists(), f"Sample data not found: {sample_data_path}"

        # Step 1: Copy sample data to temp directory
        data_path = tmp_path / "data.jsonl"
        shutil.copy(sample_data_path, data_path)
        assert data_path.exists()

        # Step 2: Run dry-run trainer
        model_dir = tmp_path / "tiny_model"
        from backend.training.train import run_training

        run_training(
            data_path=str(data_path),
            output_dir=str(model_dir),
            dry_run=True,
            use_lora=False,
        )

        # Verify model stub was created
        model_stub_path = model_dir / "model_stub.json"
        assert model_stub_path.exists(), "model_stub.json not created"

        # Verify model stub content
        with open(model_stub_path) as f:
            model_data = json.load(f)
        assert model_data["model_type"] == "tinyforge_stub"
        assert "n_records" in model_data
        assert model_data["n_records"] > 0

        # Step 3: Export to microservice
        service_dir = tmp_path / "service"
        from backend.exporter.builder import build

        build(
            model_path=str(model_stub_path),
            output_dir=str(service_dir),
            overwrite=True,
            export_onnx=True,
        )

        # Verify service files were created
        assert (service_dir / "app.py").exists(), "app.py not created"
        assert (service_dir / "model_metadata.json").exists(), "model_metadata.json not created"
        assert (service_dir / "model_loader.py").exists(), "model_loader.py not created"
        assert (service_dir / "schemas.py").exists(), "schemas.py not created"
        assert (service_dir / "__init__.py").exists(), "__init__.py not created"

        # Verify ONNX export files
        assert (service_dir / "onnx" / "model.onnx").exists(), "model.onnx not created"
        assert (service_dir / "export_report.json").exists(), "export_report.json not created"

        # Step 4: Import and test the generated service
        # Add service directory to path
        sys.path.insert(0, str(service_dir))

        try:
            # Load the app module dynamically (unique name to avoid conflicts)
            test_module_name = f"test_service_app_{id(self)}"
            app_module = self.load_module_from_path(test_module_name, service_dir / "app.py")
            app = app_module.app

            # Use FastAPI TestClient
            from fastapi.testclient import TestClient

            client = TestClient(app)

            # Test /health endpoint
            health_response = client.get("/health")
            assert health_response.status_code == 200
            health_data = health_response.json()
            assert health_data["status"] == "ok"

            # Test /predict endpoint
            predict_response = client.post("/predict", json={"input": "hello"})
            assert predict_response.status_code == 200

            result = predict_response.json()
            assert "output" in result, "Response missing 'output' key"
            assert "confidence" in result, "Response missing 'confidence' key"

            # Verify stub behavior (reverses input)
            assert result["output"] == "olleh", f"Expected 'olleh', got '{result['output']}'"
            assert result["confidence"] == 0.75

        finally:
            # Cleanup: remove module from sys.modules
            modules_to_remove = [
                k for k in list(sys.modules.keys())
                if k.startswith("test_service_app") or k in ("model_loader", "schemas")
            ]
            for mod in modules_to_remove:
                sys.modules.pop(mod, None)

            # Remove service dir from path
            if str(service_dir) in sys.path:
                sys.path.remove(str(service_dir))

    def test_trainer_creates_model_stub(self, tmp_path: Path, sample_data_path: Path):
        """Test that dry-run trainer creates model_stub.json."""
        data_path = tmp_path / "data.jsonl"
        shutil.copy(sample_data_path, data_path)

        model_dir = tmp_path / "model"
        from backend.training.train import run_training

        run_training(
            data_path=str(data_path),
            output_dir=str(model_dir),
            dry_run=True,
            use_lora=False,
        )

        model_stub_path = model_dir / "model_stub.json"
        assert model_stub_path.exists()

        with open(model_stub_path) as f:
            data = json.load(f)

        assert data["model_type"] == "tinyforge_stub"
        assert data["notes"] == "dry-run artifact"

    def test_trainer_with_lora(self, tmp_path: Path, sample_data_path: Path):
        """Test that dry-run trainer with LoRA adds lora metadata."""
        data_path = tmp_path / "data.jsonl"
        shutil.copy(sample_data_path, data_path)

        model_dir = tmp_path / "model"
        from backend.training.train import run_training

        run_training(
            data_path=str(data_path),
            output_dir=str(model_dir),
            dry_run=True,
            use_lora=True,
        )

        model_stub_path = model_dir / "model_stub.json"
        with open(model_stub_path) as f:
            data = json.load(f)

        assert data["lora_applied"] is True
        assert "lora_config" in data
        assert data["lora_config"]["r"] == 8

    def test_exporter_creates_service_files(self, tmp_path: Path, sample_data_path: Path):
        """Test that exporter creates all required service files."""
        # Setup: create model stub
        data_path = tmp_path / "data.jsonl"
        shutil.copy(sample_data_path, data_path)

        model_dir = tmp_path / "model"
        from backend.training.train import run_training

        run_training(
            data_path=str(data_path),
            output_dir=str(model_dir),
            dry_run=True,
            use_lora=False,
        )

        # Export
        service_dir = tmp_path / "service"
        from backend.exporter.builder import build

        build(
            model_path=str(model_dir / "model_stub.json"),
            output_dir=str(service_dir),
            overwrite=True,
            export_onnx=False,
        )

        # Verify all expected files
        expected_files = [
            "app.py",
            "model_metadata.json",
            "model_loader.py",
            "schemas.py",
            "__init__.py",
            "requirements.txt",
        ]

        for filename in expected_files:
            assert (service_dir / filename).exists(), f"Missing: {filename}"

    def test_service_predict_various_inputs(self, tmp_path: Path, sample_data_path: Path):
        """Test service prediction with various inputs."""
        # Setup: full workflow
        data_path = tmp_path / "data.jsonl"
        shutil.copy(sample_data_path, data_path)

        model_dir = tmp_path / "model"
        from backend.training.train import run_training

        run_training(
            data_path=str(data_path),
            output_dir=str(model_dir),
            dry_run=True,
            use_lora=False,
        )

        service_dir = tmp_path / "service"
        from backend.exporter.builder import build

        build(
            model_path=str(model_dir / "model_stub.json"),
            output_dir=str(service_dir),
            overwrite=True,
            export_onnx=False,
        )

        # Import service
        sys.path.insert(0, str(service_dir))

        try:
            test_module_name = f"test_service_predict_{id(self)}"
            app_module = self.load_module_from_path(test_module_name, service_dir / "app.py")
            app = app_module.app

            from fastapi.testclient import TestClient

            client = TestClient(app)

            # Test cases: input -> expected reversed output
            test_cases = [
                ("hello", "olleh"),
                ("TinyForge", "egroFyniT"),
                ("12345", "54321"),
                ("a", "a"),
                ("", ""),
            ]

            for input_text, expected_output in test_cases:
                response = client.post("/predict", json={"input": input_text})
                assert response.status_code == 200
                result = response.json()
                assert result["output"] == expected_output, (
                    f"Input '{input_text}': expected '{expected_output}', got '{result['output']}'"
                )
                assert result["confidence"] == 0.75

        finally:
            modules_to_remove = [
                k for k in list(sys.modules.keys())
                if k.startswith("test_service_predict") or k in ("model_loader", "schemas")
            ]
            for mod in modules_to_remove:
                sys.modules.pop(mod, None)

            if str(service_dir) in sys.path:
                sys.path.remove(str(service_dir))
