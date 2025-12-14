"""
Tests for TinyForgeAI Playground Module

Comprehensive test suite covering:
- PlaygroundServer initialization and configuration
- Health endpoint
- HTML export functionality
- Example input handling
- Error cases (invalid inputs, missing models)
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from backend.playground.server import (
    PlaygroundConfig,
    PlaygroundServer,
    create_playground,
)
from backend.playground.exporter import (
    ExportConfig,
    PlaygroundExporter,
    export_playground,
)


# ============================================
# PlaygroundConfig Tests
# ============================================

class TestPlaygroundConfig:
    """Test PlaygroundConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PlaygroundConfig()
        assert config.model_path == ""
        assert config.model_name == "TinyForge Model"
        assert config.model_type == "Q&A"
        assert config.task_type == "question-answering"
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.share is False
        assert config.share_duration == 72
        assert config.share_provider == "ngrok"
        assert config.examples == []

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PlaygroundConfig(
            model_path="/path/to/model",
            model_name="My Custom Model",
            model_type="Classification",
            task_type="text-classification",
            title="Custom Playground",
            description="A custom model playground",
            placeholder="Enter text here...",
            default_input="Hello world",
            examples=["Example 1", "Example 2"],
            host="127.0.0.1",
            port=9000,
            share=True,
            share_duration=24,
            share_provider="cloudflare",
        )
        assert config.model_path == "/path/to/model"
        assert config.model_name == "My Custom Model"
        assert config.model_type == "Classification"
        assert config.port == 9000
        assert config.share is True
        assert len(config.examples) == 2


# ============================================
# PlaygroundServer Tests
# ============================================

class TestPlaygroundServer:
    """Test PlaygroundServer class."""

    def test_server_initialization(self):
        """Test server initialization with default config."""
        server = PlaygroundServer()
        assert server.config is not None
        assert server._model is None
        assert server._tokenizer is None
        assert server._inference_fn is None
        assert server._app is None
        assert server._public_url is None

    def test_server_initialization_with_custom_config(self):
        """Test server initialization with custom config."""
        config = PlaygroundConfig(
            model_name="Test Model",
            port=9000,
        )
        server = PlaygroundServer(config)
        assert server.config.model_name == "Test Model"
        assert server.config.port == 9000

    def test_set_inference_function(self):
        """Test setting custom inference function."""
        server = PlaygroundServer()

        def custom_fn(text: str) -> str:
            return f"Processed: {text}"

        server.set_inference_function(custom_fn)
        assert server._inference_fn is not None
        assert server._inference_fn("test") == "Processed: test"

    def test_load_model_file_not_found(self):
        """Test loading model from non-existent path raises error."""
        server = PlaygroundServer()
        with pytest.raises(FileNotFoundError):
            server.load_model("/nonexistent/path/to/model")

    def test_load_stub_model(self):
        """Test loading stub model from dry-run output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            stub_path = model_path / "model_stub.json"
            stub_path.write_text(json.dumps({
                "model_name": "test-model",
                "training_type": "dry-run",
            }))

            server = PlaygroundServer()
            server.load_model(str(model_path))

            assert server._inference_fn is not None
            result = server._inference_fn("Hello")
            assert "[Stub Response]" in result

    def test_setup_stub_inference(self):
        """Test stub inference setup."""
        server = PlaygroundServer()
        server._setup_stub_inference(None)

        assert server._inference_fn is not None
        result = server._inference_fn("Test input")
        assert "Stub Response" in result
        assert "Test input" in result


# ============================================
# FastAPI App Tests
# ============================================

class TestPlaygroundAPI:
    """Test PlaygroundServer FastAPI endpoints."""

    @pytest.fixture
    def server(self):
        """Create a server with stub inference."""
        config = PlaygroundConfig(
            title="Test Playground",
            description="Test description",
            model_name="test-model",
            examples=["Example 1", "Example 2"],
        )
        server = PlaygroundServer(config)
        server._setup_stub_inference(None)
        return server

    @pytest.fixture
    def client(self, server):
        """Create test client for the playground API."""
        from fastapi.testclient import TestClient
        app = server._create_app()
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test /health endpoint returns correct format."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "playground"

    def test_home_endpoint(self, client):
        """Test / endpoint returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Test Playground" in response.text
        assert "TinyForge" in response.text

    def test_home_contains_examples(self, client):
        """Test home page contains example chips."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Example 1" in response.text
        assert "Example 2" in response.text
        assert "example-chip" in response.text

    def test_infer_endpoint_success(self, client):
        """Test /infer endpoint with valid input."""
        response = client.post(
            "/infer",
            json={"input": "Hello, world!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert "latency_ms" in data
        assert "tokens" in data
        assert data["latency_ms"] >= 0

    def test_infer_endpoint_empty_input(self, client):
        """Test /infer endpoint with empty input returns error."""
        response = client.post(
            "/infer",
            json={"input": ""}
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_infer_endpoint_missing_input(self, client):
        """Test /infer endpoint with missing input returns error."""
        response = client.post(
            "/infer",
            json={}
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_infer_without_model(self):
        """Test /infer endpoint without loaded model returns error."""
        from fastapi.testclient import TestClient

        server = PlaygroundServer()
        app = server._create_app()
        client = TestClient(app)

        response = client.post(
            "/infer",
            json={"input": "test"}
        )
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "not loaded" in data["error"].lower()


# ============================================
# ExportConfig Tests
# ============================================

class TestExportConfig:
    """Test ExportConfig dataclass."""

    def test_default_export_config(self):
        """Test default export configuration."""
        config = ExportConfig()
        assert config.title == "TinyForge Playground"
        assert config.description == "Run AI inference directly in your browser"
        assert config.embed_model is True
        assert config.max_model_size_mb == 50

    def test_custom_export_config(self):
        """Test custom export configuration."""
        config = ExportConfig(
            title="Custom Export",
            description="Custom description",
            placeholder="Custom placeholder",
            default_input="Default text",
            embed_model=False,
            max_model_size_mb=100,
        )
        assert config.title == "Custom Export"
        assert config.embed_model is False
        assert config.max_model_size_mb == 100


# ============================================
# PlaygroundExporter Tests
# ============================================

class TestPlaygroundExporter:
    """Test PlaygroundExporter class."""

    def test_exporter_initialization(self):
        """Test exporter initialization."""
        exporter = PlaygroundExporter()
        assert exporter.config is not None
        assert exporter.config.embed_model is True

    def test_exporter_with_custom_config(self):
        """Test exporter with custom config."""
        config = ExportConfig(title="Custom Title")
        exporter = PlaygroundExporter(config)
        assert exporter.config.title == "Custom Title"

    def test_load_vocabulary_fallback(self):
        """Test vocabulary loading returns fallback when no vocab file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            exporter = PlaygroundExporter()
            vocab = exporter._load_vocabulary(model_path)

            assert "[PAD]" in vocab
            assert "[UNK]" in vocab
            assert vocab["[PAD]"] == 0
            assert vocab["[UNK]"] == 1

    def test_load_vocabulary_from_json(self):
        """Test vocabulary loading from vocab.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            vocab_file = model_path / "vocab.json"
            vocab_file.write_text(json.dumps({
                "hello": 0,
                "world": 1,
                "[UNK]": 2,
            }))

            exporter = PlaygroundExporter()
            vocab = exporter._load_vocabulary(model_path)

            assert "hello" in vocab
            assert "world" in vocab
            assert vocab["hello"] == 0

    def test_create_stub_playground(self):
        """Test creating stub playground HTML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playground.html"
            exporter = PlaygroundExporter()
            result = exporter._create_stub_playground(output_path)

            assert result == output_path
            assert output_path.exists()

            content = output_path.read_text()
            assert "TinyForge Playground" in content
            assert "Offline Export Not Available" in content


# ============================================
# Convenience Function Tests
# ============================================

class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_playground(self):
        """Test create_playground convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            stub_file = model_path / "model_stub.json"
            stub_file.write_text(json.dumps({"type": "stub"}))

            server = create_playground(
                model_path=str(model_path),
                title="Test Title",
                description="Test Description",
                examples=["ex1", "ex2"],
                port=9000,
                share=False,
            )

            assert server.config.title == "Test Title"
            assert server.config.description == "Test Description"
            assert server.config.port == 9000
            assert len(server.config.examples) == 2

    def test_export_playground_stub(self):
        """Test export_playground creates stub when no ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()

            output_path = Path(tmpdir) / "output.html"

            config = ExportConfig(title="Export Test")
            exporter = PlaygroundExporter(config)
            result = exporter._create_stub_playground(output_path)

            assert result.exists()
            content = result.read_text()
            assert "Offline Export Not Available" in content


# ============================================
# Integration Tests
# ============================================

class TestPlaygroundIntegration:
    """Integration tests for playground workflow."""

    def test_full_playground_workflow(self):
        """Test complete playground workflow with stub model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from fastapi.testclient import TestClient

            model_path = Path(tmpdir)
            stub_file = model_path / "model_stub.json"
            stub_file.write_text(json.dumps({
                "model_name": "integration-test-model",
                "type": "stub",
            }))

            config = PlaygroundConfig(
                model_path=str(model_path),
                title="Integration Test",
                examples=["Test example"],
            )

            server = PlaygroundServer(config)
            server.load_model()
            app = server._create_app()
            client = TestClient(app)

            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

            response = client.get("/")
            assert response.status_code == 200
            assert "Integration Test" in response.text

            response = client.post("/infer", json={"input": "test query"})
            assert response.status_code == 200
            assert "output" in response.json()

    def test_playground_with_custom_inference(self):
        """Test playground with custom inference function."""
        from fastapi.testclient import TestClient

        def custom_inference(text: str) -> str:
            return f"Custom result for: {text}"

        server = PlaygroundServer()
        server.set_inference_function(custom_inference)
        app = server._create_app()
        client = TestClient(app)

        response = client.post("/infer", json={"input": "hello"})
        assert response.status_code == 200
        data = response.json()
        assert "Custom result for: hello" in data["output"]
