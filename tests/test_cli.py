"""Tests for the foremforge CLI."""

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from cli.foremforge import cli, PROJECT_DIRS, STARTER_FILES


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_data_file(tmp_path):
    """Create a sample JSONL data file."""
    data_file = tmp_path / "sample.jsonl"
    records = [
        {"input": "What is 2+2?", "output": "4"},
        {"input": "What is the capital of France?", "output": "Paris"},
        {"input": "Hello", "output": "Hi there!"},
    ]
    with open(data_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return data_file


@pytest.fixture
def model_stub_file(tmp_path):
    """Create a sample model stub JSON file."""
    stub_data = {
        "model_type": "tinyforge_stub",
        "n_records": 3,
        "created_time": "2025-01-01T00:00:00Z",
    }
    stub_file = tmp_path / "model_stub.json"
    with open(stub_file, "w") as f:
        json.dump(stub_data, f)
    return stub_file


@pytest.fixture
def service_dir(tmp_path):
    """Create a dummy service directory with app.py."""
    service_path = tmp_path / "service"
    service_path.mkdir()

    # Create a minimal app.py
    app_content = '''"""Minimal FastAPI app for testing."""
from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}
'''
    (service_path / "app.py").write_text(app_content)
    return service_path


class TestCliHelp:
    """Tests for CLI help messages."""

    def test_main_help(self, runner):
        """Test that main --help works."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "TinyForgeAI CLI" in result.output

    def test_init_help(self, runner):
        """Test that init --help works."""
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.output

    def test_train_help(self, runner):
        """Test that train --help works."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--data" in result.output
        assert "--dry-run" in result.output

    def test_export_help(self, runner):
        """Test that export --help works."""
        result = runner.invoke(cli, ["export", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--export-onnx" in result.output

    def test_serve_help(self, runner):
        """Test that serve --help works."""
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--dry-run" in result.output

    def test_version(self, runner):
        """Test that --version works."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestInitCommand:
    """Tests for the init command."""

    def test_init_creates_directories(self, runner, tmp_path):
        """Test that init --yes creates project directories."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--yes"])

            assert result.exit_code == 0
            assert "Initialization complete" in result.output

            # Check that directories were created
            for dir_name in ["backend", "connectors", "docs", "tests"]:
                assert Path(dir_name).exists(), f"Directory {dir_name} not created"

    def test_init_creates_starter_files(self, runner, tmp_path):
        """Test that init --yes creates starter files."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--yes"])

            assert result.exit_code == 0

            # Check that starter files were created
            assert Path("README.md").exists()
            assert Path(".env.example").exists()

    def test_init_reports_created_items(self, runner, tmp_path):
        """Test that init output mentions created items."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--yes"])

            assert result.exit_code == 0
            assert "Created directory" in result.output or "Created file" in result.output

    def test_init_idempotent(self, runner, tmp_path):
        """Test that init on existing structure says nothing to create."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # First init
            runner.invoke(cli, ["init", "--yes"])

            # Second init should detect existing structure
            result = runner.invoke(cli, ["init", "--yes"])

            assert result.exit_code == 0
            assert "already exists" in result.output or "Nothing to create" in result.output

    def test_init_without_yes_prompts(self, runner, tmp_path):
        """Test that init without --yes prompts for confirmation."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Provide 'n' to decline
            result = runner.invoke(cli, ["init"], input="n\n")

            assert result.exit_code == 1
            assert "Aborted" in result.output


class TestTrainCommand:
    """Tests for the train command."""

    def test_train_dry_run_succeeds(self, runner, sample_data_file, tmp_path):
        """Test that train --dry-run completes successfully."""
        out_dir = tmp_path / "model_out"

        result = runner.invoke(cli, [
            "train",
            "--data", str(sample_data_file),
            "--out", str(out_dir),
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "dry-run" in result.output.lower()

    def test_train_creates_artifact(self, runner, sample_data_file, tmp_path):
        """Test that train creates model artifact."""
        out_dir = tmp_path / "model_out"

        result = runner.invoke(cli, [
            "train",
            "--data", str(sample_data_file),
            "--out", str(out_dir),
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert (out_dir / "model_stub.json").exists()

    def test_train_reports_n_records(self, runner, sample_data_file, tmp_path):
        """Test that train output contains n_records."""
        out_dir = tmp_path / "model_out"

        result = runner.invoke(cli, [
            "train",
            "--data", str(sample_data_file),
            "--out", str(out_dir),
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "n_records" in result.output
        assert "3" in result.output  # Our sample has 3 records

    def test_train_reports_artifact_path(self, runner, sample_data_file, tmp_path):
        """Test that train output contains artifact path."""
        out_dir = tmp_path / "model_out"

        result = runner.invoke(cli, [
            "train",
            "--data", str(sample_data_file),
            "--out", str(out_dir),
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "artifact" in result.output
        assert "model_stub.json" in result.output

    def test_train_with_lora(self, runner, sample_data_file, tmp_path):
        """Test that train --use-lora works."""
        out_dir = tmp_path / "model_out"

        result = runner.invoke(cli, [
            "train",
            "--data", str(sample_data_file),
            "--out", str(out_dir),
            "--dry-run",
            "--use-lora",
        ])

        assert result.exit_code == 0
        assert "LoRA" in result.output

    def test_train_missing_data_fails(self, runner, tmp_path):
        """Test that train with missing data file fails."""
        out_dir = tmp_path / "model_out"

        result = runner.invoke(cli, [
            "train",
            "--data", "/nonexistent/data.jsonl",
            "--out", str(out_dir),
            "--dry-run",
        ])

        # Click validates path exists, so this should fail
        assert result.exit_code != 0


class TestExportCommand:
    """Tests for the export command."""

    def test_export_creates_service(self, runner, model_stub_file, tmp_path):
        """Test that export creates microservice directory."""
        out_dir = tmp_path / "service_out"

        result = runner.invoke(cli, [
            "export",
            "--model", str(model_stub_file),
            "--out", str(out_dir),
        ])

        assert result.exit_code == 0
        assert out_dir.exists()

    def test_export_creates_metadata(self, runner, model_stub_file, tmp_path):
        """Test that export creates model_metadata.json."""
        out_dir = tmp_path / "service_out"

        result = runner.invoke(cli, [
            "export",
            "--model", str(model_stub_file),
            "--out", str(out_dir),
        ])

        assert result.exit_code == 0
        assert (out_dir / "model_metadata.json").exists()

    def test_export_reports_location(self, runner, model_stub_file, tmp_path):
        """Test that export output mentions service location."""
        out_dir = tmp_path / "service_out"

        result = runner.invoke(cli, [
            "export",
            "--model", str(model_stub_file),
            "--out", str(out_dir),
        ])

        assert result.exit_code == 0
        assert "microservice" in result.output
        assert str(out_dir) in result.output

    def test_export_with_overwrite(self, runner, model_stub_file, tmp_path):
        """Test that export --overwrite replaces existing directory."""
        out_dir = tmp_path / "service_out"

        # First export
        runner.invoke(cli, [
            "export",
            "--model", str(model_stub_file),
            "--out", str(out_dir),
        ])

        # Second export with overwrite
        result = runner.invoke(cli, [
            "export",
            "--model", str(model_stub_file),
            "--out", str(out_dir),
            "--overwrite",
        ])

        assert result.exit_code == 0

    def test_export_without_overwrite_fails_on_existing(self, runner, model_stub_file, tmp_path):
        """Test that export fails on existing directory without --overwrite."""
        out_dir = tmp_path / "service_out"

        # First export
        runner.invoke(cli, [
            "export",
            "--model", str(model_stub_file),
            "--out", str(out_dir),
        ])

        # Second export without overwrite
        result = runner.invoke(cli, [
            "export",
            "--model", str(model_stub_file),
            "--out", str(out_dir),
        ])

        assert result.exit_code != 0
        assert "Error" in result.output

    def test_export_with_onnx(self, runner, model_stub_file, tmp_path):
        """Test that export --export-onnx creates ONNX files."""
        out_dir = tmp_path / "service_out"

        result = runner.invoke(cli, [
            "export",
            "--model", str(model_stub_file),
            "--out", str(out_dir),
            "--export-onnx",
        ])

        assert result.exit_code == 0
        assert "onnx_path" in result.output
        assert (out_dir / "onnx" / "model.onnx").exists()


class TestServeCommand:
    """Tests for the serve command."""

    def test_serve_dry_run_shows_command(self, runner, service_dir):
        """Test that serve --dry-run shows the uvicorn command."""
        result = runner.invoke(cli, [
            "serve",
            "--dir", str(service_dir),
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "uvicorn app:app" in result.output
        assert "--host 0.0.0.0" in result.output
        assert "--port 8000" in result.output

    def test_serve_dry_run_with_custom_port(self, runner, service_dir):
        """Test that serve --dry-run shows custom port."""
        result = runner.invoke(cli, [
            "serve",
            "--dir", str(service_dir),
            "--port", "9000",
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "--port 9000" in result.output

    def test_serve_dry_run_with_custom_host(self, runner, service_dir):
        """Test that serve --dry-run shows custom host."""
        result = runner.invoke(cli, [
            "serve",
            "--dir", str(service_dir),
            "--host", "127.0.0.1",
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "--host 127.0.0.1" in result.output

    def test_serve_missing_app_fails(self, runner, tmp_path):
        """Test that serve fails if app.py is missing."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = runner.invoke(cli, [
            "serve",
            "--dir", str(empty_dir),
            "--dry-run",
        ])

        assert result.exit_code != 0
        assert "app.py not found" in result.output

    def test_serve_shows_service_dir(self, runner, service_dir):
        """Test that serve --dry-run shows the service directory."""
        result = runner.invoke(cli, [
            "serve",
            "--dir", str(service_dir),
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert str(service_dir) in result.output
