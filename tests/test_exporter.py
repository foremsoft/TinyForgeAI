"""Tests for the TinyForgeAI exporter builder."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from backend.exporter.builder import build, get_template_dir


@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bin", delete=False) as f:
        f.write("fake model data")
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory path for output."""
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "output_service")
    yield output_path
    # Cleanup
    import shutil

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_build_creates_output_directory(temp_model_file, temp_output_dir):
    """Test that build creates the output directory."""
    build(model_path=temp_model_file, output_dir=temp_output_dir)
    assert os.path.isdir(temp_output_dir)


def test_build_copies_template_files(temp_model_file, temp_output_dir):
    """Test that build copies all template files to output."""
    build(model_path=temp_model_file, output_dir=temp_output_dir)

    expected_files = ["app.py", "model_loader.py", "schemas.py", "requirements.txt"]
    for filename in expected_files:
        assert os.path.exists(os.path.join(temp_output_dir, filename))


def test_build_creates_model_metadata(temp_model_file, temp_output_dir):
    """Test that build creates model_metadata.json."""
    build(model_path=temp_model_file, output_dir=temp_output_dir)

    metadata_path = os.path.join(temp_output_dir, "model_metadata.json")
    assert os.path.exists(metadata_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    assert "model_path" in metadata
    assert "created_time" in metadata
    assert metadata["source"] == "tinyforge-exporter"
    assert metadata["model_stub"] is True


def test_build_fails_if_model_not_found(temp_output_dir):
    """Test that build raises FileNotFoundError for missing model."""
    with pytest.raises(FileNotFoundError):
        build(model_path="/nonexistent/model.bin", output_dir=temp_output_dir)


def test_build_fails_if_output_exists_without_overwrite(
    temp_model_file, temp_output_dir
):
    """Test that build fails if output exists and overwrite is False."""
    # Create output directory first
    os.makedirs(temp_output_dir)

    with pytest.raises(FileExistsError):
        build(model_path=temp_model_file, output_dir=temp_output_dir, overwrite=False)


def test_build_succeeds_with_overwrite(temp_model_file, temp_output_dir):
    """Test that build succeeds when overwrite is True."""
    # First build
    build(model_path=temp_model_file, output_dir=temp_output_dir)

    # Second build with overwrite
    build(model_path=temp_model_file, output_dir=temp_output_dir, overwrite=True)

    assert os.path.isdir(temp_output_dir)
    assert os.path.exists(os.path.join(temp_output_dir, "app.py"))


def test_template_dir_exists():
    """Test that the template directory exists and contains expected files."""
    template_dir = get_template_dir()
    assert template_dir.exists()
    assert (template_dir / "app.py").exists()
    assert (template_dir / "model_loader.py").exists()
    assert (template_dir / "schemas.py").exists()
