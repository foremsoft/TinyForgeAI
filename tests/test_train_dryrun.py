"""Tests for the dry-run training script."""

import json
import os
import subprocess
import sys

import pytest


def test_train_dryrun_creates_model_stub(tmp_path):
    """Test that dry-run training creates model_stub.json."""
    sample_file = os.path.join(
        os.path.dirname(__file__), "..", "examples", "sample_qna.jsonl"
    )

    result = subprocess.run(
        [
            sys.executable,
            "backend/training/train.py",
            "--data",
            sample_file,
            "--out",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )

    assert result.returncode == 0, f"Training failed: {result.stderr}"
    assert (tmp_path / "model_stub.json").exists()


def test_train_dryrun_model_stub_has_correct_n_records(tmp_path):
    """Test that model_stub.json contains n_records == 3."""
    sample_file = os.path.join(
        os.path.dirname(__file__), "..", "examples", "sample_qna.jsonl"
    )

    subprocess.run(
        [
            sys.executable,
            "backend/training/train.py",
            "--data",
            sample_file,
            "--out",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )

    with open(tmp_path / "model_stub.json", "r") as f:
        artifact = json.load(f)

    assert artifact["n_records"] == 3


def test_train_dryrun_model_stub_has_required_fields(tmp_path):
    """Test that model_stub.json contains all required fields."""
    sample_file = os.path.join(
        os.path.dirname(__file__), "..", "examples", "sample_qna.jsonl"
    )

    subprocess.run(
        [
            sys.executable,
            "backend/training/train.py",
            "--data",
            sample_file,
            "--out",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )

    with open(tmp_path / "model_stub.json", "r") as f:
        artifact = json.load(f)

    assert "model_type" in artifact
    assert artifact["model_type"] == "tinyforge_stub"
    assert "created_time" in artifact
    assert "notes" in artifact


def test_train_without_dryrun_also_works(tmp_path):
    """Test that training without --dry-run flag also creates artifact."""
    sample_file = os.path.join(
        os.path.dirname(__file__), "..", "examples", "sample_qna.jsonl"
    )

    result = subprocess.run(
        [
            sys.executable,
            "backend/training/train.py",
            "--data",
            sample_file,
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )

    assert result.returncode == 0
    assert (tmp_path / "model_stub.json").exists()


def test_train_fails_on_missing_data_file(tmp_path):
    """Test that training fails gracefully for missing data file."""
    result = subprocess.run(
        [
            sys.executable,
            "backend/training/train.py",
            "--data",
            "/nonexistent/data.jsonl",
            "--out",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )

    assert result.returncode != 0
