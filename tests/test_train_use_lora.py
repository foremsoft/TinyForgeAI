"""Tests for training with LoRA adapter."""

import json
import os
import subprocess
import sys

import pytest


def test_train_with_lora_creates_model_stub(tmp_path):
    """Test that training with --use-lora creates model_stub.json."""
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
            "--use-lora",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )

    assert result.returncode == 0, f"Training failed: {result.stderr}"
    assert (tmp_path / "model_stub.json").exists()


def test_train_with_lora_sets_lora_applied_true(tmp_path):
    """Test that model_stub.json contains lora_applied: true."""
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
            "--use-lora",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )

    with open(tmp_path / "model_stub.json", "r") as f:
        artifact = json.load(f)

    assert artifact["lora_applied"] is True


def test_train_with_lora_includes_lora_config(tmp_path):
    """Test that model_stub.json contains lora_config."""
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
            "--use-lora",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )

    with open(tmp_path / "model_stub.json", "r") as f:
        artifact = json.load(f)

    assert "lora_config" in artifact
    assert artifact["lora_config"]["r"] == 8
    assert artifact["lora_config"]["alpha"] == 16


def test_train_with_lora_includes_lora_timestamp(tmp_path):
    """Test that model_stub.json contains lora_timestamp."""
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
            "--use-lora",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )

    with open(tmp_path / "model_stub.json", "r") as f:
        artifact = json.load(f)

    assert "lora_timestamp" in artifact
    assert len(artifact["lora_timestamp"]) > 0


def test_train_without_lora_has_no_lora_keys(tmp_path):
    """Test that training without --use-lora does not include LoRA keys."""
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

    assert "lora_applied" not in artifact
    assert "lora_config" not in artifact
