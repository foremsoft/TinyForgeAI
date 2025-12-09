"""
ONNX export stub for TinyForgeAI.

Provides functionality to create placeholder ONNX files from model stub artifacts.
This is a lightweight implementation for offline testing and development.
Real ONNX export would require torch.onnx.export or similar.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def export_to_onnx(model_stub_path: str, output_dir: str) -> str:
    """
    Read a stub model JSON and write a placeholder ONNX file.

    Creates a placeholder ONNX file with metadata about the export.
    This stub implementation does not perform real ONNX conversion.

    Args:
        model_stub_path: Path to the model stub JSON file.
        output_dir: Directory where the ONNX file will be written.

    Returns:
        Path to the created ONNX file (output_dir/model.onnx).

    Raises:
        FileNotFoundError: If model_stub_path does not exist.
        ValueError: If model_stub_path is not a JSON file.
    """
    model_path = Path(model_stub_path)
    output_path = Path(output_dir)

    # Validate model stub path exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model stub not found: {model_stub_path}")

    # Validate it's a JSON file
    if model_path.suffix.lower() != ".json":
        raise ValueError(
            f"Expected JSON model stub file, got: {model_path.suffix}. "
            f"File: {model_stub_path}"
        )

    # Read the model stub to include in metadata
    with open(model_path, "r", encoding="utf-8") as f:
        model_stub_data = json.load(f)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate export metadata
    export_time = datetime.now(timezone.utc).isoformat()
    metadata = {
        "exported_from": str(model_stub_path),
        "export_time": export_time,
        "version": "0.1-stub",
        "source_model": model_stub_data,
    }

    # Write placeholder ONNX file
    onnx_file_path = output_path / "model.onnx"
    with open(onnx_file_path, "w", encoding="utf-8") as f:
        f.write("# TinyForgeAI ONNX placeholder\n")
        f.write(f"# Exported: {export_time}\n")
        f.write(f"# Source: {model_stub_path}\n")
        f.write("# This is a stub file for testing purposes.\n")

    # Write metadata file
    meta_file_path = output_path / "model.onnx.meta.json"
    with open(meta_file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return str(onnx_file_path)


def get_onnx_metadata(onnx_path: str) -> Optional[dict]:
    """
    Read metadata for an ONNX file.

    Args:
        onnx_path: Path to the ONNX file.

    Returns:
        Metadata dictionary if metadata file exists, None otherwise.
    """
    meta_path = Path(onnx_path).with_suffix(".onnx.meta.json")
    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
