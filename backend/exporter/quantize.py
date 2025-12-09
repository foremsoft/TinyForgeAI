"""
Quantization hook for TinyForgeAI.

Provides functionality to create placeholder quantized model files.
This is a lightweight implementation for offline testing and development.
Real quantization would require onnxruntime or similar quantization tools.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.exporter.onnx_export import get_onnx_metadata


def quantize_onnx(
    onnx_path: str,
    quantized_output_path: str,
    mode: str = "int8",
) -> str:
    """
    Create a placeholder quantized model file.

    Creates a placeholder quantized ONNX file with metadata about the quantization.
    This stub implementation does not perform real quantization.

    Args:
        onnx_path: Path to the source ONNX file.
        quantized_output_path: Directory where the quantized file will be written.
        mode: Quantization mode (e.g., "int8", "fp16"). Default is "int8".

    Returns:
        Path to the created quantized ONNX file (quantized_output_path/quantized.onnx).

    Raises:
        FileNotFoundError: If onnx_path does not exist.
        ValueError: If mode is not a valid quantization mode.
    """
    onnx_file = Path(onnx_path)
    output_path = Path(quantized_output_path)

    # Validate source ONNX file exists
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    # Validate quantization mode
    valid_modes = {"int8", "fp16", "uint8", "int4"}
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid quantization mode: {mode}. "
            f"Valid modes: {', '.join(sorted(valid_modes))}"
        )

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get original ONNX metadata if available
    original_metadata = get_onnx_metadata(onnx_path) or {}

    # Generate quantization metadata
    quantize_time = datetime.now(timezone.utc).isoformat()
    metadata = {
        **original_metadata,
        "quantized": True,
        "mode": mode,
        "quantize_time": quantize_time,
        "source_onnx": str(onnx_path),
    }

    # Write placeholder quantized ONNX file
    quantized_file_path = output_path / "quantized.onnx"
    with open(quantized_file_path, "w", encoding="utf-8") as f:
        f.write(f"# quantized: {mode}\n")
        f.write("# TinyForgeAI Quantized ONNX placeholder\n")
        f.write(f"# Quantized: {quantize_time}\n")
        f.write(f"# Source: {onnx_path}\n")
        f.write("# This is a stub file for testing purposes.\n")

    # Write metadata file
    meta_file_path = output_path / "quantized.onnx.meta.json"
    with open(meta_file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return str(quantized_file_path)


def get_quantized_metadata(quantized_path: str) -> Optional[dict]:
    """
    Read metadata for a quantized ONNX file.

    Args:
        quantized_path: Path to the quantized ONNX file.

    Returns:
        Metadata dictionary if metadata file exists, None otherwise.
    """
    meta_path = Path(quantized_path).with_suffix(".onnx.meta.json")
    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
