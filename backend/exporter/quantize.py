"""
Model quantization module for TinyForgeAI.

Provides functionality to quantize models for reduced size and faster inference.
Supports multiple quantization modes including INT8, FP16, and dynamic quantization.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from backend.exceptions import QuantizationError
from backend.exporter.onnx_export import get_onnx_metadata

logger = logging.getLogger(__name__)

# Check for quantization dependencies
ONNXRUNTIME_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import onnxruntime
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
    from onnxruntime.quantization.calibrate import CalibrationDataReader
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    logger.warning(
        "ONNX Runtime quantization not available. "
        "Install with: pip install onnxruntime"
    )

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass


class CalibrationDataReaderImpl(CalibrationDataReader if ONNXRUNTIME_AVAILABLE else object):
    """Calibration data reader for static quantization."""

    def __init__(self, calibration_data: list):
        """
        Initialize with calibration data.

        Args:
            calibration_data: List of input dictionaries for calibration.
        """
        self.data = iter(calibration_data)

    def get_next(self) -> Optional[Dict[str, Any]]:
        """Get next calibration sample."""
        try:
            return next(self.data)
        except StopIteration:
            return None


def quantize_onnx(
    onnx_path: str,
    output_path: str,
    mode: str = "int8_dynamic",
    calibration_data: Optional[list] = None,
) -> str:
    """
    Quantize an ONNX model.

    Args:
        onnx_path: Path to the source ONNX file.
        output_path: Path for the quantized output file.
        mode: Quantization mode. Options:
            - "int8_dynamic": Dynamic INT8 quantization (default, no calibration needed)
            - "int8_static": Static INT8 quantization (requires calibration data)
            - "uint8_dynamic": Dynamic UINT8 quantization
            - "uint8_static": Static UINT8 quantization
            - "fp16": Float16 quantization (requires GPU support)
        calibration_data: List of input dictionaries for static quantization.

    Returns:
        Path to the quantized ONNX file.

    Raises:
        QuantizationError: If quantization fails.
        FileNotFoundError: If source ONNX file doesn't exist.
    """
    onnx_file = Path(onnx_path)
    output_file = Path(output_path)

    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Valid modes
    valid_modes = {"int8_dynamic", "int8_static", "uint8_dynamic", "uint8_static", "fp16"}
    if mode not in valid_modes:
        raise QuantizationError(
            f"Invalid quantization mode: {mode}. Valid modes: {', '.join(sorted(valid_modes))}",
            quantization_type=mode,
        )

    # Check dependencies
    if not ONNXRUNTIME_AVAILABLE:
        logger.warning("ONNX Runtime not available, falling back to stub quantization")
        return _quantize_stub(onnx_path, str(output_file), mode)

    logger.info(f"Quantizing model: {onnx_path} -> {output_path} (mode: {mode})")

    try:
        if mode == "fp16":
            _quantize_fp16(onnx_file, output_file)
        elif mode.endswith("_dynamic"):
            quant_type = QuantType.QInt8 if mode.startswith("int8") else QuantType.QUInt8
            _quantize_dynamic(onnx_file, output_file, quant_type)
        else:  # static quantization
            if not calibration_data:
                raise QuantizationError(
                    "Static quantization requires calibration_data",
                    quantization_type=mode,
                )
            quant_type = QuantType.QInt8 if mode.startswith("int8") else QuantType.QUInt8
            _quantize_static(onnx_file, output_file, quant_type, calibration_data)

    except QuantizationError:
        raise
    except Exception as e:
        raise QuantizationError(
            f"Quantization failed: {e}",
            quantization_type=mode,
        )

    # Write metadata
    _write_quantization_metadata(
        output_file,
        source_path=str(onnx_file),
        mode=mode,
    )

    # Log size reduction
    original_size = onnx_file.stat().st_size
    quantized_size = output_file.stat().st_size
    reduction = (1 - quantized_size / original_size) * 100
    logger.info(
        f"Quantization complete. Size: {original_size / 1024 / 1024:.2f}MB -> "
        f"{quantized_size / 1024 / 1024:.2f}MB ({reduction:.1f}% reduction)"
    )

    return str(output_file)


def _quantize_dynamic(
    input_path: Path,
    output_path: Path,
    quant_type: "QuantType",
) -> None:
    """Apply dynamic quantization."""
    from onnxruntime.quantization import quantize_dynamic

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=quant_type,
        optimize_model=True,
    )
    logger.info(f"Dynamic quantization complete: {output_path}")


def _quantize_static(
    input_path: Path,
    output_path: Path,
    quant_type: "QuantType",
    calibration_data: list,
) -> None:
    """Apply static quantization with calibration."""
    from onnxruntime.quantization import quantize_static

    # Create calibration data reader
    calibration_reader = CalibrationDataReaderImpl(calibration_data)

    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        quant_format=quant_type,
        optimize_model=True,
    )
    logger.info(f"Static quantization complete: {output_path}")


def _quantize_fp16(input_path: Path, output_path: Path) -> None:
    """Convert model to FP16."""
    try:
        from onnxruntime.transformers import float16
        import onnx

        model = onnx.load(str(input_path))
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, str(output_path))
        logger.info(f"FP16 conversion complete: {output_path}")

    except ImportError:
        # Fallback using onnxconverter-common
        try:
            from onnxconverter_common import float16
            import onnx

            model = onnx.load(str(input_path))
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, str(output_path))
            logger.info(f"FP16 conversion complete: {output_path}")

        except ImportError:
            raise QuantizationError(
                "FP16 quantization requires onnxruntime-tools or onnxconverter-common. "
                "Install with: pip install onnxruntime-tools",
                quantization_type="fp16",
            )


def _quantize_stub(onnx_path: str, output_path: str, mode: str) -> str:
    """
    Fallback stub quantization when dependencies are not available.

    Creates a placeholder file with metadata.
    """
    onnx_file = Path(onnx_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

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
        "note": "Stub quantization - ONNX Runtime not available",
    }

    # Write placeholder quantized ONNX file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# quantized: {mode}\n")
        f.write("# TinyForgeAI Quantized ONNX placeholder\n")
        f.write(f"# Quantized: {quantize_time}\n")
        f.write(f"# Source: {onnx_path}\n")
        f.write("# This is a stub file - install ONNX Runtime for real quantization.\n")
        f.write("# pip install onnxruntime\n")

    # Write metadata file
    meta_file_path = output_file.parent / f"{output_file.name}.meta.json"
    with open(meta_file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.warning(f"Created stub quantized file at: {output_file}")
    return str(output_file)


def _write_quantization_metadata(
    output_path: Path,
    source_path: str,
    mode: str,
) -> None:
    """Write quantization metadata."""
    # Get source metadata if available
    source_metadata = get_onnx_metadata(source_path) or {}

    metadata = {
        **source_metadata,
        "quantized": True,
        "quantization_mode": mode,
        "quantize_time": datetime.now(timezone.utc).isoformat(),
        "source_onnx": source_path,
    }

    # Update format info
    if mode == "fp16":
        metadata["precision"] = "float16"
    elif mode.startswith("int8"):
        metadata["precision"] = "int8"
    elif mode.startswith("uint8"):
        metadata["precision"] = "uint8"

    meta_path = output_path.parent / f"{output_path.name}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def get_quantized_metadata(quantized_path: str) -> Optional[Dict[str, Any]]:
    """
    Read metadata for a quantized ONNX file.

    Args:
        quantized_path: Path to the quantized ONNX file.

    Returns:
        Metadata dictionary if metadata file exists, None otherwise.
    """
    meta_path = Path(quantized_path).parent / f"{Path(quantized_path).name}.meta.json"
    if not meta_path.exists():
        # Try legacy location
        meta_path = Path(quantized_path).with_suffix(".onnx.meta.json")

    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_quantization_speedup(mode: str) -> Dict[str, Any]:
    """
    Estimate performance improvements for a quantization mode.

    Args:
        mode: Quantization mode.

    Returns:
        Dictionary with estimated speedup and size reduction.
    """
    estimates = {
        "int8_dynamic": {
            "size_reduction": "2-4x",
            "inference_speedup": "1.5-3x",
            "accuracy_loss": "< 1%",
            "requires_calibration": False,
        },
        "int8_static": {
            "size_reduction": "2-4x",
            "inference_speedup": "2-4x",
            "accuracy_loss": "< 0.5%",
            "requires_calibration": True,
        },
        "uint8_dynamic": {
            "size_reduction": "2-4x",
            "inference_speedup": "1.5-3x",
            "accuracy_loss": "< 1%",
            "requires_calibration": False,
        },
        "uint8_static": {
            "size_reduction": "2-4x",
            "inference_speedup": "2-4x",
            "accuracy_loss": "< 0.5%",
            "requires_calibration": True,
        },
        "fp16": {
            "size_reduction": "2x",
            "inference_speedup": "1.5-2x (GPU)",
            "accuracy_loss": "negligible",
            "requires_calibration": False,
            "note": "Best with GPU inference",
        },
    }

    return estimates.get(mode, {"error": f"Unknown mode: {mode}"})


def list_quantization_modes() -> Dict[str, str]:
    """
    List available quantization modes with descriptions.

    Returns:
        Dictionary mapping mode names to descriptions.
    """
    return {
        "int8_dynamic": "Dynamic INT8 quantization - no calibration needed, good balance",
        "int8_static": "Static INT8 quantization - requires calibration, best accuracy",
        "uint8_dynamic": "Dynamic UINT8 quantization - no calibration needed",
        "uint8_static": "Static UINT8 quantization - requires calibration",
        "fp16": "Float16 quantization - best for GPU inference",
    }
