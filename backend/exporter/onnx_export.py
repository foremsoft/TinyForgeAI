"""
ONNX export module for TinyForgeAI.

Provides functionality to export trained models to ONNX format for
optimized inference across different platforms and runtimes.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.exceptions import ONNXExportError, ModelNotFoundError

logger = logging.getLogger(__name__)

# Check for ONNX export dependencies
ONNX_AVAILABLE = False
TRANSFORMERS_ONNX_AVAILABLE = False

try:
    import torch
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    logger.warning(
        "ONNX export dependencies not installed. "
        "Install with: pip install torch onnx"
    )

try:
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
    from transformers.onnx import export as transformers_onnx_export
    from transformers.onnx import FeaturesManager
    TRANSFORMERS_ONNX_AVAILABLE = True
except ImportError:
    logger.warning(
        "Transformers ONNX support not available. "
        "Install with: pip install transformers[onnx]"
    )


def export_to_onnx(
    model_path: str,
    output_dir: str,
    opset_version: int = 14,
    model_type: str = "seq2seq",
    optimize: bool = True,
) -> str:
    """
    Export a trained model to ONNX format.

    Args:
        model_path: Path to the trained model directory or HuggingFace model name.
        output_dir: Directory where the ONNX file will be written.
        opset_version: ONNX opset version (default: 14).
        model_type: Model type - "seq2seq" or "causal" (default: seq2seq).
        optimize: Whether to optimize the ONNX model (default: True).

    Returns:
        Path to the created ONNX file.

    Raises:
        ONNXExportError: If export fails.
        ModelNotFoundError: If the model cannot be found.
    """
    model_path = Path(model_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if we have real ONNX export capabilities
    if not ONNX_AVAILABLE:
        logger.warning("ONNX not available, falling back to stub export")
        return _export_stub(str(model_path), str(output_path))

    if not TRANSFORMERS_ONNX_AVAILABLE:
        logger.warning("Transformers ONNX not available, falling back to stub export")
        return _export_stub(str(model_path), str(output_path))

    logger.info(f"Exporting model to ONNX: {model_path} -> {output_dir}")

    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            feature = "seq2seq-lm"
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            feature = "causal-lm"

        model.eval()

    except OSError as e:
        if "does not appear to have" in str(e) or "not found" in str(e).lower():
            raise ModelNotFoundError(str(model_path))
        raise ONNXExportError(f"Failed to load model: {e}", model_name=str(model_path))
    except Exception as e:
        raise ONNXExportError(f"Failed to load model: {e}", model_name=str(model_path))

    # Export using transformers ONNX export
    onnx_file_path = output_path / "model.onnx"

    try:
        # Get the ONNX config for the model
        onnx_config_class = FeaturesManager.get_config(type(model).__name__, feature)
        onnx_config = onnx_config_class(model.config)

        # Perform export
        transformers_onnx_export(
            preprocessor=tokenizer,
            model=model,
            config=onnx_config,
            opset=opset_version,
            output=onnx_file_path,
        )

        logger.info(f"ONNX model exported to: {onnx_file_path}")

    except Exception as e:
        # Fall back to manual torch.onnx.export
        logger.warning(f"Transformers ONNX export failed: {e}, trying torch.onnx.export")
        try:
            _export_with_torch(model, tokenizer, onnx_file_path, opset_version)
        except Exception as torch_e:
            raise ONNXExportError(
                f"ONNX export failed: {torch_e}",
                model_name=str(model_path),
            )

    # Optimize if requested
    if optimize and onnx_file_path.exists():
        try:
            _optimize_onnx(onnx_file_path)
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")

    # Validate the exported model
    try:
        _validate_onnx(onnx_file_path)
    except Exception as e:
        raise ONNXExportError(
            f"ONNX validation failed: {e}",
            model_name=str(model_path),
        )

    # Write metadata
    _write_metadata(
        output_path,
        model_path=str(model_path),
        opset_version=opset_version,
        model_type=model_type,
        optimized=optimize,
    )

    return str(onnx_file_path)


def _export_with_torch(
    model,
    tokenizer,
    output_path: Path,
    opset_version: int,
) -> None:
    """Export model using torch.onnx.export directly."""
    import torch

    # Create dummy input
    dummy_text = "Hello, this is a test input for ONNX export."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    # Get input names
    input_names = list(inputs.keys())

    # Dynamic axes for variable sequence length
    dynamic_axes = {name: {0: "batch_size", 1: "sequence"} for name in input_names}

    # Determine output names based on model type
    if hasattr(model, "generate"):
        output_names = ["logits"]
        dynamic_axes["logits"] = {0: "batch_size", 1: "sequence"}
    else:
        output_names = ["output"]
        dynamic_axes["output"] = {0: "batch_size"}

    # Export
    torch.onnx.export(
        model,
        tuple(inputs.values()),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info(f"Model exported with torch.onnx.export to: {output_path}")


def _optimize_onnx(onnx_path: Path) -> None:
    """Optimize ONNX model for inference."""
    try:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions

        # Load and optimize
        opt_model = optimizer.optimize_model(
            str(onnx_path),
            model_type="bert",  # Generic transformer optimization
            num_heads=0,  # Auto-detect
            hidden_size=0,  # Auto-detect
        )

        # Save optimized model
        optimized_path = onnx_path.parent / "model_optimized.onnx"
        opt_model.save_model_to_file(str(optimized_path))

        # Replace original with optimized
        optimized_path.rename(onnx_path)

        logger.info("ONNX model optimized successfully")

    except ImportError:
        logger.warning(
            "onnxruntime-tools not available for optimization. "
            "Install with: pip install onnxruntime-tools"
        )
    except Exception as e:
        logger.warning(f"ONNX optimization skipped: {e}")


def _validate_onnx(onnx_path: Path) -> None:
    """Validate the exported ONNX model."""
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    logger.info("ONNX model validation passed")


def _write_metadata(
    output_dir: Path,
    model_path: str,
    opset_version: int,
    model_type: str,
    optimized: bool,
) -> None:
    """Write export metadata."""
    metadata = {
        "exported_from": model_path,
        "export_time": datetime.now(timezone.utc).isoformat(),
        "format": "onnx",
        "opset_version": opset_version,
        "model_type": model_type,
        "optimized": optimized,
        "version": "1.0.0",
    }

    meta_path = output_dir / "model.onnx.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _export_stub(model_stub_path: str, output_dir: str) -> str:
    """
    Fallback stub export when ONNX dependencies are not available.

    Creates a placeholder file with metadata.
    """
    model_path = Path(model_stub_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate export metadata
    export_time = datetime.now(timezone.utc).isoformat()
    metadata = {
        "exported_from": str(model_stub_path),
        "export_time": export_time,
        "version": "0.1-stub",
        "note": "Stub export - ONNX dependencies not available",
    }

    # Try to read source model metadata
    if model_path.is_dir():
        meta_file = model_path / "model_metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata["source_model"] = json.load(f)
    elif model_path.suffix.lower() == ".json" and model_path.exists():
        with open(model_path, "r", encoding="utf-8") as f:
            metadata["source_model"] = json.load(f)

    # Write placeholder ONNX file
    onnx_file_path = output_path / "model.onnx"
    with open(onnx_file_path, "w", encoding="utf-8") as f:
        f.write("# TinyForgeAI ONNX placeholder\n")
        f.write(f"# Exported: {export_time}\n")
        f.write(f"# Source: {model_stub_path}\n")
        f.write("# This is a stub file - install ONNX dependencies for real export.\n")
        f.write("# pip install torch onnx transformers[onnx]\n")

    # Write metadata file
    meta_file_path = output_path / "model.onnx.meta.json"
    with open(meta_file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.warning(f"Created stub ONNX file at: {onnx_file_path}")
    return str(onnx_file_path)


def get_onnx_metadata(onnx_path: str) -> Optional[Dict[str, Any]]:
    """
    Read metadata for an ONNX file.

    Args:
        onnx_path: Path to the ONNX file.

    Returns:
        Metadata dictionary if metadata file exists, None otherwise.
    """
    meta_path = Path(onnx_path).with_suffix(".onnx.meta.json")
    if not meta_path.exists():
        # Try alternate location
        meta_path = Path(onnx_path).parent / "model.onnx.meta.json"

    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_onnx_info(onnx_path: str) -> Optional[Dict[str, Any]]:
    """
    Get information about an ONNX model.

    Args:
        onnx_path: Path to the ONNX file.

    Returns:
        Dictionary with model info or None if file doesn't exist.
    """
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        return None

    info = {
        "path": str(onnx_file),
        "size_bytes": onnx_file.stat().st_size,
        "size_mb": round(onnx_file.stat().st_size / (1024 * 1024), 2),
    }

    # Try to get detailed info if onnx is available
    if ONNX_AVAILABLE:
        try:
            import onnx

            model = onnx.load(str(onnx_file))
            info["opset_version"] = model.opset_import[0].version
            info["ir_version"] = model.ir_version
            info["producer"] = model.producer_name
            info["num_nodes"] = len(model.graph.node)
            info["inputs"] = [
                {"name": inp.name, "shape": [d.dim_value for d in inp.type.tensor_type.shape.dim]}
                for inp in model.graph.input
            ]
            info["outputs"] = [
                {"name": out.name}
                for out in model.graph.output
            ]
        except Exception as e:
            logger.warning(f"Could not read ONNX model details: {e}")

    # Add metadata if available
    metadata = get_onnx_metadata(onnx_path)
    if metadata:
        info["metadata"] = metadata

    return info
