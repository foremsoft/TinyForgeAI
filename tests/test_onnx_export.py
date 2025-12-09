"""Tests for ONNX export and quantization functionality."""

import json
import os
from pathlib import Path

import pytest

from backend.exporter.onnx_export import export_to_onnx, get_onnx_metadata
from backend.exporter.quantize import quantize_onnx, get_quantized_metadata
from backend.exporter.builder import build


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


class TestExportToOnnx:
    """Tests for the export_to_onnx function."""

    def test_export_creates_onnx_file(self, tmp_path, model_stub_file):
        """Test that export_to_onnx creates the ONNX file."""
        output_dir = tmp_path / "onnx_out"
        onnx_path = export_to_onnx(str(model_stub_file), str(output_dir))

        assert Path(onnx_path).exists()
        assert onnx_path.endswith("model.onnx")

    def test_export_creates_output_directory(self, tmp_path, model_stub_file):
        """Test that export_to_onnx creates the output directory."""
        output_dir = tmp_path / "new_dir" / "onnx_out"
        export_to_onnx(str(model_stub_file), str(output_dir))

        assert output_dir.exists()

    def test_export_creates_metadata_file(self, tmp_path, model_stub_file):
        """Test that export_to_onnx creates metadata file."""
        output_dir = tmp_path / "onnx_out"
        export_to_onnx(str(model_stub_file), str(output_dir))

        meta_path = output_dir / "model.onnx.meta.json"
        assert meta_path.exists()

    def test_export_metadata_contains_exported_from(self, tmp_path, model_stub_file):
        """Test that metadata contains exported_from field."""
        output_dir = tmp_path / "onnx_out"
        onnx_path = export_to_onnx(str(model_stub_file), str(output_dir))

        metadata = get_onnx_metadata(onnx_path)
        assert metadata is not None
        assert "exported_from" in metadata
        assert str(model_stub_file) in metadata["exported_from"]

    def test_export_metadata_contains_export_time(self, tmp_path, model_stub_file):
        """Test that metadata contains export_time field."""
        output_dir = tmp_path / "onnx_out"
        onnx_path = export_to_onnx(str(model_stub_file), str(output_dir))

        metadata = get_onnx_metadata(onnx_path)
        assert "export_time" in metadata

    def test_export_metadata_contains_version(self, tmp_path, model_stub_file):
        """Test that metadata contains version field."""
        output_dir = tmp_path / "onnx_out"
        onnx_path = export_to_onnx(str(model_stub_file), str(output_dir))

        metadata = get_onnx_metadata(onnx_path)
        assert metadata["version"] == "0.1-stub"

    def test_export_onnx_file_has_placeholder_comment(self, tmp_path, model_stub_file):
        """Test that ONNX file has the placeholder comment."""
        output_dir = tmp_path / "onnx_out"
        onnx_path = export_to_onnx(str(model_stub_file), str(output_dir))

        with open(onnx_path, "r") as f:
            content = f.read()
        assert "# TinyForgeAI ONNX placeholder" in content

    def test_export_raises_for_missing_model(self, tmp_path):
        """Test that FileNotFoundError is raised for missing model."""
        output_dir = tmp_path / "onnx_out"
        with pytest.raises(FileNotFoundError):
            export_to_onnx("/nonexistent/model.json", str(output_dir))

    def test_export_raises_for_non_json_file(self, tmp_path):
        """Test that ValueError is raised for non-JSON model file."""
        # Create a non-JSON file
        model_file = tmp_path / "model.bin"
        model_file.write_text("binary data")

        output_dir = tmp_path / "onnx_out"
        with pytest.raises(ValueError, match="Expected JSON"):
            export_to_onnx(str(model_file), str(output_dir))


class TestQuantizeOnnx:
    """Tests for the quantize_onnx function."""

    @pytest.fixture
    def onnx_file(self, tmp_path, model_stub_file):
        """Create an ONNX file for quantization tests."""
        output_dir = tmp_path / "onnx_out"
        onnx_path = export_to_onnx(str(model_stub_file), str(output_dir))
        return onnx_path

    def test_quantize_creates_quantized_file(self, tmp_path, onnx_file):
        """Test that quantize_onnx creates the quantized file."""
        quant_dir = tmp_path / "quant"
        quant_path = quantize_onnx(onnx_file, str(quant_dir))

        assert Path(quant_path).exists()
        assert quant_path.endswith("quantized.onnx")

    def test_quantize_creates_output_directory(self, tmp_path, onnx_file):
        """Test that quantize_onnx creates the output directory."""
        quant_dir = tmp_path / "new_dir" / "quant"
        quantize_onnx(onnx_file, str(quant_dir))

        assert quant_dir.exists()

    def test_quantize_file_has_mode_comment(self, tmp_path, onnx_file):
        """Test that quantized file has the mode comment."""
        quant_dir = tmp_path / "quant"
        quant_path = quantize_onnx(onnx_file, str(quant_dir), mode="int8")

        with open(quant_path, "r") as f:
            content = f.read()
        assert "# quantized: int8" in content

    def test_quantize_metadata_has_quantized_flag(self, tmp_path, onnx_file):
        """Test that metadata has quantized flag set to True."""
        quant_dir = tmp_path / "quant"
        quant_path = quantize_onnx(onnx_file, str(quant_dir))

        metadata = get_quantized_metadata(quant_path)
        assert metadata is not None
        assert metadata["quantized"] is True

    def test_quantize_metadata_has_mode(self, tmp_path, onnx_file):
        """Test that metadata has the quantization mode."""
        quant_dir = tmp_path / "quant"
        quant_path = quantize_onnx(onnx_file, str(quant_dir), mode="fp16")

        metadata = get_quantized_metadata(quant_path)
        assert metadata["mode"] == "fp16"

    def test_quantize_preserves_original_metadata(self, tmp_path, onnx_file):
        """Test that original ONNX metadata is preserved."""
        quant_dir = tmp_path / "quant"
        quant_path = quantize_onnx(onnx_file, str(quant_dir))

        metadata = get_quantized_metadata(quant_path)
        # Should have exported_from from original
        assert "exported_from" in metadata

    def test_quantize_raises_for_missing_onnx(self, tmp_path):
        """Test that FileNotFoundError is raised for missing ONNX file."""
        quant_dir = tmp_path / "quant"
        with pytest.raises(FileNotFoundError):
            quantize_onnx("/nonexistent/model.onnx", str(quant_dir))

    def test_quantize_raises_for_invalid_mode(self, tmp_path, onnx_file):
        """Test that ValueError is raised for invalid mode."""
        quant_dir = tmp_path / "quant"
        with pytest.raises(ValueError, match="Invalid quantization mode"):
            quantize_onnx(onnx_file, str(quant_dir), mode="invalid")

    def test_quantize_accepts_valid_modes(self, tmp_path, onnx_file):
        """Test that all valid modes are accepted."""
        valid_modes = ["int8", "fp16", "uint8", "int4"]
        for mode in valid_modes:
            quant_dir = tmp_path / f"quant_{mode}"
            quant_path = quantize_onnx(onnx_file, str(quant_dir), mode=mode)
            assert Path(quant_path).exists()


class TestBuilderOnnxExport:
    """Tests for builder integration with ONNX export."""

    def test_build_with_export_onnx_creates_report(self, tmp_path):
        """Test that build with export_onnx creates export_report.json."""
        # Create a model stub JSON
        model_stub = tmp_path / "model_stub.json"
        with open(model_stub, "w") as f:
            json.dump({"model_type": "stub", "n_records": 1}, f)

        output_dir = tmp_path / "output_service"
        build(
            model_path=str(model_stub),
            output_dir=str(output_dir),
            export_onnx=True,
        )

        report_path = output_dir / "export_report.json"
        assert report_path.exists()

    def test_build_with_export_onnx_report_has_paths(self, tmp_path):
        """Test that export report contains onnx_path and quantized_path."""
        model_stub = tmp_path / "model_stub.json"
        with open(model_stub, "w") as f:
            json.dump({"model_type": "stub", "n_records": 1}, f)

        output_dir = tmp_path / "output_service"
        build(
            model_path=str(model_stub),
            output_dir=str(output_dir),
            export_onnx=True,
        )

        report_path = output_dir / "export_report.json"
        with open(report_path, "r") as f:
            report = json.load(f)

        assert "onnx_path" in report
        assert "quantized_path" in report
        assert "export_time" in report

    def test_build_with_export_onnx_creates_onnx_dir(self, tmp_path):
        """Test that build creates onnx directory with model files."""
        model_stub = tmp_path / "model_stub.json"
        with open(model_stub, "w") as f:
            json.dump({"model_type": "stub", "n_records": 1}, f)

        output_dir = tmp_path / "output_service"
        build(
            model_path=str(model_stub),
            output_dir=str(output_dir),
            export_onnx=True,
        )

        onnx_dir = output_dir / "onnx"
        assert onnx_dir.exists()
        assert (onnx_dir / "model.onnx").exists()
        assert (onnx_dir / "quant" / "quantized.onnx").exists()

    def test_build_without_export_onnx_no_report(self, tmp_path):
        """Test that build without export_onnx doesn't create report."""
        model_stub = tmp_path / "model_stub.json"
        with open(model_stub, "w") as f:
            json.dump({"model_type": "stub", "n_records": 1}, f)

        output_dir = tmp_path / "output_service"
        build(
            model_path=str(model_stub),
            output_dir=str(output_dir),
            export_onnx=False,
        )

        report_path = output_dir / "export_report.json"
        assert not report_path.exists()

    def test_build_returns_export_report(self, tmp_path):
        """Test that build returns the export report dict."""
        model_stub = tmp_path / "model_stub.json"
        with open(model_stub, "w") as f:
            json.dump({"model_type": "stub", "n_records": 1}, f)

        output_dir = tmp_path / "output_service"
        report = build(
            model_path=str(model_stub),
            output_dir=str(output_dir),
            export_onnx=True,
        )

        assert report is not None
        assert "onnx_path" in report
        assert "quantized_path" in report

    def test_build_returns_none_without_export(self, tmp_path):
        """Test that build returns None when export_onnx is False."""
        model_stub = tmp_path / "model_stub.json"
        with open(model_stub, "w") as f:
            json.dump({"model_type": "stub", "n_records": 1}, f)

        output_dir = tmp_path / "output_service"
        report = build(
            model_path=str(model_stub),
            output_dir=str(output_dir),
            export_onnx=False,
        )

        assert report is None
