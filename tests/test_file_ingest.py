"""Tests for the file ingestion connector."""

import os
from pathlib import Path

import pytest

from connectors.file_ingest import (
    check_dependencies,
    get_supported_formats,
    ingest_file,
)

# Get the examples/files directory
SAMPLES_DIR = Path(__file__).parent.parent / "examples" / "files"


class TestIngestTxt:
    """Tests for TXT file ingestion."""

    def test_ingest_txt_returns_string(self):
        """Test that ingesting a TXT file returns a string."""
        text = ingest_file(str(SAMPLES_DIR / "sample.txt"))
        assert isinstance(text, str)

    def test_ingest_txt_returns_non_empty(self):
        """Test that ingested TXT content is non-empty."""
        text = ingest_file(str(SAMPLES_DIR / "sample.txt"))
        assert len(text) > 0

    def test_ingest_txt_contains_expected_content(self):
        """Test that TXT file contains expected content."""
        text = ingest_file(str(SAMPLES_DIR / "sample.txt"))
        assert "TinyForgeAI" in text

    def test_ingest_txt_with_custom_encoding(self, tmp_path):
        """Test ingesting TXT with custom encoding."""
        # Create a file with latin-1 specific character
        test_file = tmp_path / "latin1.txt"
        test_file.write_bytes("Caf\xe9".encode("latin-1"))

        text = ingest_file(str(test_file), encoding="latin-1")
        assert "Café" in text


class TestIngestMd:
    """Tests for Markdown file ingestion."""

    def test_ingest_md_returns_string(self):
        """Test that ingesting an MD file returns a string."""
        text = ingest_file(str(SAMPLES_DIR / "sample.md"))
        assert isinstance(text, str)

    def test_ingest_md_returns_non_empty(self):
        """Test that ingested MD content is non-empty."""
        text = ingest_file(str(SAMPLES_DIR / "sample.md"))
        assert len(text) > 0

    def test_ingest_md_preserves_markdown_syntax(self):
        """Test that markdown syntax is preserved (not rendered)."""
        text = ingest_file(str(SAMPLES_DIR / "sample.md"))
        # Should contain raw markdown syntax
        assert "#" in text or "**" in text or "-" in text

    def test_ingest_md_contains_heading(self):
        """Test that MD file contains expected heading."""
        text = ingest_file(str(SAMPLES_DIR / "sample.md"))
        assert "Sample Markdown Document" in text or "Markdown" in text


class TestIngestDocx:
    """Tests for DOCX file ingestion."""

    @pytest.fixture
    def docx_available(self):
        """Check if python-docx is available."""
        deps = check_dependencies()
        if not deps["python-docx"]:
            pytest.skip("python-docx not installed")
        return True

    def test_ingest_docx_returns_string(self, docx_available):
        """Test that ingesting a DOCX file returns a string."""
        text = ingest_file(str(SAMPLES_DIR / "sample.docx"))
        assert isinstance(text, str)

    def test_ingest_docx_returns_non_empty(self, docx_available):
        """Test that ingested DOCX content is non-empty."""
        text = ingest_file(str(SAMPLES_DIR / "sample.docx"))
        assert len(text) > 0

    def test_ingest_docx_contains_expected_content(self, docx_available):
        """Test that DOCX file contains expected content."""
        text = ingest_file(str(SAMPLES_DIR / "sample.docx"))
        assert "TinyForgeAI" in text or "sample" in text.lower()

    def test_ingest_docx_raises_without_dependency(self, monkeypatch):
        """Test that RuntimeError is raised when python-docx not available."""
        # Monkeypatch the availability flag
        import connectors.file_ingest as module

        monkeypatch.setattr(module, "_DOCX_AVAILABLE", False)

        with pytest.raises(RuntimeError, match="python-docx"):
            ingest_file(str(SAMPLES_DIR / "sample.docx"))


class TestIngestPdf:
    """Tests for PDF file ingestion."""

    @pytest.fixture
    def pdf_available(self):
        """Check if a PDF library is available."""
        deps = check_dependencies()
        if not (deps["PyMuPDF"] or deps["pdfminer.six"]):
            pytest.skip("No PDF library installed")
        return True

    def test_ingest_pdf_returns_string(self, pdf_available):
        """Test that ingesting a PDF file returns a string."""
        text = ingest_file(str(SAMPLES_DIR / "sample.pdf"))
        assert isinstance(text, str)

    def test_ingest_pdf_returns_non_empty(self, pdf_available):
        """Test that ingested PDF content is non-empty."""
        text = ingest_file(str(SAMPLES_DIR / "sample.pdf"))
        assert len(text) > 0

    def test_ingest_pdf_contains_expected_content(self, pdf_available):
        """Test that PDF file contains expected content."""
        text = ingest_file(str(SAMPLES_DIR / "sample.pdf"))
        assert "PDF" in text or "sample" in text.lower() or "TinyForgeAI" in text

    def test_ingest_pdf_raises_without_dependency(self, monkeypatch):
        """Test that RuntimeError is raised when no PDF library available."""
        import connectors.file_ingest as module

        monkeypatch.setattr(module, "_PYMUPDF_AVAILABLE", False)
        monkeypatch.setattr(module, "_PDFMINER_AVAILABLE", False)

        with pytest.raises(RuntimeError, match="PyMuPDF|pdfminer"):
            ingest_file(str(SAMPLES_DIR / "sample.pdf"))


class TestIngestErrors:
    """Tests for error handling in file ingestion."""

    def test_file_not_found_raises(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            ingest_file("nonexistent_file.txt")

    def test_unsupported_format_raises(self, tmp_path):
        """Test that ValueError is raised for unsupported formats."""
        # Create a file with unsupported extension
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            ingest_file(str(test_file))

    def test_unsupported_format_error_lists_supported(self, tmp_path):
        """Test that error message lists supported formats."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")

        with pytest.raises(ValueError, match=r"\.txt.*\.md.*\.docx.*\.pdf"):
            ingest_file(str(test_file))


class TestSupportedFormats:
    """Tests for get_supported_formats function."""

    def test_returns_dict(self):
        """Test that get_supported_formats returns a dict."""
        formats = get_supported_formats()
        assert isinstance(formats, dict)

    def test_contains_all_formats(self):
        """Test that all expected formats are present."""
        formats = get_supported_formats()
        assert ".txt" in formats
        assert ".md" in formats
        assert ".docx" in formats
        assert ".pdf" in formats

    def test_txt_always_available(self):
        """Test that TXT format is always available."""
        formats = get_supported_formats()
        assert formats[".txt"]["available"] is True

    def test_md_always_available(self):
        """Test that MD format is always available."""
        formats = get_supported_formats()
        assert formats[".md"]["available"] is True

    def test_format_entries_have_required_keys(self):
        """Test that each format entry has required keys."""
        formats = get_supported_formats()
        for ext, info in formats.items():
            assert "available" in info
            assert "requires" in info


class TestCheckDependencies:
    """Tests for check_dependencies function."""

    def test_returns_dict(self):
        """Test that check_dependencies returns a dict."""
        deps = check_dependencies()
        assert isinstance(deps, dict)

    def test_contains_all_dependencies(self):
        """Test that all expected dependencies are present."""
        deps = check_dependencies()
        assert "python-docx" in deps
        assert "PyMuPDF" in deps
        assert "pdfminer.six" in deps

    def test_values_are_booleans(self):
        """Test that all values are booleans."""
        deps = check_dependencies()
        for name, available in deps.items():
            assert isinstance(available, bool), f"{name} should be boolean"
