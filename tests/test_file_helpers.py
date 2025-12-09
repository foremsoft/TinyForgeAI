"""Tests for file ingestion helper functions."""

from pathlib import Path

import pytest

from connectors.file_ingest import (
    _ingest_md,
    _ingest_txt,
)


class TestIngestTxtHelper:
    """Tests for the _ingest_txt helper function."""

    def test_reads_file_content(self, tmp_path):
        """Test that _ingest_txt reads file content correctly."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = _ingest_txt(test_file)
        assert result == "Hello, World!"

    def test_handles_multiline_content(self, tmp_path):
        """Test that multiline content is preserved."""
        test_file = tmp_path / "test.txt"
        content = "Line 1\nLine 2\nLine 3"
        test_file.write_text(content)

        result = _ingest_txt(test_file)
        assert result == content

    def test_handles_empty_file(self, tmp_path):
        """Test that empty files return empty string."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        result = _ingest_txt(test_file)
        assert result == ""

    def test_handles_unicode_content(self, tmp_path):
        """Test that unicode content is handled correctly."""
        test_file = tmp_path / "unicode.txt"
        content = "Hello 世界 🌍 Привет"
        test_file.write_text(content, encoding="utf-8")

        result = _ingest_txt(test_file)
        assert result == content

    def test_respects_custom_encoding(self, tmp_path):
        """Test that custom encoding is respected."""
        test_file = tmp_path / "latin1.txt"
        # Write bytes directly with latin-1 encoding
        test_file.write_bytes("Café résumé".encode("latin-1"))

        result = _ingest_txt(test_file, encoding="latin-1")
        assert "Café" in result
        assert "résumé" in result


class TestIngestMdHelper:
    """Tests for the _ingest_md helper function."""

    def test_reads_markdown_content(self, tmp_path):
        """Test that _ingest_md reads markdown content correctly."""
        test_file = tmp_path / "test.md"
        content = "# Heading\n\nParagraph text."
        test_file.write_text(content)

        result = _ingest_md(test_file)
        assert result == content

    def test_preserves_markdown_syntax(self, tmp_path):
        """Test that markdown syntax is preserved, not rendered."""
        test_file = tmp_path / "test.md"
        content = "**bold** and *italic* and `code`"
        test_file.write_text(content)

        result = _ingest_md(test_file)
        assert "**bold**" in result
        assert "*italic*" in result
        assert "`code`" in result

    def test_preserves_lists(self, tmp_path):
        """Test that list syntax is preserved."""
        test_file = tmp_path / "test.md"
        content = "- Item 1\n- Item 2\n- Item 3"
        test_file.write_text(content)

        result = _ingest_md(test_file)
        assert "- Item 1" in result
        assert "- Item 2" in result
        assert "- Item 3" in result

    def test_preserves_code_blocks(self, tmp_path):
        """Test that code blocks are preserved."""
        test_file = tmp_path / "test.md"
        content = "```python\nprint('hello')\n```"
        test_file.write_text(content)

        result = _ingest_md(test_file)
        assert "```python" in result
        assert "print('hello')" in result

    def test_handles_empty_file(self, tmp_path):
        """Test that empty files return empty string."""
        test_file = tmp_path / "empty.md"
        test_file.write_text("")

        result = _ingest_md(test_file)
        assert result == ""

    def test_respects_custom_encoding(self, tmp_path):
        """Test that custom encoding is respected."""
        test_file = tmp_path / "latin1.md"
        test_file.write_bytes("# Résumé\n\nCafé notes".encode("latin-1"))

        result = _ingest_md(test_file, encoding="latin-1")
        assert "Résumé" in result
        assert "Café" in result


class TestPathHandling:
    """Tests for path handling in helper functions."""

    def test_accepts_path_object(self, tmp_path):
        """Test that Path objects are accepted."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Should work with Path object
        result = _ingest_txt(test_file)
        assert result == "content"

    def test_handles_nested_directories(self, tmp_path):
        """Test that files in nested directories are handled."""
        nested_dir = tmp_path / "a" / "b" / "c"
        nested_dir.mkdir(parents=True)
        test_file = nested_dir / "test.txt"
        test_file.write_text("nested content")

        result = _ingest_txt(test_file)
        assert result == "nested content"

    def test_handles_spaces_in_path(self, tmp_path):
        """Test that paths with spaces are handled."""
        spaced_dir = tmp_path / "path with spaces"
        spaced_dir.mkdir()
        test_file = spaced_dir / "file with spaces.txt"
        test_file.write_text("spaced content")

        result = _ingest_txt(test_file)
        assert result == "spaced content"
