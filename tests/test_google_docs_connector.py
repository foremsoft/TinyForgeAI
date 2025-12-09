"""Tests for the Google Docs connector."""

import os
import subprocess
import sys

import pytest

from connectors.google_docs_connector import fetch_doc_text, list_docs_in_folder


@pytest.fixture(autouse=True)
def ensure_mock_mode(monkeypatch):
    """Ensure tests run in mock mode."""
    monkeypatch.setenv("GOOGLE_OAUTH_DISABLED", "true")


def test_fetch_doc_text_returns_non_empty_string():
    """Test that fetch_doc_text returns a non-empty string."""
    text = fetch_doc_text("sample_doc1")
    assert isinstance(text, str)
    assert len(text) > 0


def test_fetch_doc_text_contains_expected_content():
    """Test that fetched text contains expected snippet from sample file."""
    text = fetch_doc_text("sample_doc1")
    assert "TinyForgeAI" in text


def test_fetch_doc_text_sample_doc2():
    """Test fetching second sample document."""
    text = fetch_doc_text("sample_doc2")
    assert "FAQ" in text or "Frequently Asked Questions" in text


def test_fetch_doc_text_raises_for_missing_doc():
    """Test that FileNotFoundError is raised for missing document."""
    with pytest.raises(FileNotFoundError, match="not found"):
        fetch_doc_text("nonexistent_document")


def test_list_docs_in_folder_returns_list():
    """Test that list_docs_in_folder returns a list."""
    docs = list_docs_in_folder("any_folder_id")
    assert isinstance(docs, list)


def test_list_docs_in_folder_has_at_least_one_doc():
    """Test that list returns at least one document."""
    docs = list_docs_in_folder("any_folder_id")
    assert len(docs) >= 1


def test_list_docs_in_folder_docs_have_required_keys():
    """Test that each document has 'id' and 'title' keys."""
    docs = list_docs_in_folder("any_folder_id")
    for doc in docs:
        assert "id" in doc
        assert "title" in doc


def test_list_docs_in_folder_contains_sample_doc1():
    """Test that sample_doc1 is in the list."""
    docs = list_docs_in_folder("any_folder_id")
    doc_ids = [doc["id"] for doc in docs]
    assert "sample_doc1" in doc_ids


def test_cli_fetch_doc_exits_zero():
    """Test that CLI exits with code 0 for valid doc."""
    result = subprocess.run(
        [
            sys.executable,
            "connectors/google_docs_connector.py",
            "--doc-id",
            "sample_doc1",
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env={**os.environ, "GOOGLE_OAUTH_DISABLED": "true"},
    )
    assert result.returncode == 0


def test_cli_fetch_doc_prints_content():
    """Test that CLI prints document content to stdout."""
    result = subprocess.run(
        [
            sys.executable,
            "connectors/google_docs_connector.py",
            "--doc-id",
            "sample_doc1",
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env={**os.environ, "GOOGLE_OAUTH_DISABLED": "true"},
    )
    assert "TinyForgeAI" in result.stdout


def test_cli_missing_doc_exits_nonzero():
    """Test that CLI exits with non-zero code for missing doc."""
    result = subprocess.run(
        [
            sys.executable,
            "connectors/google_docs_connector.py",
            "--doc-id",
            "nonexistent_doc",
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env={**os.environ, "GOOGLE_OAUTH_DISABLED": "true"},
    )
    assert result.returncode != 0


def test_fetch_doc_text_normalizes_whitespace():
    """Test that fetched text has normalized whitespace."""
    text = fetch_doc_text("sample_doc1")
    # Should not have excessive whitespace
    assert "  " not in text or "\n\n\n" not in text
