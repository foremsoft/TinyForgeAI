"""Tests for the Google Docs connector."""

import json
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from connectors.google_docs_connector import (
    GoogleDocsConnector,
    GoogleDocsConfig,
    GoogleDoc,
    fetch_doc_text,
    list_docs_in_folder,
    GOOGLE_API_AVAILABLE,
)


@pytest.fixture(autouse=True)
def ensure_mock_mode(monkeypatch):
    """Ensure tests run in mock mode."""
    monkeypatch.setenv("GOOGLE_OAUTH_DISABLED", "true")
    monkeypatch.setenv("GOOGLE_DOCS_MOCK", "true")


# =============================================================================
# Legacy Function Interface Tests
# =============================================================================

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


def test_fetch_doc_text_normalizes_whitespace():
    """Test that fetched text has normalized whitespace."""
    text = fetch_doc_text("sample_doc1")
    # Should not have excessive whitespace
    assert "  " not in text or "\n\n\n" not in text


# =============================================================================
# GoogleDocsConnector Class Tests
# =============================================================================

class TestGoogleDocsConnector:
    """Tests for the GoogleDocsConnector class."""

    def test_init_default_config(self):
        """Test connector initializes with default config."""
        connector = GoogleDocsConnector()
        assert connector.config is not None
        assert connector.config.mock_mode is True

    def test_init_custom_config(self):
        """Test connector initializes with custom config."""
        config = GoogleDocsConfig(
            mock_mode=True,
            include_headers=False,
            paragraph_separator="\n",
        )
        connector = GoogleDocsConnector(config)
        assert connector.config.include_headers is False
        assert connector.config.paragraph_separator == "\n"

    def test_mock_mode_from_env_variable(self, monkeypatch):
        """Test mock mode is set from environment variable."""
        monkeypatch.setenv("GOOGLE_DOCS_MOCK", "false")
        monkeypatch.delenv("GOOGLE_OAUTH_DISABLED", raising=False)
        config = GoogleDocsConfig(mock_mode=False)
        connector = GoogleDocsConnector(config)
        # Note: Without credentials, this would fail in real mode
        # but the mock_mode should be False based on env
        assert connector.config.mock_mode is False

    def test_fetch_doc_text_mock_mode(self):
        """Test fetching document in mock mode."""
        connector = GoogleDocsConnector()
        text = connector.fetch_doc_text("sample_doc1")
        assert isinstance(text, str)
        assert len(text) > 0
        assert "TinyForgeAI" in text

    def test_fetch_doc_text_file_not_found(self):
        """Test FileNotFoundError for missing document."""
        connector = GoogleDocsConnector()
        with pytest.raises(FileNotFoundError):
            connector.fetch_doc_text("nonexistent_doc_12345")

    def test_list_docs_in_folder_mock_mode(self):
        """Test listing documents in mock mode."""
        connector = GoogleDocsConnector()
        docs = connector.list_docs_in_folder("any_folder")
        assert isinstance(docs, list)
        assert len(docs) >= 1
        assert all(isinstance(doc, GoogleDoc) for doc in docs)

    def test_google_doc_to_dict(self):
        """Test GoogleDoc.to_dict() method."""
        doc = GoogleDoc(
            id="test_id",
            title="Test Title",
            revision_id="rev123",
            created_time="2024-01-01T00:00:00Z",
            modified_time="2024-01-02T00:00:00Z",
        )
        doc_dict = doc.to_dict()
        assert doc_dict["id"] == "test_id"
        assert doc_dict["title"] == "Test Title"
        assert doc_dict["revision_id"] == "rev123"

    def test_get_doc_metadata_mock_mode(self):
        """Test getting document metadata in mock mode."""
        connector = GoogleDocsConnector()
        metadata = connector.get_doc_metadata("sample_doc1")
        assert isinstance(metadata, GoogleDoc)
        assert metadata.id == "sample_doc1"
        assert metadata.title is not None

    def test_get_doc_metadata_not_found(self):
        """Test FileNotFoundError for metadata of missing document."""
        connector = GoogleDocsConnector()
        with pytest.raises(FileNotFoundError):
            connector.get_doc_metadata("nonexistent_doc_12345")


# =============================================================================
# Text Extraction Tests
# =============================================================================

class TestTextExtraction:
    """Tests for text extraction from Google Docs API structures."""

    def test_extract_simple_paragraph(self):
        """Test extraction of a simple paragraph."""
        connector = GoogleDocsConnector()
        document = {
            "title": "Test Doc",
            "body": {
                "content": [
                    {
                        "paragraph": {
                            "elements": [
                                {"textRun": {"content": "Hello, world!"}}
                            ]
                        }
                    }
                ]
            }
        }
        text = connector._extract_text_from_document(document)
        assert "Hello, world!" in text
        assert "Test Doc" in text

    def test_extract_multiple_paragraphs(self):
        """Test extraction of multiple paragraphs."""
        connector = GoogleDocsConnector()
        document = {
            "title": "Test Doc",
            "body": {
                "content": [
                    {
                        "paragraph": {
                            "elements": [
                                {"textRun": {"content": "First paragraph."}}
                            ]
                        }
                    },
                    {
                        "paragraph": {
                            "elements": [
                                {"textRun": {"content": "Second paragraph."}}
                            ]
                        }
                    }
                ]
            }
        }
        text = connector._extract_text_from_document(document)
        assert "First paragraph." in text
        assert "Second paragraph." in text

    def test_extract_with_bullet_list(self):
        """Test extraction of bullet list items."""
        connector = GoogleDocsConnector()
        document = {
            "title": "Test Doc",
            "body": {
                "content": [
                    {
                        "paragraph": {
                            "elements": [
                                {"textRun": {"content": "Item 1"}}
                            ],
                            "bullet": {"nestingLevel": 0}
                        }
                    },
                    {
                        "paragraph": {
                            "elements": [
                                {"textRun": {"content": "Item 2"}}
                            ],
                            "bullet": {"nestingLevel": 0}
                        }
                    }
                ]
            }
        }
        text = connector._extract_text_from_document(document)
        assert "• Item 1" in text
        assert "• Item 2" in text

    def test_extract_nested_bullet_list(self):
        """Test extraction of nested bullet list items."""
        connector = GoogleDocsConnector()
        document = {
            "title": "Test Doc",
            "body": {
                "content": [
                    {
                        "paragraph": {
                            "elements": [
                                {"textRun": {"content": "Parent item"}}
                            ],
                            "bullet": {"nestingLevel": 0}
                        }
                    },
                    {
                        "paragraph": {
                            "elements": [
                                {"textRun": {"content": "Child item"}}
                            ],
                            "bullet": {"nestingLevel": 1}
                        }
                    }
                ]
            }
        }
        text = connector._extract_text_from_document(document)
        assert "• Parent item" in text
        # Nested bullet has indentation in the raw paragraph extraction
        assert "• Child item" in text

    def test_extract_table(self):
        """Test extraction of table content."""
        connector = GoogleDocsConnector()
        document = {
            "title": "Test Doc",
            "body": {
                "content": [
                    {
                        "table": {
                            "tableRows": [
                                {
                                    "tableCells": [
                                        {
                                            "content": [
                                                {
                                                    "paragraph": {
                                                        "elements": [
                                                            {"textRun": {"content": "Header 1"}}
                                                        ]
                                                    }
                                                }
                                            ]
                                        },
                                        {
                                            "content": [
                                                {
                                                    "paragraph": {
                                                        "elements": [
                                                            {"textRun": {"content": "Header 2"}}
                                                        ]
                                                    }
                                                }
                                            ]
                                        }
                                    ]
                                },
                                {
                                    "tableCells": [
                                        {
                                            "content": [
                                                {
                                                    "paragraph": {
                                                        "elements": [
                                                            {"textRun": {"content": "Cell 1"}}
                                                        ]
                                                    }
                                                }
                                            ]
                                        },
                                        {
                                            "content": [
                                                {
                                                    "paragraph": {
                                                        "elements": [
                                                            {"textRun": {"content": "Cell 2"}}
                                                        ]
                                                    }
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
        }
        text = connector._extract_text_from_document(document)
        assert "Header 1" in text
        assert "Header 2" in text
        assert "Cell 1" in text
        assert "|" in text  # Table cells separated by pipe

    def test_extract_without_headers(self):
        """Test extraction with headers disabled."""
        config = GoogleDocsConfig(include_headers=False)
        connector = GoogleDocsConnector(config)
        document = {
            "title": "Should Not Appear",
            "body": {
                "content": [
                    {
                        "paragraph": {
                            "elements": [
                                {"textRun": {"content": "Only this content."}}
                            ]
                        }
                    }
                ]
            }
        }
        text = connector._extract_text_from_document(document)
        assert "Should Not Appear" not in text
        assert "Only this content." in text

    def test_extract_inline_image_placeholder(self):
        """Test that inline images are marked with placeholder."""
        connector = GoogleDocsConnector()
        document = {
            "title": "Test Doc",
            "body": {
                "content": [
                    {
                        "paragraph": {
                            "elements": [
                                {"textRun": {"content": "Text before image "}},
                                {"inlineObjectElement": {"inlineObjectId": "img123"}},
                                {"textRun": {"content": " text after image"}}
                            ]
                        }
                    }
                ]
            }
        }
        text = connector._extract_text_from_document(document)
        assert "[Image]" in text


# =============================================================================
# Streaming Tests
# =============================================================================

class TestStreamingSamples:
    """Tests for streaming training samples."""

    def test_stream_samples_returns_iterator(self):
        """Test that stream_samples returns an iterator."""
        connector = GoogleDocsConnector()
        samples = connector.stream_samples("mock_folder")
        # Should be able to iterate
        sample_list = list(samples)
        assert isinstance(sample_list, list)

    def test_stream_samples_has_required_keys(self):
        """Test that streamed samples have required keys."""
        connector = GoogleDocsConnector()
        for sample in connector.stream_samples("mock_folder"):
            assert "input" in sample
            assert "output" in sample
            assert "metadata" in sample
            assert "source" in sample["metadata"]
            break  # Just check first sample

    def test_stream_samples_chunk_by_document(self):
        """Test streaming with chunk_by='document'."""
        connector = GoogleDocsConnector()
        samples = list(connector.stream_samples("mock_folder", chunk_by="document"))
        # Each document should produce one sample
        assert len(samples) >= 1
        for sample in samples:
            assert sample["metadata"]["chunk_type"] == "document"

    def test_stream_samples_chunk_by_paragraph(self):
        """Test streaming with chunk_by='paragraph'."""
        connector = GoogleDocsConnector()
        samples = list(connector.stream_samples("mock_folder", chunk_by="paragraph"))
        # Should produce multiple samples from paragraphs
        for sample in samples:
            assert sample["metadata"]["chunk_type"] == "paragraph"


# =============================================================================
# Real API Tests (Mocked)
# =============================================================================

class TestRealAPIWithMocks:
    """Tests for real API functionality using mocks."""

    @pytest.mark.skipif(not GOOGLE_API_AVAILABLE, reason="Google API not installed")
    def test_real_mode_requires_credentials(self, monkeypatch):
        """Test that real mode raises error without credentials."""
        monkeypatch.setenv("GOOGLE_DOCS_MOCK", "false")
        monkeypatch.delenv("GOOGLE_OAUTH_DISABLED", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

        config = GoogleDocsConfig(mock_mode=False)
        connector = GoogleDocsConnector(config)

        with pytest.raises(RuntimeError, match="No valid Google credentials"):
            connector.fetch_doc_text("some_doc_id")

    @pytest.mark.skipif(not GOOGLE_API_AVAILABLE, reason="Google API not installed")
    def test_fetch_doc_real_with_mock_service(self, monkeypatch):
        """Test real fetch with mocked Google API service."""
        monkeypatch.setenv("GOOGLE_DOCS_MOCK", "false")
        monkeypatch.delenv("GOOGLE_OAUTH_DISABLED", raising=False)

        config = GoogleDocsConfig(mock_mode=False)
        connector = GoogleDocsConnector(config)

        # Mock the credentials and service
        mock_service = MagicMock()
        mock_service.documents().get().execute.return_value = {
            "title": "Mocked Document",
            "body": {
                "content": [
                    {
                        "paragraph": {
                            "elements": [
                                {"textRun": {"content": "This is mocked content."}}
                            ]
                        }
                    }
                ]
            }
        }
        connector._docs_service = mock_service
        connector._credentials = MagicMock()

        text = connector._fetch_doc_real("fake_doc_id")
        assert "Mocked Document" in text
        assert "This is mocked content" in text

    @pytest.mark.skipif(not GOOGLE_API_AVAILABLE, reason="Google API not installed")
    def test_list_docs_real_with_mock_service(self, monkeypatch):
        """Test real list_docs with mocked Google Drive API service."""
        monkeypatch.setenv("GOOGLE_DOCS_MOCK", "false")
        monkeypatch.delenv("GOOGLE_OAUTH_DISABLED", raising=False)

        config = GoogleDocsConfig(mock_mode=False)
        connector = GoogleDocsConnector(config)

        # Mock the credentials and service
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {
            "files": [
                {
                    "id": "doc1",
                    "name": "Document 1",
                    "createdTime": "2024-01-01T00:00:00Z",
                    "modifiedTime": "2024-01-02T00:00:00Z",
                },
                {
                    "id": "doc2",
                    "name": "Document 2",
                    "createdTime": "2024-01-03T00:00:00Z",
                    "modifiedTime": "2024-01-04T00:00:00Z",
                },
            ],
            "nextPageToken": None,
        }
        connector._drive_service = mock_service
        connector._credentials = MagicMock()

        docs = connector._list_docs_real("fake_folder_id")
        assert len(docs) == 2
        assert docs[0].id == "doc1"
        assert docs[0].title == "Document 1"
        assert docs[1].id == "doc2"


# =============================================================================
# CLI Tests
# =============================================================================

class TestCLI:
    """Tests for the CLI interface."""

    def test_cli_fetch_doc_exits_zero(self):
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
            env={**os.environ, "GOOGLE_OAUTH_DISABLED": "true", "GOOGLE_DOCS_MOCK": "true"},
        )
        assert result.returncode == 0

    def test_cli_fetch_doc_prints_content(self):
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
            env={**os.environ, "GOOGLE_OAUTH_DISABLED": "true", "GOOGLE_DOCS_MOCK": "true"},
        )
        assert "TinyForgeAI" in result.stdout

    def test_cli_missing_doc_exits_nonzero(self):
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
            env={**os.environ, "GOOGLE_OAUTH_DISABLED": "true", "GOOGLE_DOCS_MOCK": "true"},
        )
        assert result.returncode != 0

    def test_cli_list_returns_json(self):
        """Test that --list returns JSON output."""
        result = subprocess.run(
            [
                sys.executable,
                "connectors/google_docs_connector.py",
                "--list",
            ],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env={**os.environ, "GOOGLE_OAUTH_DISABLED": "true", "GOOGLE_DOCS_MOCK": "true"},
        )
        assert result.returncode == 0
        # Each line should be valid JSON
        for line in result.stdout.strip().split("\n"):
            if line:
                doc = json.loads(line)
                assert "id" in doc


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Tests for configuration options."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = GoogleDocsConfig()
        assert config.mock_mode is True
        assert config.include_headers is True
        assert config.include_lists is True
        assert config.include_tables is True
        assert config.paragraph_separator == "\n\n"

    def test_config_custom_samples_dir(self, tmp_path):
        """Test custom samples directory."""
        # Create a temporary sample file
        sample_file = tmp_path / "custom_doc.txt"
        sample_file.write_text("Custom document content")

        config = GoogleDocsConfig(
            mock_mode=True,
            samples_dir=str(tmp_path),
        )
        connector = GoogleDocsConnector(config)

        text = connector.fetch_doc_text("custom_doc")
        assert "Custom document content" in text

    def test_env_variable_precedence(self, monkeypatch):
        """Test that environment variables take precedence."""
        # Set conflicting values
        monkeypatch.setenv("GOOGLE_DOCS_MOCK", "true")
        config = GoogleDocsConfig(mock_mode=False)
        connector = GoogleDocsConnector(config)
        # Environment variable should win
        assert connector.config.mock_mode is True
