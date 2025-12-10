"""Tests for Google Drive connector."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from connectors.google_drive_connector import (
    GoogleDriveConnector,
    GoogleDriveConfig,
    DriveFile,
)


class TestGoogleDriveConfig:
    """Tests for GoogleDriveConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GoogleDriveConfig()
        assert config.mock_mode is True
        assert config.page_size == 100
        assert config.include_trashed is False
        assert "text/plain" in config.mime_types

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GoogleDriveConfig(
            mock_mode=False,
            page_size=50,
            service_account_file="/path/to/service.json",
        )
        assert config.mock_mode is False
        assert config.page_size == 50
        assert config.service_account_file == "/path/to/service.json"


class TestDriveFile:
    """Tests for DriveFile dataclass."""

    def test_drive_file_creation(self):
        """Test DriveFile creation."""
        file = DriveFile(
            id="file123",
            name="test.txt",
            mime_type="text/plain",
            size=1024,
        )
        assert file.id == "file123"
        assert file.name == "test.txt"
        assert file.mime_type == "text/plain"
        assert file.size == 1024

    def test_drive_file_to_dict(self):
        """Test DriveFile to_dict method."""
        file = DriveFile(
            id="file123",
            name="test.txt",
            mime_type="text/plain",
        )
        data = file.to_dict()
        assert data["id"] == "file123"
        assert data["name"] == "test.txt"
        assert "mime_type" in data


class TestGoogleDriveConnectorMock:
    """Tests for GoogleDriveConnector in mock mode."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create connector with temp samples directory."""
        samples_dir = tmp_path / "samples"
        samples_dir.mkdir()

        # Create sample files
        (samples_dir / "test_data.jsonl").write_text(
            '{"input": "Hello", "output": "World"}\n'
            '{"input": "Foo", "output": "Bar"}\n'
        )
        (samples_dir / "qa.json").write_text(
            json.dumps([
                {"question": "What is AI?", "answer": "Artificial Intelligence"},
            ])
        )

        config = GoogleDriveConfig(
            mock_mode=True,
            samples_dir=str(samples_dir),
        )
        return GoogleDriveConnector(config)

    def test_list_files_mock(self, connector):
        """Test listing files in mock mode."""
        files = connector.list_files()
        assert len(files) >= 1
        assert all(isinstance(f, DriveFile) for f in files)

    def test_get_file_content_mock(self, connector, tmp_path):
        """Test getting file content in mock mode."""
        # Create a test file
        samples_dir = Path(connector.config.samples_dir)
        test_file = samples_dir / "content_test.txt"
        test_file.write_text("Test content here")

        content = connector.get_file_content("content_test")
        assert content == "Test content here"

    def test_get_file_content_not_found(self, connector):
        """Test getting non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            connector.get_file_content("nonexistent_file_id")

    def test_stream_samples_jsonl(self, connector, tmp_path):
        """Test streaming samples from JSONL file."""
        # The connector needs to be able to read the file by ID
        # In mock mode, get_file_content looks for file by stem, so we test differently
        mapping = {"input": "input", "output": "output"}

        # List files first to see what we have
        files = connector.list_files()
        jsonl_files = [f for f in files if f.name.endswith(".jsonl")]

        # Verify we have a jsonl file
        assert len(jsonl_files) >= 1

        # Now test streaming - files are listed and content retrieved
        samples = list(connector.stream_samples("", mapping))
        # Should have samples (at least from jsonl file)
        assert len(samples) >= 2

    def test_stream_samples_json(self, connector):
        """Test streaming samples from JSON file."""
        mapping = {"input": "question", "output": "answer"}
        samples = list(connector.stream_samples("", mapping))

        # Should get samples from qa.json
        json_samples = [s for s in samples if s.get("metadata", {}).get("source_file") == "qa.json"]
        assert len(json_samples) >= 1


class TestGoogleDriveConnectorEnvironment:
    """Tests for environment variable handling."""

    def test_mock_env_true(self):
        """Test GOOGLE_DRIVE_MOCK=true enables mock mode."""
        with patch.dict(os.environ, {"GOOGLE_DRIVE_MOCK": "true"}):
            config = GoogleDriveConfig(mock_mode=False)
            connector = GoogleDriveConnector(config)
            assert connector.config.mock_mode is True

    def test_mock_env_false(self):
        """Test GOOGLE_DRIVE_MOCK=false disables mock mode."""
        with patch.dict(os.environ, {"GOOGLE_DRIVE_MOCK": "false"}):
            config = GoogleDriveConfig(mock_mode=True)
            connector = GoogleDriveConnector(config)
            assert connector.config.mock_mode is False

    def test_global_mock_env(self):
        """Test CONNECTOR_MOCK=true enables mock mode."""
        with patch.dict(os.environ, {"CONNECTOR_MOCK": "true"}):
            config = GoogleDriveConfig(mock_mode=False)
            connector = GoogleDriveConnector(config)
            assert connector.config.mock_mode is True


class TestGoogleDriveConnectorReal:
    """Tests for real API behavior (mocked)."""

    def test_get_service_without_credentials(self):
        """Test that missing credentials raises error."""
        config = GoogleDriveConfig(mock_mode=False)
        connector = GoogleDriveConnector(config)

        with pytest.raises((ImportError, RuntimeError)):
            connector._get_service()

    def test_list_files_real_returns_results(self):
        """Test that real API call returns DriveFile objects."""
        # Mock the Google API client
        mock_service = MagicMock()
        mock_files = MagicMock()
        mock_service.files.return_value = mock_files
        mock_files.list.return_value.execute.return_value = {
            "files": [
                {
                    "id": "file1",
                    "name": "test.txt",
                    "mimeType": "text/plain",
                }
            ],
            "nextPageToken": None,
        }

        config = GoogleDriveConfig(mock_mode=False)
        connector = GoogleDriveConnector(config)
        connector._service = mock_service

        files = connector._list_files_real()
        assert len(files) == 1
        assert files[0].id == "file1"
        assert files[0].name == "test.txt"


class TestGoogleDriveConnectorMimeTypes:
    """Tests for MIME type handling."""

    def test_guess_mime_type(self):
        """Test MIME type guessing from extension."""
        connector = GoogleDriveConnector()

        assert connector._guess_mime_type(".txt") == "text/plain"
        assert connector._guess_mime_type(".json") == "application/json"
        assert connector._guess_mime_type(".md") == "text/markdown"
        assert connector._guess_mime_type(".csv") == "text/csv"
        assert connector._guess_mime_type(".unknown") == "application/octet-stream"


class TestGoogleDriveConnectorIntegration:
    """Integration tests using real sample files."""

    @pytest.fixture
    def connector_with_examples(self):
        """Create connector using examples directory if it exists."""
        examples_dir = Path(__file__).parent.parent / "examples" / "google_drive_samples"
        if not examples_dir.exists():
            pytest.skip("Examples directory not found")

        config = GoogleDriveConfig(
            mock_mode=True,
            samples_dir=str(examples_dir),
        )
        return GoogleDriveConnector(config)

    def test_list_example_files(self, connector_with_examples):
        """Test listing files from examples directory."""
        files = connector_with_examples.list_files()
        assert len(files) >= 1

    def test_stream_example_samples(self, connector_with_examples):
        """Test streaming samples from examples directory."""
        mapping = {"input": "input", "output": "output"}
        samples = list(connector_with_examples.stream_samples("", mapping))
        assert len(samples) >= 1
