"""Tests for Notion connector."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from connectors.notion_connector import (
    NotionConnector,
    NotionConfig,
    NotionPage,
    NotionDatabase,
)


class TestNotionConfig:
    """Tests for NotionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NotionConfig()
        assert config.mock_mode is True
        assert config.page_size == 100
        assert config.api_version == "2022-06-28"
        assert config.include_children is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = NotionConfig(
            mock_mode=False,
            api_token="secret_token",
            page_size=50,
        )
        assert config.mock_mode is False
        assert config.api_token == "secret_token"
        assert config.page_size == 50


class TestNotionPage:
    """Tests for NotionPage dataclass."""

    def test_page_creation(self):
        """Test NotionPage creation."""
        page = NotionPage(
            id="page-123",
            title="Test Page",
            url="https://notion.so/test",
        )
        assert page.id == "page-123"
        assert page.title == "Test Page"
        assert page.url == "https://notion.so/test"
        assert page.archived is False

    def test_page_to_dict(self):
        """Test NotionPage to_dict method."""
        page = NotionPage(
            id="page-123",
            title="Test Page",
        )
        data = page.to_dict()
        assert data["id"] == "page-123"
        assert data["title"] == "Test Page"
        assert "properties" in data


class TestNotionDatabase:
    """Tests for NotionDatabase dataclass."""

    def test_database_creation(self):
        """Test NotionDatabase creation."""
        db = NotionDatabase(
            id="db-123",
            title="Test Database",
        )
        assert db.id == "db-123"
        assert db.title == "Test Database"

    def test_database_to_dict(self):
        """Test NotionDatabase to_dict method."""
        db = NotionDatabase(
            id="db-123",
            title="Test Database",
            properties={"Name": {"type": "title"}},
        )
        data = db.to_dict()
        assert data["id"] == "db-123"
        assert "properties" in data


class TestNotionConnectorMock:
    """Tests for NotionConnector in mock mode."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create connector with temp samples directory."""
        samples_dir = tmp_path / "samples"
        samples_dir.mkdir()

        # Create sample database file
        db_data = {
            "title": "Test Database",
            "properties": {
                "Question": {"type": "title"},
                "Answer": {"type": "rich_text"},
            },
            "pages": [
                {
                    "id": "page-001",
                    "title": "What is AI?",
                    "properties": {
                        "Question": {
                            "type": "title",
                            "title": [{"plain_text": "What is AI?"}],
                        },
                        "Answer": {
                            "type": "rich_text",
                            "rich_text": [{"plain_text": "Artificial Intelligence"}],
                        },
                    },
                    "content": "AI is the simulation of human intelligence.",
                },
                {
                    "id": "page-002",
                    "title": "What is ML?",
                    "properties": {
                        "Question": {
                            "type": "title",
                            "title": [{"plain_text": "What is ML?"}],
                        },
                        "Answer": {
                            "type": "rich_text",
                            "rich_text": [{"plain_text": "Machine Learning"}],
                        },
                    },
                },
            ],
        }
        (samples_dir / "test_db.json").write_text(json.dumps(db_data))

        config = NotionConfig(
            mock_mode=True,
            samples_dir=str(samples_dir),
        )
        return NotionConnector(config)

    def test_list_pages_mock(self, connector):
        """Test listing pages in mock mode."""
        pages = connector.list_pages(database_id="test_db")
        assert len(pages) == 2
        assert all(isinstance(p, NotionPage) for p in pages)
        assert pages[0].title == "What is AI?"

    def test_get_page_content_mock(self, connector, tmp_path):
        """Test getting page content in mock mode."""
        # Create a dedicated page file for content retrieval
        samples_dir = Path(connector.config.samples_dir)
        page_file = samples_dir / "page-001.json"
        page_file.write_text(json.dumps({
            "id": "page-001",
            "content": "AI is the simulation of human intelligence.",
        }))

        content = connector.get_page_content("page-001")
        assert "AI is the simulation" in content

    def test_get_page_content_not_found(self, connector):
        """Test getting non-existent page raises error."""
        with pytest.raises(FileNotFoundError):
            connector.get_page_content("nonexistent_page")

    def test_get_database_mock(self, connector):
        """Test getting database metadata in mock mode."""
        db = connector.get_database("test_db")
        assert isinstance(db, NotionDatabase)
        assert db.title == "Test Database"

    def test_stream_samples(self, connector):
        """Test streaming samples from database."""
        mapping = {"input": "Question", "output": "Answer"}
        samples = list(connector.stream_samples("test_db", mapping))

        assert len(samples) == 2
        assert samples[0]["metadata"]["source"] == "notion"
        assert samples[0]["metadata"]["page_id"] == "page-001"


class TestNotionConnectorEnvironment:
    """Tests for environment variable handling."""

    def test_mock_env_true(self):
        """Test NOTION_MOCK=true enables mock mode."""
        with patch.dict(os.environ, {"NOTION_MOCK": "true"}):
            config = NotionConfig(mock_mode=False)
            connector = NotionConnector(config)
            assert connector.config.mock_mode is True

    def test_mock_env_false(self):
        """Test NOTION_MOCK=false disables mock mode."""
        with patch.dict(os.environ, {"NOTION_MOCK": "false"}):
            config = NotionConfig(mock_mode=True)
            connector = NotionConnector(config)
            assert connector.config.mock_mode is False

    def test_api_token_from_env(self):
        """Test API token loaded from environment."""
        with patch.dict(os.environ, {"NOTION_API_TOKEN": "secret_test_token"}):
            config = NotionConfig()
            connector = NotionConnector(config)
            assert connector.config.api_token == "secret_test_token"

    def test_global_mock_env(self):
        """Test CONNECTOR_MOCK=true enables mock mode."""
        with patch.dict(os.environ, {"CONNECTOR_MOCK": "true"}):
            config = NotionConfig(mock_mode=False)
            connector = NotionConnector(config)
            assert connector.config.mock_mode is True


class TestNotionConnectorPropertyExtraction:
    """Tests for property value extraction."""

    def test_extract_title_property(self):
        """Test extracting title property."""
        connector = NotionConnector()
        properties = {
            "Name": {
                "type": "title",
                "title": [{"plain_text": "Test Title"}],
            }
        }
        value = connector._extract_property_value(properties, "Name")
        assert value == "Test Title"

    def test_extract_rich_text_property(self):
        """Test extracting rich_text property."""
        connector = NotionConnector()
        properties = {
            "Description": {
                "type": "rich_text",
                "rich_text": [{"plain_text": "Test description"}],
            }
        }
        value = connector._extract_property_value(properties, "Description")
        assert value == "Test description"

    def test_extract_select_property(self):
        """Test extracting select property."""
        connector = NotionConnector()
        properties = {
            "Status": {
                "type": "select",
                "select": {"name": "Done"},
            }
        }
        value = connector._extract_property_value(properties, "Status")
        assert value == "Done"

    def test_extract_multi_select_property(self):
        """Test extracting multi_select property."""
        connector = NotionConnector()
        properties = {
            "Tags": {
                "type": "multi_select",
                "multi_select": [{"name": "AI"}, {"name": "ML"}],
            }
        }
        value = connector._extract_property_value(properties, "Tags")
        assert value == "AI, ML"

    def test_extract_number_property(self):
        """Test extracting number property."""
        connector = NotionConnector()
        properties = {
            "Count": {
                "type": "number",
                "number": 42,
            }
        }
        value = connector._extract_property_value(properties, "Count")
        assert value == "42"

    def test_extract_checkbox_property(self):
        """Test extracting checkbox property."""
        connector = NotionConnector()
        properties = {
            "Done": {
                "type": "checkbox",
                "checkbox": True,
            }
        }
        value = connector._extract_property_value(properties, "Done")
        assert value == "True"

    def test_extract_nonexistent_property(self):
        """Test extracting non-existent property returns None."""
        connector = NotionConnector()
        properties = {}
        value = connector._extract_property_value(properties, "Missing")
        assert value is None


class TestNotionConnectorBlockExtraction:
    """Tests for block text extraction."""

    def test_extract_paragraph_block(self):
        """Test extracting text from paragraph block."""
        connector = NotionConnector()
        block = {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"plain_text": "Test paragraph"}],
            },
        }
        text = connector._extract_block_text(block)
        assert text == "Test paragraph"

    def test_extract_heading_block(self):
        """Test extracting text from heading block."""
        connector = NotionConnector()
        block = {
            "type": "heading_1",
            "heading_1": {
                "rich_text": [{"plain_text": "Main Title"}],
            },
        }
        text = connector._extract_block_text(block)
        assert text == "# Main Title"

    def test_extract_code_block(self):
        """Test extracting text from code block."""
        connector = NotionConnector()
        block = {
            "type": "code",
            "code": {
                "rich_text": [{"plain_text": "print('hello')"}],
                "language": "python",
            },
        }
        text = connector._extract_block_text(block)
        assert "```python" in text
        assert "print('hello')" in text

    def test_extract_bulleted_list_block(self):
        """Test extracting text from bulleted list block."""
        connector = NotionConnector()
        block = {
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"plain_text": "List item"}],
            },
        }
        text = connector._extract_block_text(block)
        assert text == "- List item"


class TestNotionConnectorReal:
    """Tests for real API behavior (mocked)."""

    def test_get_session_without_token(self):
        """Test that missing token raises error."""
        config = NotionConfig(mock_mode=False, api_token=None)
        connector = NotionConnector(config)
        connector.config.api_token = None  # Ensure no token

        with pytest.raises(RuntimeError):
            connector._get_session()

    def test_list_pages_requires_database_id(self):
        """Test that real API requires database_id."""
        config = NotionConfig(mock_mode=False, api_token="test")
        connector = NotionConnector(config)

        # Mock session to avoid actual API calls
        connector._session = MagicMock()

        with pytest.raises(ValueError, match="database_id is required"):
            connector._list_pages_real(database_id=None)


class TestNotionConnectorIntegration:
    """Integration tests using real sample files."""

    @pytest.fixture
    def connector_with_examples(self):
        """Create connector using examples directory if it exists."""
        examples_dir = Path(__file__).parent.parent / "examples" / "notion_samples"
        if not examples_dir.exists():
            pytest.skip("Examples directory not found")

        config = NotionConfig(
            mock_mode=True,
            samples_dir=str(examples_dir),
        )
        return NotionConnector(config)

    def test_list_example_pages(self, connector_with_examples):
        """Test listing pages from examples directory."""
        pages = connector_with_examples.list_pages(database_id="training_database")
        assert len(pages) >= 1

    def test_stream_example_samples(self, connector_with_examples):
        """Test streaming samples from examples directory."""
        mapping = {"input": "Question", "output": "Answer"}
        samples = list(connector_with_examples.stream_samples("training_database", mapping))
        assert len(samples) >= 1
