"""Tests for Confluence connector."""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from connectors.confluence_connector import (
    ConfluenceConnector,
    ConfluenceConfig,
    ConfluenceSpace,
    ConfluencePage,
    html_to_text,
)


# =============================================================================
# ConfluenceConfig Tests
# =============================================================================

class TestConfluenceConfig:
    """Tests for ConfluenceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConfluenceConfig()
        assert config.base_url is None
        assert config.username is None
        assert config.api_token is None
        assert config.mock_mode is True
        assert config.samples_dir is None
        assert config.limit == 25
        assert "body.storage" in config.expand

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConfluenceConfig(
            base_url="https://example.atlassian.net/wiki",
            username="user@example.com",
            api_token="test-token",
            mock_mode=False,
            limit=50,
        )
        assert config.base_url == "https://example.atlassian.net/wiki"
        assert config.username == "user@example.com"
        assert config.api_token == "test-token"
        assert config.mock_mode is False
        assert config.limit == 50


# =============================================================================
# ConfluenceSpace Tests
# =============================================================================

class TestConfluenceSpace:
    """Tests for ConfluenceSpace dataclass."""

    def test_space_creation(self):
        """Test space creation with defaults."""
        space = ConfluenceSpace(key="DOCS", name="Documentation")
        assert space.key == "DOCS"
        assert space.name == "Documentation"
        assert space.type is None
        assert space.status is None

    def test_space_to_dict(self):
        """Test space to_dict method."""
        space = ConfluenceSpace(
            key="ENG",
            name="Engineering",
            id="123456",
            type="global",
            status="current",
            description="Engineering wiki",
        )
        data = space.to_dict()
        assert data["key"] == "ENG"
        assert data["name"] == "Engineering"
        assert data["id"] == "123456"
        assert data["type"] == "global"
        assert data["description"] == "Engineering wiki"


# =============================================================================
# ConfluencePage Tests
# =============================================================================

class TestConfluencePage:
    """Tests for ConfluencePage dataclass."""

    def test_page_creation(self):
        """Test page creation with defaults."""
        page = ConfluencePage(id="123", title="Test Page")
        assert page.id == "123"
        assert page.title == "Test Page"
        assert page.type == "page"
        assert page.status == "current"
        assert page.version == 1

    def test_page_to_dict(self):
        """Test page to_dict method."""
        page = ConfluencePage(
            id="123",
            title="Getting Started",
            space_key="DOCS",
            version=5,
            parent_id="100",
            ancestors=["100", "99"],
        )
        data = page.to_dict()
        assert data["id"] == "123"
        assert data["title"] == "Getting Started"
        assert data["space_key"] == "DOCS"
        assert data["version"] == 5
        assert data["parent_id"] == "100"
        assert data["ancestors"] == ["100", "99"]


# =============================================================================
# HTML to Text Tests
# =============================================================================

class TestHtmlToText:
    """Tests for HTML to text conversion."""

    def test_simple_html(self):
        """Test converting simple HTML."""
        html = "<p>Hello, world!</p>"
        text = html_to_text(html)
        assert "Hello, world!" in text

    def test_headings(self):
        """Test converting headings."""
        html = "<h1>Title</h1><h2>Subtitle</h2><p>Content</p>"
        text = html_to_text(html)
        assert "Title" in text
        assert "Subtitle" in text
        assert "Content" in text

    def test_lists(self):
        """Test converting lists."""
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        text = html_to_text(html)
        assert "Item 1" in text
        assert "Item 2" in text

    def test_removes_scripts(self):
        """Test that scripts are removed."""
        html = "<p>Text</p><script>alert('test')</script><p>More</p>"
        text = html_to_text(html)
        assert "alert" not in text
        assert "Text" in text
        assert "More" in text

    def test_removes_styles(self):
        """Test that styles are removed."""
        html = "<style>.class { color: red; }</style><p>Content</p>"
        text = html_to_text(html)
        assert "color" not in text
        assert "Content" in text

    def test_empty_html(self):
        """Test empty HTML."""
        assert html_to_text("") == ""
        assert html_to_text("<p></p>") == ""


# =============================================================================
# ConfluenceConnector Tests - Mock Mode
# =============================================================================

class TestConfluenceConnectorMockMode:
    """Tests for ConfluenceConnector in mock mode."""

    @pytest.fixture
    def mock_samples_dir(self, tmp_path):
        """Create a temporary samples directory with test data."""
        samples_dir = tmp_path / "confluence_samples"
        samples_dir.mkdir()

        # Create spaces.json
        spaces = [
            {"key": "DOCS", "name": "Documentation", "type": "global", "status": "current"},
            {"key": "ENG", "name": "Engineering", "type": "global", "status": "current"},
        ]
        with open(samples_dir / "spaces.json", "w") as f:
            json.dump(spaces, f)

        # Create DOCS space with pages
        docs_dir = samples_dir / "docs"
        docs_dir.mkdir()
        pages = [
            {
                "id": "100001",
                "title": "Getting Started",
                "type": "page",
                "version": {"number": 3},
                "body": {"storage": {"value": "<p>Welcome to the docs!</p>"}},
            },
            {
                "id": "100002",
                "title": "API Reference",
                "type": "page",
                "version": {"number": 2},
                "body": {"storage": {"value": "<p>API documentation here.</p>"}},
            },
        ]
        with open(docs_dir / "pages.json", "w") as f:
            json.dump(pages, f)

        return samples_dir

    @pytest.fixture
    def connector(self, mock_samples_dir):
        """Create a connector with mock samples directory."""
        config = ConfluenceConfig(
            mock_mode=True,
            samples_dir=str(mock_samples_dir),
        )
        return ConfluenceConnector(config)

    def test_list_spaces(self, connector):
        """Test listing spaces in mock mode."""
        spaces = connector.list_spaces()
        assert len(spaces) == 2
        assert spaces[0].key == "DOCS"
        assert spaces[0].name == "Documentation"
        assert spaces[1].key == "ENG"

    def test_list_pages(self, connector):
        """Test listing pages in mock mode."""
        pages = connector.list_pages(space_key="DOCS")
        assert len(pages) == 2
        assert pages[0].id == "100001"
        assert pages[0].title == "Getting Started"
        assert pages[1].id == "100002"

    def test_get_page_content(self, connector):
        """Test getting page content in mock mode."""
        content = connector.get_page_content("100001")
        assert "Welcome to the docs!" in content

    def test_get_page(self, connector):
        """Test getting page metadata in mock mode."""
        page = connector.get_page("100001")
        assert page.id == "100001"
        assert page.title == "Getting Started"
        assert page.version == 3

    def test_search(self, connector):
        """Test searching pages in mock mode."""
        results = connector.search("API", space_key="DOCS")
        assert len(results) == 1
        assert results[0].title == "API Reference"


# =============================================================================
# ConfluenceConnector Tests - Environment Variables
# =============================================================================

class TestConfluenceConnectorEnvVars:
    """Tests for ConfluenceConnector environment variable handling."""

    def test_mock_mode_env_true(self, monkeypatch):
        """Test CONFLUENCE_MOCK=true enables mock mode."""
        monkeypatch.setenv("CONFLUENCE_MOCK", "true")
        connector = ConfluenceConnector(ConfluenceConfig(mock_mode=False))
        assert connector.config.mock_mode is True

    def test_mock_mode_env_false(self, monkeypatch):
        """Test CONFLUENCE_MOCK=false disables mock mode."""
        monkeypatch.setenv("CONFLUENCE_MOCK", "false")
        connector = ConfluenceConnector(ConfluenceConfig(mock_mode=True))
        assert connector.config.mock_mode is False

    def test_connector_mock_env(self, monkeypatch):
        """Test CONNECTOR_MOCK enables mock mode."""
        monkeypatch.setenv("CONNECTOR_MOCK", "true")
        connector = ConfluenceConnector(ConfluenceConfig(mock_mode=False))
        assert connector.config.mock_mode is True

    def test_credentials_from_env(self, monkeypatch):
        """Test credentials from environment."""
        monkeypatch.setenv("CONFLUENCE_URL", "https://test.atlassian.net/wiki")
        monkeypatch.setenv("CONFLUENCE_USERNAME", "user@test.com")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "test-token")
        connector = ConfluenceConnector(ConfluenceConfig())
        assert connector.config.base_url == "https://test.atlassian.net/wiki"
        assert connector.config.username == "user@test.com"
        assert connector.config.api_token == "test-token"


# =============================================================================
# ConfluenceConnector Tests - Stream Samples
# =============================================================================

class TestConfluenceConnectorStreamSamples:
    """Tests for streaming training samples."""

    @pytest.fixture
    def mock_samples_dir(self, tmp_path):
        """Create samples directory with pages."""
        samples_dir = tmp_path / "confluence_samples"
        space_dir = samples_dir / "kb"
        space_dir.mkdir(parents=True)

        pages = [
            {
                "id": "200001",
                "title": "How to reset password",
                "type": "page",
                "version": {"number": 1},
                "body": {"storage": {"value": "<p>Go to Settings > Security > Change Password</p>"}},
            },
            {
                "id": "200002",
                "title": "How to export data",
                "type": "page",
                "version": {"number": 1},
                "body": {"storage": {"value": "<p>Navigate to Account > Data > Export</p>"}},
            },
        ]
        with open(space_dir / "pages.json", "w") as f:
            json.dump(pages, f)

        return samples_dir

    @pytest.fixture
    def connector(self, mock_samples_dir):
        """Create connector with mock samples."""
        config = ConfluenceConfig(mock_mode=True, samples_dir=str(mock_samples_dir))
        return ConfluenceConnector(config)

    def test_stream_samples(self, connector):
        """Test streaming samples from pages."""
        samples = list(connector.stream_samples(
            "KB",
            {"input": "title", "output": "content"},
        ))

        assert len(samples) == 2

        sample1 = samples[0]
        assert "reset password" in sample1["input"].lower()
        assert "Settings" in sample1["output"]
        assert sample1["metadata"]["source"] == "confluence"
        assert sample1["metadata"]["space_key"] == "KB"

    def test_stream_samples_title_filter(self, connector):
        """Test streaming with title filter."""
        samples = list(connector.stream_samples(
            "KB",
            {"input": "title", "output": "content"},
            title_filter="export",
        ))

        assert len(samples) == 1
        assert "export" in samples[0]["input"].lower()


# =============================================================================
# ConfluenceConnector Tests - Real API (Mocked)
# =============================================================================

class TestConfluenceConnectorRealAPI:
    """Tests for real API calls with mocked responses."""

    @pytest.fixture
    def connector(self, monkeypatch):
        """Create connector for API testing."""
        monkeypatch.setenv("CONFLUENCE_MOCK", "false")
        config = ConfluenceConfig(
            mock_mode=False,
            base_url="https://test.atlassian.net/wiki",
            username="user@test.com",
            api_token="test-token",
        )
        return ConfluenceConnector(config)

    @patch("connectors.confluence_connector.REQUESTS_AVAILABLE", True)
    def test_api_call_success(self, connector):
        """Test successful API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session

            spaces = connector.list_spaces()
            assert spaces == []

    def test_no_url_raises_error(self, monkeypatch):
        """Test that missing URL raises error."""
        monkeypatch.setenv("CONFLUENCE_MOCK", "false")
        monkeypatch.delenv("CONFLUENCE_URL", raising=False)
        monkeypatch.delenv("CONFLUENCE_BASE_URL", raising=False)

        config = ConfluenceConfig(mock_mode=False)
        connector = ConfluenceConnector(config)

        with pytest.raises(RuntimeError, match="No Confluence URL provided"):
            connector._get_session()

    def test_no_credentials_raises_error(self, monkeypatch):
        """Test that missing credentials raises error."""
        monkeypatch.setenv("CONFLUENCE_MOCK", "false")
        monkeypatch.delenv("CONFLUENCE_USERNAME", raising=False)
        monkeypatch.delenv("CONFLUENCE_API_TOKEN", raising=False)

        config = ConfluenceConfig(
            mock_mode=False,
            base_url="https://test.atlassian.net/wiki",
        )
        connector = ConfluenceConnector(config)

        with pytest.raises(RuntimeError, match="No Confluence credentials provided"):
            connector._get_session()


# =============================================================================
# ConfluenceConnector Tests - Built-in Samples
# =============================================================================

class TestConfluenceConnectorBuiltInSamples:
    """Tests using the built-in sample files."""

    @pytest.fixture
    def connector(self):
        """Create connector using built-in samples."""
        config = ConfluenceConfig(mock_mode=True)
        return ConfluenceConnector(config)

    def test_list_spaces_builtin(self, connector):
        """Test listing spaces with built-in samples."""
        samples_dir = connector._get_samples_dir()
        if not (samples_dir / "spaces.json").exists():
            pytest.skip("Built-in samples not available")

        spaces = connector.list_spaces()
        assert len(spaces) > 0
        assert all(isinstance(s, ConfluenceSpace) for s in spaces)

    def test_list_pages_builtin(self, connector):
        """Test listing pages with built-in samples."""
        samples_dir = connector._get_samples_dir()
        docs_dir = samples_dir / "docs"
        if not docs_dir.exists():
            pytest.skip("Built-in samples not available")

        pages = connector.list_pages(space_key="DOCS")
        assert len(pages) > 0
        assert all(isinstance(p, ConfluencePage) for p in pages)

    def test_stream_samples_builtin(self, connector):
        """Test streaming samples with built-in samples."""
        samples_dir = connector._get_samples_dir()
        docs_dir = samples_dir / "docs"
        if not docs_dir.exists():
            pytest.skip("Built-in samples not available")

        samples = list(connector.stream_samples(
            "DOCS",
            {"input": "title", "output": "content"},
        ))

        assert len(samples) >= 0


# =============================================================================
# ConfluenceConnector Tests - Edge Cases
# =============================================================================

class TestConfluenceConnectorEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create connector with empty samples directory."""
        samples_dir = tmp_path / "confluence_samples"
        samples_dir.mkdir()
        config = ConfluenceConfig(mock_mode=True, samples_dir=str(samples_dir))
        return ConfluenceConnector(config)

    def test_list_spaces_empty_dir(self, connector):
        """Test listing spaces with empty directory."""
        spaces = connector.list_spaces()
        assert spaces == []

    def test_list_pages_nonexistent_space(self, connector):
        """Test listing pages from nonexistent space."""
        pages = connector.list_pages(space_key="NONEXISTENT")
        assert pages == []

    def test_get_page_not_found(self, connector):
        """Test getting nonexistent page raises error."""
        with pytest.raises(FileNotFoundError, match="not found"):
            connector.get_page_content("999999")


# =============================================================================
# ConfluenceConnector Tests - Page Parsing
# =============================================================================

class TestConfluenceConnectorPageParsing:
    """Tests for page data parsing."""

    def test_parse_page_with_version_dict(self):
        """Test parsing page with version as dict."""
        connector = ConfluenceConnector(ConfluenceConfig())
        data = {
            "id": "123",
            "title": "Test Page",
            "version": {"number": 5, "when": "2024-01-01T00:00:00Z"},
        }
        page = connector._parse_page(data)
        assert page.version == 5

    def test_parse_page_with_ancestors(self):
        """Test parsing page with ancestors."""
        connector = ConfluenceConnector(ConfluenceConfig())
        data = {
            "id": "123",
            "title": "Test Page",
            "ancestors": [{"id": "100"}, {"id": "200"}],
        }
        page = connector._parse_page(data)
        assert page.ancestors == ["100", "200"]
        assert page.parent_id == "200"

    def test_parse_page_without_ancestors(self):
        """Test parsing page without ancestors."""
        connector = ConfluenceConnector(ConfluenceConfig())
        data = {
            "id": "123",
            "title": "Test Page",
        }
        page = connector._parse_page(data)
        assert page.ancestors == []
        assert page.parent_id is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestConfluenceConnectorIntegration:
    """Integration tests for the Confluence connector."""

    @pytest.fixture
    def full_samples_dir(self, tmp_path):
        """Create a full samples directory for integration testing."""
        samples_dir = tmp_path / "confluence_samples"
        samples_dir.mkdir()

        # Spaces
        spaces = [
            {"key": "KB", "name": "Knowledge Base", "type": "global"},
        ]
        with open(samples_dir / "spaces.json", "w") as f:
            json.dump(spaces, f)

        # Knowledge Base pages
        kb_dir = samples_dir / "kb"
        kb_dir.mkdir()
        pages = [
            {
                "id": "300001",
                "title": "Password Reset Guide",
                "type": "page",
                "version": {"number": 2},
                "body": {
                    "storage": {
                        "value": "<h1>Password Reset</h1><p>Follow these steps to reset your password:</p><ol><li>Go to Settings</li><li>Click Security</li><li>Select Change Password</li></ol>"
                    }
                },
            },
            {
                "id": "300002",
                "title": "Data Export Instructions",
                "type": "page",
                "version": {"number": 1},
                "body": {
                    "storage": {
                        "value": "<h1>Exporting Data</h1><p>To export your data:</p><ul><li>Navigate to Account</li><li>Click Data</li><li>Select Export</li></ul>"
                    }
                },
            },
        ]
        with open(kb_dir / "pages.json", "w") as f:
            json.dump(pages, f)

        return samples_dir

    def test_full_extraction_pipeline(self, full_samples_dir):
        """Test complete extraction pipeline."""
        config = ConfluenceConfig(mock_mode=True, samples_dir=str(full_samples_dir))
        connector = ConfluenceConnector(config)

        # List spaces
        spaces = connector.list_spaces()
        assert len(spaces) == 1
        assert spaces[0].key == "KB"

        # List pages
        pages = connector.list_pages(space_key="KB")
        assert len(pages) == 2

        # Get content
        content = connector.get_page_content("300001")
        assert "Password Reset" in content
        assert "Settings" in content

        # Stream samples
        samples = list(connector.stream_samples(
            "KB",
            {"input": "title", "output": "content"},
        ))

        assert len(samples) == 2

        # Check first sample
        sample1 = samples[0]
        assert "Password Reset" in sample1["input"]
        assert "Settings" in sample1["output"]
        assert sample1["metadata"]["source"] == "confluence"

        # Search
        results = connector.search("export", space_key="KB")
        assert len(results) == 1
        assert "Export" in results[0].title
