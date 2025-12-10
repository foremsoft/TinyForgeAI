"""Tests for the REST API connector module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from connectors.api_connector import (
    APIConnector,
    APIConnectorConfig,
    PaginationConfig,
)


@pytest.fixture
def api_config():
    """Create a basic API connector configuration."""
    return APIConnectorConfig(
        base_url="https://api.example.com/v1",
        auth_type="bearer",
        auth_value="test-token-123",
    )


@pytest.fixture
def connector(api_config):
    """Create an API connector instance."""
    return APIConnector(api_config)


class TestAPIConnectorConfig:
    """Tests for APIConnectorConfig."""

    def test_base_url_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base_url."""
        config = APIConnectorConfig(base_url="https://api.example.com/")
        assert config.base_url == "https://api.example.com"

    def test_get_auth_headers_bearer(self):
        """Test bearer token auth headers."""
        config = APIConnectorConfig(
            base_url="https://api.example.com",
            auth_type="bearer",
            auth_value="my-token",
        )
        headers = config.get_auth_headers()
        assert headers == {"Authorization": "Bearer my-token"}

    def test_get_auth_headers_api_key(self):
        """Test API key auth headers."""
        config = APIConnectorConfig(
            base_url="https://api.example.com",
            auth_type="api_key",
            auth_value="my-api-key",
        )
        headers = config.get_auth_headers()
        assert headers == {"X-API-Key": "my-api-key"}

    def test_get_auth_headers_basic(self):
        """Test basic auth headers."""
        config = APIConnectorConfig(
            base_url="https://api.example.com",
            auth_type="basic",
            auth_value="user:pass",
        )
        headers = config.get_auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Basic ")

    def test_get_auth_headers_none(self):
        """Test no auth headers when auth_type is None."""
        config = APIConnectorConfig(base_url="https://api.example.com")
        headers = config.get_auth_headers()
        assert headers == {}


class TestPaginationConfig:
    """Tests for PaginationConfig."""

    def test_default_values(self):
        """Test default pagination configuration."""
        config = PaginationConfig()
        assert config.style == "none"
        assert config.page_size == 100
        assert config.max_pages is None

    def test_page_style(self):
        """Test page-based pagination configuration."""
        config = PaginationConfig(
            style="page",
            page_param="p",
            limit_param="per_page",
            page_size=50,
        )
        assert config.style == "page"
        assert config.page_param == "p"
        assert config.limit_param == "per_page"
        assert config.page_size == 50


class TestAPIConnector:
    """Tests for APIConnector."""

    def test_extract_items_from_list(self, connector):
        """Test extracting items from list response."""
        response = [{"id": 1}, {"id": 2}]
        items = connector._extract_items(response)
        assert len(items) == 2

    def test_extract_items_from_data_key(self, connector):
        """Test extracting items from 'data' key."""
        response = {"data": [{"id": 1}, {"id": 2}], "meta": {}}
        items = connector._extract_items(response)
        assert len(items) == 2

    def test_extract_items_from_items_key(self, connector):
        """Test extracting items from 'items' key."""
        response = {"items": [{"id": 1}], "total": 1}
        items = connector._extract_items(response)
        assert len(items) == 1

    def test_extract_items_from_results_key(self, connector):
        """Test extracting items from 'results' key."""
        response = {"results": [{"id": 1}, {"id": 2}, {"id": 3}]}
        items = connector._extract_items(response)
        assert len(items) == 3

    def test_extract_items_single_dict(self, connector):
        """Test extracting single dict as item."""
        response = {"id": 1, "name": "test"}
        items = connector._extract_items(response)
        assert len(items) == 1
        assert items[0]["id"] == 1

    def test_extract_cursor_from_common_keys(self, connector):
        """Test extracting cursor from common response keys."""
        response = {"data": [], "next_cursor": "abc123"}
        cursor = connector._extract_cursor(response, None)
        assert cursor == "abc123"

    def test_extract_cursor_from_path(self, connector):
        """Test extracting cursor using JSONPath."""
        response = {"data": [], "pagination": {"next": "xyz789"}}
        cursor = connector._extract_cursor(response, "pagination.next")
        assert cursor == "xyz789"

    def test_extract_cursor_returns_none_when_missing(self, connector):
        """Test that missing cursor returns None."""
        response = {"data": []}
        cursor = connector._extract_cursor(response, None)
        assert cursor is None

    def test_get_nested_value(self, connector):
        """Test getting nested values with dot notation."""
        item = {"user": {"profile": {"name": "John"}}}
        value = connector._get_nested_value(item, "user.profile.name")
        assert value == "John"

    def test_get_nested_value_simple(self, connector):
        """Test getting simple value."""
        item = {"name": "John"}
        value = connector._get_nested_value(item, "name")
        assert value == "John"

    def test_get_nested_value_returns_none_for_missing(self, connector):
        """Test that missing path returns None."""
        item = {"name": "John"}
        value = connector._get_nested_value(item, "missing.path")
        assert value is None

    def test_item_to_sample_basic(self, connector):
        """Test converting item to training sample."""
        item = {"question": "What is AI?", "answer": "Artificial Intelligence"}
        mapping = {"input": "question", "output": "answer"}

        sample = connector._item_to_sample(item, mapping)

        assert sample["input"] == "What is AI?"
        assert sample["output"] == "Artificial Intelligence"
        assert sample["metadata"]["source"] == "api"

    def test_item_to_sample_nested_fields(self, connector):
        """Test converting item with nested fields."""
        item = {
            "data": {"text": "Hello"},
            "response": {"content": "Hi there"},
        }
        mapping = {"input": "data.text", "output": "response.content"}

        sample = connector._item_to_sample(item, mapping)

        assert sample["input"] == "Hello"
        assert sample["output"] == "Hi there"

    def test_item_to_sample_missing_input_raises(self, connector):
        """Test that missing input field raises KeyError."""
        item = {"answer": "test"}
        mapping = {"input": "question", "output": "answer"}

        with pytest.raises(KeyError, match="input field"):
            connector._item_to_sample(item, mapping)

    def test_item_to_sample_missing_output_raises(self, connector):
        """Test that missing output field raises KeyError."""
        item = {"question": "test"}
        mapping = {"input": "question", "output": "answer"}

        with pytest.raises(KeyError, match="output field"):
            connector._item_to_sample(item, mapping)

    def test_item_to_sample_missing_mapping_input_raises(self, connector):
        """Test that missing mapping input key raises KeyError."""
        item = {"question": "test", "answer": "test"}
        mapping = {"output": "answer"}

        with pytest.raises(KeyError, match="'input' key"):
            connector._item_to_sample(item, mapping)

    def test_item_to_sample_missing_mapping_output_raises(self, connector):
        """Test that missing mapping output key raises KeyError."""
        item = {"question": "test", "answer": "test"}
        mapping = {"input": "question"}

        with pytest.raises(KeyError, match="'output' key"):
            connector._item_to_sample(item, mapping)


class TestAPIConnectorIntegration:
    """Integration tests for APIConnector with mocked HTTP."""

    @patch("connectors.api_connector._REQUESTS_AVAILABLE", False)
    def test_works_without_requests_library(self, api_config):
        """Test that connector works with urllib fallback."""
        connector = APIConnector(api_config)
        assert connector is not None

    def test_stream_samples_with_mock_response(self, connector):
        """Test streaming samples with mocked API response."""
        mock_response = {
            "data": [
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
            ]
        }

        with patch.object(connector, "_make_request", return_value=mock_response):
            mapping = {"input": "question", "output": "answer"}
            samples = list(connector.stream_samples("/qa", mapping))

            assert len(samples) == 2
            assert samples[0]["input"] == "Q1"
            assert samples[1]["output"] == "A2"

    def test_stream_samples_with_pagination(self, connector):
        """Test streaming samples with pagination."""
        page1 = {"data": [{"q": "Q1", "a": "A1"}]}
        page2 = {"data": [{"q": "Q2", "a": "A2"}]}
        page3 = {"data": []}  # Empty page signals end

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page1
            elif call_count == 2:
                return page2
            return page3

        pagination = PaginationConfig(style="page", page_size=1, max_pages=3)

        with patch.object(connector, "_make_request", side_effect=mock_request):
            mapping = {"input": "q", "output": "a"}
            samples = list(
                connector.stream_samples("/qa", mapping, pagination=pagination)
            )

            assert len(samples) == 2

    def test_test_connection_success(self, connector):
        """Test connection test with successful response."""
        with patch.object(connector, "_make_request", return_value={"status": "ok"}):
            assert connector.test_connection() is True

    def test_test_connection_failure(self, connector):
        """Test connection test with failed response."""
        with patch.object(
            connector, "_make_request", side_effect=RuntimeError("Connection failed")
        ):
            assert connector.test_connection() is False

    def test_stream_samples_with_offset_pagination(self, connector):
        """Test streaming samples with offset-based pagination."""
        page1 = {"data": [{"q": "Q1", "a": "A1"}]}
        page2 = {"data": [{"q": "Q2", "a": "A2"}]}
        page3 = {"data": []}

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page1
            elif call_count == 2:
                return page2
            return page3

        pagination = PaginationConfig(style="offset", page_size=1, max_pages=3)

        with patch.object(connector, "_make_request", side_effect=mock_request):
            mapping = {"input": "q", "output": "a"}
            samples = list(
                connector.stream_samples("/qa", mapping, pagination=pagination)
            )

            assert len(samples) == 2
            assert samples[0]["input"] == "Q1"
            assert samples[1]["input"] == "Q2"

    def test_stream_samples_with_cursor_pagination(self, connector):
        """Test streaming samples with cursor-based pagination."""
        page1 = {"data": [{"q": "Q1", "a": "A1"}], "next_cursor": "cursor_2"}
        page2 = {"data": [{"q": "Q2", "a": "A2"}], "next_cursor": None}

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page1
            return page2

        pagination = PaginationConfig(style="cursor", page_size=1)

        with patch.object(connector, "_make_request", side_effect=mock_request):
            mapping = {"input": "q", "output": "a"}
            samples = list(
                connector.stream_samples("/qa", mapping, pagination=pagination)
            )

            assert len(samples) == 2

    def test_rate_limiting(self, api_config):
        """Test that rate limiting applies delay between requests."""
        api_config.rate_limit_delay = 0.1
        connector = APIConnector(api_config)

        page1 = {"data": [{"q": "Q1", "a": "A1"}]}
        page2 = {"data": []}

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page1
            return page2

        pagination = PaginationConfig(style="page", page_size=1, max_pages=2)

        import time
        start = time.time()
        with patch.object(connector, "_make_request", side_effect=mock_request):
            mapping = {"input": "q", "output": "a"}
            list(connector.stream_samples("/qa", mapping, pagination=pagination))
        elapsed = time.time() - start

        # Should have at least one rate limit delay
        assert elapsed >= 0.1


class TestRetryLogic:
    """Tests for retry with exponential backoff."""

    def test_retry_on_server_error(self):
        """Test that requests are retried on 500 errors."""
        config = APIConnectorConfig(
            base_url="https://api.example.com",
            max_retries=2,
            retry_backoff=0.01,  # Fast for testing
        )
        connector = APIConnector(config)

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Simulate server error response
                mock_resp = MagicMock()
                mock_resp.status_code = 500
                mock_resp.headers = {}
                return mock_resp
            # Success on third try
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b'{"data": "success"}'
            mock_resp.headers = {"Content-Type": "application/json"}
            return mock_resp

        with patch.object(connector, "_get_session") as mock_session:
            mock_session.return_value.request = mock_request
            result = connector._make_request("/test")

        assert result == {"data": "success"}
        assert call_count == 3

    def test_no_retry_on_client_error(self):
        """Test that 4xx errors are not retried (except 429)."""
        config = APIConnectorConfig(
            base_url="https://api.example.com",
            max_retries=3,
        )
        connector = APIConnector(config)

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            # Simulate requests.HTTPError
            from requests import HTTPError
            mock_resp.raise_for_status.side_effect = HTTPError("404 Not Found")
            return mock_resp

        with patch.object(connector, "_get_session") as mock_session:
            mock_session.return_value.request = mock_request
            with pytest.raises(RuntimeError):
                connector._make_request("/test")

        # Should only have tried once (no retries for 404)
        assert call_count == 1

    def test_retry_on_rate_limit(self):
        """Test that 429 rate limit errors are retried."""
        config = APIConnectorConfig(
            base_url="https://api.example.com",
            max_retries=1,
            retry_backoff=0.01,
        )
        connector = APIConnector(config)

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            if call_count == 1:
                mock_resp.status_code = 429
                mock_resp.headers = {"Retry-After": "0.01"}
                return mock_resp
            mock_resp.status_code = 200
            mock_resp.content = b'{"ok": true}'
            mock_resp.headers = {"Content-Type": "application/json"}
            return mock_resp

        with patch.object(connector, "_get_session") as mock_session:
            mock_session.return_value.request = mock_request
            result = connector._make_request("/test")

        assert result == {"ok": True}
        assert call_count == 2


class TestXMLParsing:
    """Tests for XML response parsing."""

    def test_parse_simple_xml(self):
        """Test parsing simple XML structure."""
        config = APIConnectorConfig(
            base_url="https://api.example.com",
            response_format="xml",
        )
        connector = APIConnector(config)

        xml_content = b"<root><item>value</item></root>"
        result = connector._parse_xml(xml_content)

        assert result == {"root": {"item": "value"}}

    def test_parse_xml_with_attributes(self):
        """Test parsing XML with attributes."""
        config = APIConnectorConfig(
            base_url="https://api.example.com",
            response_format="xml",
        )
        connector = APIConnector(config)

        xml_content = b'<root><item id="1" type="test">value</item></root>'
        result = connector._parse_xml(xml_content)

        assert result["root"]["item"]["@attributes"]["id"] == "1"
        assert result["root"]["item"]["#text"] == "value"

    def test_parse_xml_with_multiple_children(self):
        """Test parsing XML with repeated elements."""
        config = APIConnectorConfig(
            base_url="https://api.example.com",
            response_format="xml",
        )
        connector = APIConnector(config)

        xml_content = b"<root><item>one</item><item>two</item><item>three</item></root>"
        result = connector._parse_xml(xml_content)

        assert result["root"]["item"] == ["one", "two", "three"]

    def test_auto_detect_xml_content_type(self):
        """Test that XML is auto-detected from content-type."""
        config = APIConnectorConfig(base_url="https://api.example.com")
        connector = APIConnector(config)

        xml_content = b"<response><status>ok</status></response>"
        result = connector._parse_response(xml_content, "application/xml")

        assert result == {"response": {"status": "ok"}}


class TestResponseCaching:
    """Tests for response caching."""

    def test_cache_stores_response(self):
        """Test that responses are cached."""
        config = APIConnectorConfig(
            base_url="https://api.example.com",
            cache_enabled=True,
            cache_ttl=300,
        )
        connector = APIConnector(config)

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b'{"data": "cached"}'
            mock_resp.headers = {"Content-Type": "application/json"}
            return mock_resp

        with patch.object(connector, "_get_session") as mock_session:
            mock_session.return_value.request = mock_request

            # First request
            result1 = connector._make_request("/test")
            # Second request (should use cache)
            result2 = connector._make_request("/test")

        assert result1 == result2
        assert call_count == 1  # Only one actual request

    def test_cache_respects_ttl(self):
        """Test that cache expires after TTL."""
        config = APIConnectorConfig(
            base_url="https://api.example.com",
            cache_enabled=True,
            cache_ttl=0,  # Expire immediately
        )
        connector = APIConnector(config)

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = json.dumps({"count": call_count}).encode()
            mock_resp.headers = {"Content-Type": "application/json"}
            return mock_resp

        with patch.object(connector, "_get_session") as mock_session:
            mock_session.return_value.request = mock_request

            result1 = connector._make_request("/test")
            import time
            time.sleep(0.01)  # Ensure TTL expires
            result2 = connector._make_request("/test")

        assert call_count == 2  # Both requests made

    def test_clear_cache(self):
        """Test that cache can be cleared."""
        config = APIConnectorConfig(
            base_url="https://api.example.com",
            cache_enabled=True,
        )
        connector = APIConnector(config)

        # Manually add to cache
        connector._cache["test_key"] = (0, {"cached": True})

        connector.clear_cache()

        assert len(connector._cache) == 0

    def test_cache_disabled_by_default(self):
        """Test that caching is disabled by default."""
        config = APIConnectorConfig(base_url="https://api.example.com")
        connector = APIConnector(config)

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b'{"data": "test"}'
            mock_resp.headers = {"Content-Type": "application/json"}
            return mock_resp

        with patch.object(connector, "_get_session") as mock_session:
            mock_session.return_value.request = mock_request

            connector._make_request("/test")
            connector._make_request("/test")

        assert call_count == 2  # No caching


class TestBatchRequests:
    """Tests for request batching."""

    def test_batch_request_splits_items(self):
        """Test that items are split into batches."""
        config = APIConnectorConfig(base_url="https://api.example.com")
        connector = APIConnector(config)

        items = [{"id": i} for i in range(5)]
        batches_received = []

        def mock_request(endpoint, method, json_body=None, **kwargs):
            batches_received.append(json_body["items"])
            return {"processed": len(json_body["items"])}

        with patch.object(connector, "_make_request", side_effect=mock_request):
            results = list(connector.batch_request("/bulk", items, batch_size=2))

        assert len(results) == 3  # 5 items / 2 per batch = 3 batches
        assert len(batches_received[0]) == 2
        assert len(batches_received[1]) == 2
        assert len(batches_received[2]) == 1

    def test_batch_request_custom_param(self):
        """Test batch request with custom parameter name."""
        config = APIConnectorConfig(base_url="https://api.example.com")
        connector = APIConnector(config)

        items = [{"id": 1}, {"id": 2}]
        received_body = None

        def mock_request(endpoint, method, json_body=None, **kwargs):
            nonlocal received_body
            received_body = json_body
            return {"ok": True}

        with patch.object(connector, "_make_request", side_effect=mock_request):
            list(connector.batch_request("/bulk", items, batch_param="records"))

        assert "records" in received_body
        assert received_body["records"] == items
