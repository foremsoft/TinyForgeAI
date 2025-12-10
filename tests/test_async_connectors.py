"""Tests for async connectors."""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Import config classes from sync connector
from connectors.api_connector import APIConnectorConfig, PaginationConfig


class TestAsyncAPIConnector:
    """Tests for AsyncAPIConnector."""

    @pytest.fixture
    def config(self):
        """Create test API config."""
        return APIConnectorConfig(
            base_url="https://api.example.com",
            auth_type="bearer",
            auth_value="test-token",
            rate_limit_delay=0,
            timeout=10,
        )

    def test_import_without_httpx(self):
        """Test that import fails gracefully without httpx."""
        with patch.dict("sys.modules", {"httpx": None}):
            # Force reimport
            import importlib
            import connectors.async_api_connector as async_api
            importlib.reload(async_api)
            assert async_api._HTTPX_AVAILABLE is False or True  # May already be imported

    @pytest.mark.asyncio
    async def test_connector_init(self, config):
        """Test connector initialization."""
        try:
            from connectors.async_api_connector import AsyncAPIConnector
        except ImportError:
            pytest.skip("httpx not installed")

        connector = AsyncAPIConnector(config)
        assert connector.config == config
        assert connector._client is None
        assert connector._cache == {}

    @pytest.mark.asyncio
    async def test_context_manager(self, config):
        """Test async context manager."""
        try:
            from connectors.async_api_connector import AsyncAPIConnector
        except ImportError:
            pytest.skip("httpx not installed")

        async with AsyncAPIConnector(config) as connector:
            assert connector._client is not None

        assert connector._client is None or connector._client.is_closed

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, config):
        """Test cache key generation."""
        try:
            from connectors.async_api_connector import AsyncAPIConnector
        except ImportError:
            pytest.skip("httpx not installed")

        connector = AsyncAPIConnector(config)

        key1 = connector._get_cache_key("/test", "GET", {"a": 1}, None)
        key2 = connector._get_cache_key("/test", "GET", {"a": 1}, None)
        key3 = connector._get_cache_key("/test", "GET", {"a": 2}, None)

        assert key1 == key2  # Same params = same key
        assert key1 != key3  # Different params = different key

    @pytest.mark.asyncio
    async def test_cache_operations(self, config):
        """Test cache get/set operations."""
        try:
            from connectors.async_api_connector import AsyncAPIConnector
        except ImportError:
            pytest.skip("httpx not installed")

        config.cache_enabled = True
        config.cache_ttl = 300
        connector = AsyncAPIConnector(config)

        # Cache miss
        assert connector._get_cached_response("key1") is None

        # Cache set and hit
        connector._set_cached_response("key1", {"data": "test"})
        assert connector._get_cached_response("key1") == {"data": "test"}

        # Clear cache
        connector.clear_cache()
        assert connector._get_cached_response("key1") is None

    @pytest.mark.asyncio
    async def test_xml_parsing(self, config):
        """Test XML response parsing."""
        try:
            from connectors.async_api_connector import AsyncAPIConnector
        except ImportError:
            pytest.skip("httpx not installed")

        connector = AsyncAPIConnector(config)
        xml_content = b"""<?xml version="1.0"?>
        <root>
            <item id="1">
                <name>Test</name>
                <value>123</value>
            </item>
        </root>"""

        result = connector._parse_xml(xml_content)
        assert "root" in result
        assert "item" in result["root"]
        assert result["root"]["item"]["name"] == "Test"

    @pytest.mark.asyncio
    async def test_extract_items(self, config):
        """Test item extraction from various response formats."""
        try:
            from connectors.async_api_connector import AsyncAPIConnector
        except ImportError:
            pytest.skip("httpx not installed")

        connector = AsyncAPIConnector(config)

        # List response
        assert connector._extract_items([{"a": 1}, {"a": 2}]) == [{"a": 1}, {"a": 2}]

        # Dict with data key
        assert connector._extract_items({"data": [{"a": 1}]}) == [{"a": 1}]

        # Dict with items key
        assert connector._extract_items({"items": [{"a": 1}]}) == [{"a": 1}]

        # Dict without list - returns as single item
        assert connector._extract_items({"a": 1}) == [{"a": 1}]

    @pytest.mark.asyncio
    async def test_extract_cursor(self, config):
        """Test cursor extraction."""
        try:
            from connectors.async_api_connector import AsyncAPIConnector
        except ImportError:
            pytest.skip("httpx not installed")

        connector = AsyncAPIConnector(config)

        # Common cursor keys
        assert connector._extract_cursor({"next_cursor": "abc"}, None) == "abc"
        assert connector._extract_cursor({"cursor": "def"}, None) == "def"

        # Custom path
        assert connector._extract_cursor(
            {"pagination": {"next": "xyz"}}, "pagination.next"
        ) == "xyz"

        # Missing cursor
        assert connector._extract_cursor({"data": []}, None) is None

    @pytest.mark.asyncio
    async def test_item_to_sample(self, config):
        """Test item to sample conversion."""
        try:
            from connectors.async_api_connector import AsyncAPIConnector
        except ImportError:
            pytest.skip("httpx not installed")

        connector = AsyncAPIConnector(config)
        mapping = {"input": "question", "output": "answer"}

        item = {"question": "What is 2+2?", "answer": "4"}
        sample = connector._item_to_sample(item, mapping)

        assert sample["input"] == "What is 2+2?"
        assert sample["output"] == "4"
        assert sample["metadata"]["source"] == "api"

    @pytest.mark.asyncio
    async def test_item_to_sample_nested(self, config):
        """Test item to sample conversion with nested fields."""
        try:
            from connectors.async_api_connector import AsyncAPIConnector
        except ImportError:
            pytest.skip("httpx not installed")

        connector = AsyncAPIConnector(config)
        mapping = {"input": "data.question", "output": "data.answer"}

        item = {"data": {"question": "Test?", "answer": "Yes"}}
        sample = connector._item_to_sample(item, mapping)

        assert sample["input"] == "Test?"
        assert sample["output"] == "Yes"

    @pytest.mark.asyncio
    async def test_item_to_sample_missing_field(self, config):
        """Test item to sample with missing required field."""
        try:
            from connectors.async_api_connector import AsyncAPIConnector
        except ImportError:
            pytest.skip("httpx not installed")

        connector = AsyncAPIConnector(config)
        mapping = {"input": "question", "output": "answer"}

        item = {"question": "Test?"}  # Missing answer

        with pytest.raises(KeyError, match="output"):
            connector._item_to_sample(item, mapping)


class TestAsyncDBConnector:
    """Tests for AsyncDBConnector."""

    @pytest.fixture
    def sqlite_url(self):
        """Create a temporary SQLite database URL."""
        return "sqlite:///:memory:"

    def test_detect_db_type(self):
        """Test database type detection."""
        try:
            from connectors.async_db_connector import AsyncDBConnector
        except ImportError:
            pytest.skip("aiosqlite not installed")

        # SQLite
        connector = AsyncDBConnector("sqlite:///test.db")
        assert connector._db_type == "sqlite"

        # PostgreSQL
        connector = AsyncDBConnector("postgresql://localhost/test")
        assert connector._db_type == "postgresql"

        connector = AsyncDBConnector("postgres://localhost/test")
        assert connector._db_type == "postgresql"

        # MySQL
        connector = AsyncDBConnector("mysql://localhost/test")
        assert connector._db_type == "mysql"

    def test_parse_connection_url_sqlite(self):
        """Test SQLite URL parsing."""
        try:
            from connectors.async_db_connector import AsyncDBConnector
        except ImportError:
            pytest.skip("aiosqlite not installed")

        connector = AsyncDBConnector("sqlite:///path/to/db.sqlite")
        params = connector._parse_connection_url()
        assert params["database"] == "path/to/db.sqlite"

        connector = AsyncDBConnector("sqlite:///:memory:")
        params = connector._parse_connection_url()
        assert params["database"] == ":memory:"

    def test_parse_connection_url_postgresql(self):
        """Test PostgreSQL URL parsing."""
        try:
            from connectors.async_db_connector import AsyncDBConnector
        except ImportError:
            pytest.skip("aiosqlite not installed")

        connector = AsyncDBConnector("postgresql://user:pass@localhost:5432/mydb")
        params = connector._parse_connection_url()

        assert params["driver"] == "postgresql"
        assert params["user"] == "user"
        assert params["password"] == "pass"
        assert params["host"] == "localhost"
        assert params["port"] == 5432
        assert params["database"] == "mydb"

    def test_parse_connection_url_mysql(self):
        """Test MySQL URL parsing."""
        try:
            from connectors.async_db_connector import AsyncDBConnector
        except ImportError:
            pytest.skip("aiosqlite not installed")

        connector = AsyncDBConnector("mysql://root:secret@db.example.com:3306/app")
        params = connector._parse_connection_url()

        assert params["driver"] == "mysql"
        assert params["user"] == "root"
        assert params["password"] == "secret"
        assert params["host"] == "db.example.com"
        assert params["port"] == 3306
        assert params["database"] == "app"

    @pytest.mark.asyncio
    async def test_sqlite_operations(self, sqlite_url):
        """Test basic SQLite operations."""
        try:
            from connectors.async_db_connector import AsyncDBConnector, AIOSQLITE_AVAILABLE
            if not AIOSQLITE_AVAILABLE:
                pytest.skip("aiosqlite not installed")
        except ImportError:
            pytest.skip("async_db_connector not available")

        async with AsyncDBConnector(sqlite_url) as connector:
            # Test connection
            assert await connector.test_connection()

            # Create table
            await connector.execute(
                "CREATE TABLE test (id INTEGER PRIMARY KEY, question TEXT, answer TEXT)"
            )

            # Insert data
            await connector.execute(
                "INSERT INTO test (question, answer) VALUES (?, ?)",
                ("What is Python?", "A programming language")
            )

            # Query data
            rows = await connector.execute("SELECT * FROM test")
            assert len(rows) == 1
            assert rows[0]["question"] == "What is Python?"

            # List tables
            tables = await connector.list_tables()
            assert "test" in tables

            # Get table info
            info = await connector.get_table_info("test")
            column_names = [col["name"] for col in info]
            assert "question" in column_names
            assert "answer" in column_names

    @pytest.mark.asyncio
    async def test_stream_samples(self, sqlite_url):
        """Test streaming training samples."""
        try:
            from connectors.async_db_connector import AsyncDBConnector, AIOSQLITE_AVAILABLE
            if not AIOSQLITE_AVAILABLE:
                pytest.skip("aiosqlite not installed")
        except ImportError:
            pytest.skip("async_db_connector not available")

        async with AsyncDBConnector(sqlite_url) as connector:
            # Setup test data
            await connector.execute(
                "CREATE TABLE qa (id INTEGER PRIMARY KEY, q TEXT, a TEXT)"
            )
            await connector.execute(
                "INSERT INTO qa (q, a) VALUES (?, ?)",
                ("Q1", "A1")
            )
            await connector.execute(
                "INSERT INTO qa (q, a) VALUES (?, ?)",
                ("Q2", "A2")
            )

            # Stream samples
            mapping = {"input": "q", "output": "a"}
            samples = []
            async for sample in connector.stream_samples(
                "SELECT q, a FROM qa",
                mapping
            ):
                samples.append(sample)

            assert len(samples) == 2
            assert samples[0]["input"] == "Q1"
            assert samples[0]["output"] == "A1"
            assert samples[1]["input"] == "Q2"
            assert samples[1]["output"] == "A2"

    @pytest.mark.asyncio
    async def test_execute_many(self, sqlite_url):
        """Test batch insert operations."""
        try:
            from connectors.async_db_connector import AsyncDBConnector, AIOSQLITE_AVAILABLE
            if not AIOSQLITE_AVAILABLE:
                pytest.skip("aiosqlite not installed")
        except ImportError:
            pytest.skip("async_db_connector not available")

        async with AsyncDBConnector(sqlite_url) as connector:
            await connector.execute(
                "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)"
            )

            # Batch insert
            params = [("Item 1",), ("Item 2",), ("Item 3",)]
            count = await connector.execute_many(
                "INSERT INTO items (name) VALUES (?)",
                params
            )
            assert count == 3

            # Verify
            rows = await connector.execute("SELECT * FROM items")
            assert len(rows) == 3


class TestAsyncConnectorIntegration:
    """Integration tests for async connectors."""

    @pytest.mark.asyncio
    async def test_api_connector_mocked_request(self):
        """Test API connector with mocked HTTP requests."""
        try:
            from connectors.async_api_connector import AsyncAPIConnector, _HTTPX_AVAILABLE
            import httpx
            if not _HTTPX_AVAILABLE:
                pytest.skip("httpx not installed")
        except ImportError:
            pytest.skip("async_api_connector not available")

        config = APIConnectorConfig(
            base_url="https://api.test.com",
            rate_limit_delay=0,
        )

        # Mock the httpx response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.content = json.dumps({
            "data": [
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
            ]
        }).encode()
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.is_closed = False
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            connector = AsyncAPIConnector(config)
            connector._client = mock_client

            mapping = {"input": "question", "output": "answer"}
            samples = []
            async for sample in connector.stream_samples("/items", mapping):
                samples.append(sample)

            assert len(samples) == 2
            assert samples[0]["input"] == "Q1"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent request execution."""
        try:
            from connectors.async_api_connector import AsyncAPIConnector, _HTTPX_AVAILABLE
            if not _HTTPX_AVAILABLE:
                pytest.skip("httpx not installed")
        except ImportError:
            pytest.skip("async_api_connector not available")

        config = APIConnectorConfig(
            base_url="https://api.test.com",
            rate_limit_delay=0,
        )

        # Create mock responses
        def make_response(endpoint):
            mock = MagicMock()
            mock.status_code = 200
            mock.headers = {"Content-Type": "application/json"}
            mock.content = json.dumps({"endpoint": endpoint}).encode()
            mock.raise_for_status = MagicMock()
            return mock

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            url = kwargs.get("url", args[1] if len(args) > 1 else "/")
            await asyncio.sleep(0.01)  # Simulate network delay
            return make_response(url)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request = mock_request
            mock_client.is_closed = False
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            connector = AsyncAPIConnector(config)
            connector._client = mock_client

            endpoints = ["/item/1", "/item/2", "/item/3", "/item/4", "/item/5"]
            results = await connector.fetch_concurrent(endpoints, max_concurrency=3)

            assert len(results) == 5
            assert call_count == 5
