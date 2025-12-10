"""
Async Notion connector for TinyForgeAI.

Provides asynchronous functionality to list pages, get content, and stream
training samples from Notion databases and pages.

Uses httpx for async HTTP requests.

Mock Mode:
    When NOTION_MOCK=true (default), the connector reads from local sample
    files in examples/notion_samples/ instead of making real API calls.

Usage:
    from connectors.async_notion_connector import AsyncNotionConnector
    from connectors.notion_connector import NotionConfig

    # Create connector (uses mock mode by default)
    async with AsyncNotionConnector() as connector:
        # List pages in a database
        pages = await connector.list_pages(database_id="my_database")

        # Get page content
        content = await connector.get_page_content(page_id="my_page")

        # Stream training samples from database
        async for sample in connector.stream_samples(database_id="my_database", mapping={...}):
            print(sample)
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

# Add project root to path when run as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from connectors.mappers import row_to_sample
from connectors.notion_connector import NotionConfig, NotionPage, NotionDatabase

logger = logging.getLogger(__name__)

# Async HTTP client - use httpx if available
_HTTPX_AVAILABLE = False
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    pass

# Async file operations - use aiofiles if available
_AIOFILES_AVAILABLE = False
try:
    import aiofiles
    _AIOFILES_AVAILABLE = True
except ImportError:
    pass


class AsyncNotionConnector:
    """
    Async Notion connector for accessing pages and streaming training samples.

    Supports both mock mode (for local development) and real mode (using Notion API).

    Example:
        async with AsyncNotionConnector() as connector:
            # List pages
            pages = await connector.list_pages(database_id="database_id")

            # Get page content
            content = await connector.get_page_content(page_id="page_id")

            # Stream samples
            mapping = {"input": "Question", "output": "Answer"}
            async for sample in connector.stream_samples("database_id", mapping):
                print(sample)
    """

    BASE_URL = "https://api.notion.com/v1"

    def __init__(self, config: Optional[NotionConfig] = None):
        """
        Initialize the async Notion connector.

        Args:
            config: Configuration object. If None, uses defaults with mock mode.
        """
        self.config = config or NotionConfig()

        # Check environment variables for mock mode override
        env_mock = os.getenv("NOTION_MOCK", "").lower()
        if env_mock in ("true", "1", "yes"):
            self.config.mock_mode = True
        elif env_mock in ("false", "0", "no"):
            self.config.mock_mode = False

        # Also check global connector mock setting
        global_mock = os.getenv("CONNECTOR_MOCK", "").lower()
        if global_mock in ("true", "1", "yes"):
            self.config.mock_mode = True

        # Check for API token in environment
        if not self.config.api_token:
            self.config.api_token = os.getenv("NOTION_API_TOKEN") or os.getenv("NOTION_TOKEN")

        self._client: Optional[httpx.AsyncClient] = None
        logger.debug(f"AsyncNotionConnector initialized (mock_mode={self.config.mock_mode})")

    async def __aenter__(self) -> "AsyncNotionConnector":
        """Async context manager entry."""
        if not self.config.mock_mode:
            await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if not _HTTPX_AVAILABLE:
            raise ImportError(
                "Async Notion connector requires httpx. "
                "Install with: pip install httpx"
            )

        if self._client is None or self._client.is_closed:
            if not self.config.api_token:
                raise RuntimeError(
                    "No Notion API token provided. Either:\n"
                    "- Set NOTION_API_TOKEN environment variable\n"
                    "- Pass api_token in NotionConfig\n"
                    "Or set NOTION_MOCK=true for mock mode."
                )

            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.config.api_token}",
                    "Notion-Version": self.config.api_version,
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
            logger.debug("Created new httpx AsyncClient for Notion")

        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.debug("Closed httpx AsyncClient")

    def _get_samples_dir(self) -> Path:
        """Get the path to the sample files directory."""
        if self.config.samples_dir:
            return Path(self.config.samples_dir)

        # Try relative to this file first
        connector_dir = Path(__file__).parent
        samples_dir = connector_dir.parent / "examples" / "notion_samples"
        if samples_dir.exists():
            return samples_dir

        # Fallback to current working directory
        return Path("examples") / "notion_samples"

    async def list_pages(
        self,
        database_id: Optional[str] = None,
        filter_obj: Optional[Dict] = None,
        sorts: Optional[List[Dict]] = None,
    ) -> List[NotionPage]:
        """
        List pages from a Notion database asynchronously.

        Args:
            database_id: ID of the database to query.
            filter_obj: Optional filter object for the query.
            sorts: Optional list of sort objects.

        Returns:
            List of NotionPage objects.
        """
        if self.config.mock_mode:
            return await self._list_pages_mock(database_id)
        return await self._list_pages_real(database_id, filter_obj, sorts)

    async def _list_pages_mock(self, database_id: Optional[str] = None) -> List[NotionPage]:
        """List pages from mock samples directory."""
        samples_dir = self._get_samples_dir()

        if not samples_dir.exists():
            logger.warning(f"Samples directory not found: {samples_dir}")
            return []

        # Look for a database file
        db_file = samples_dir / f"{database_id}.json" if database_id else None
        if db_file and db_file.exists():
            if _AIOFILES_AVAILABLE:
                async with aiofiles.open(db_file, "r", encoding="utf-8") as f:
                    content = await f.read()
                    data = json.loads(content)
            else:
                with open(db_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            return [self._parse_page(p) for p in data.get("pages", data) if isinstance(p, dict)]

        # Fallback: treat each JSON file as a page
        pages = []
        for file_path in sorted(samples_dir.iterdir()):
            if file_path.is_file() and file_path.suffix == ".json" and not file_path.name.startswith("."):
                try:
                    if _AIOFILES_AVAILABLE:
                        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                            content = await f.read()
                            data = json.loads(content)
                    else:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                    if isinstance(data, dict) and "id" in data:
                        pages.append(self._parse_page(data))
                    elif isinstance(data, list):
                        pages.extend([self._parse_page(p) for p in data if isinstance(p, dict)])
                except (json.JSONDecodeError, KeyError):
                    continue

        return pages

    async def _list_pages_real(
        self,
        database_id: Optional[str] = None,
        filter_obj: Optional[Dict] = None,
        sorts: Optional[List[Dict]] = None,
    ) -> List[NotionPage]:
        """List pages using Notion API asynchronously."""
        if not database_id:
            raise ValueError("database_id is required for real API queries")

        client = await self._get_client()
        pages = []
        start_cursor = None

        while True:
            payload: Dict[str, Any] = {"page_size": self.config.page_size}
            if filter_obj:
                payload["filter"] = filter_obj
            if sorts:
                payload["sorts"] = sorts
            if start_cursor:
                payload["start_cursor"] = start_cursor

            response = await client.post(
                f"/databases/{database_id}/query",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            for result in data.get("results", []):
                pages.append(self._parse_page(result))

            if not data.get("has_more"):
                break
            start_cursor = data.get("next_cursor")

        return pages

    def _parse_page(self, data: Dict[str, Any]) -> NotionPage:
        """Parse page data into NotionPage object."""
        # Extract title from properties
        title = ""
        properties = data.get("properties", {})
        for prop_name, prop_value in properties.items():
            if prop_value.get("type") == "title":
                title_array = prop_value.get("title", [])
                if title_array:
                    title = "".join(t.get("plain_text", "") for t in title_array)
                break

        # If no title property, check for explicit title field
        if not title and "title" in data:
            if isinstance(data["title"], str):
                title = data["title"]
            elif isinstance(data["title"], list):
                title = "".join(t.get("plain_text", "") for t in data["title"])

        # Extract parent info
        parent = data.get("parent", {})
        parent_type = parent.get("type")
        parent_id = parent.get(parent_type) if parent_type else None

        return NotionPage(
            id=data.get("id", ""),
            title=title,
            url=data.get("url"),
            created_time=data.get("created_time"),
            last_edited_time=data.get("last_edited_time"),
            parent_type=parent_type,
            parent_id=parent_id,
            properties=properties,
            archived=data.get("archived", False),
        )

    async def get_page_content(self, page_id: str) -> str:
        """
        Get the text content of a page asynchronously.

        Args:
            page_id: ID of the page to retrieve.

        Returns:
            Text content of the page.

        Raises:
            FileNotFoundError: If page not found (mock mode).
            RuntimeError: If retrieval fails.
        """
        if self.config.mock_mode:
            return await self._get_page_content_mock(page_id)
        return await self._get_page_content_real(page_id)

    async def _get_page_content_mock(self, page_id: str) -> str:
        """Get page content from mock samples."""
        samples_dir = self._get_samples_dir()

        # Try to find the page file
        for ext in [".json", ".txt", ".md"]:
            file_path = samples_dir / f"{page_id}{ext}"
            if file_path.exists():
                if _AIOFILES_AVAILABLE:
                    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                # If JSON, extract content field
                if ext == ".json":
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict):
                            return data.get("content", data.get("text", json.dumps(data)))
                    except json.JSONDecodeError:
                        pass
                return content

        # Search in JSON files for matching page
        for file_path in samples_dir.glob("*.json"):
            try:
                if _AIOFILES_AVAILABLE:
                    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                        data = json.loads(content)
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                if isinstance(data, dict) and data.get("id") == page_id:
                    return data.get("content", data.get("text", ""))
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("id") == page_id:
                            return item.get("content", item.get("text", ""))
            except (json.JSONDecodeError, KeyError):
                continue

        available = [f.stem for f in samples_dir.glob("*.json")]
        raise FileNotFoundError(
            f"Mock page '{page_id}' not found. "
            f"Available samples: {available[:10]}{'...' if len(available) > 10 else ''}"
        )

    async def _get_page_content_real(self, page_id: str) -> str:
        """Get page content using Notion API asynchronously."""
        client = await self._get_client()

        # Get all blocks from the page
        blocks = []
        start_cursor = None

        while True:
            params = {"page_size": 100}
            if start_cursor:
                params["start_cursor"] = start_cursor

            response = await client.get(
                f"/blocks/{page_id}/children",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            blocks.extend(data.get("results", []))

            if not data.get("has_more"):
                break
            start_cursor = data.get("next_cursor")

        # Extract text from blocks
        text_parts = []
        for block in blocks:
            text = self._extract_block_text(block)
            if text:
                text_parts.append(text)

        return "\n\n".join(text_parts)

    def _extract_block_text(self, block: Dict[str, Any], depth: int = 0) -> str:
        """Extract text content from a Notion block."""
        if depth > self.config.max_depth:
            return ""

        block_type = block.get("type", "")
        block_data = block.get(block_type, {})

        # Handle text-based blocks
        text_types = [
            "paragraph", "heading_1", "heading_2", "heading_3",
            "bulleted_list_item", "numbered_list_item", "quote",
            "callout", "toggle",
        ]

        if block_type in text_types:
            rich_text = block_data.get("rich_text", [])
            text = "".join(rt.get("plain_text", "") for rt in rich_text)

            # Add prefix for headings
            if block_type == "heading_1":
                text = f"# {text}"
            elif block_type == "heading_2":
                text = f"## {text}"
            elif block_type == "heading_3":
                text = f"### {text}"
            elif block_type == "bulleted_list_item":
                text = f"- {text}"
            elif block_type == "numbered_list_item":
                text = f"1. {text}"
            elif block_type == "quote":
                text = f"> {text}"

            return text

        # Handle code blocks
        if block_type == "code":
            rich_text = block_data.get("rich_text", [])
            code = "".join(rt.get("plain_text", "") for rt in rich_text)
            language = block_data.get("language", "")
            return f"```{language}\n{code}\n```"

        # Handle divider
        if block_type == "divider":
            return "---"

        return ""

    async def get_database(self, database_id: str) -> NotionDatabase:
        """
        Get database metadata asynchronously.

        Args:
            database_id: ID of the database.

        Returns:
            NotionDatabase object.
        """
        if self.config.mock_mode:
            return await self._get_database_mock(database_id)
        return await self._get_database_real(database_id)

    async def _get_database_mock(self, database_id: str) -> NotionDatabase:
        """Get database metadata from mock samples."""
        samples_dir = self._get_samples_dir()

        # Try to find database file
        db_file = samples_dir / f"{database_id}.json"
        if db_file.exists():
            if _AIOFILES_AVAILABLE:
                async with aiofiles.open(db_file, "r", encoding="utf-8") as f:
                    content = await f.read()
                    data = json.loads(content)
            else:
                with open(db_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

            return NotionDatabase(
                id=database_id,
                title=data.get("title", database_id),
                properties=data.get("properties", {}),
            )

        # Return a mock database
        return NotionDatabase(
            id=database_id,
            title=f"Mock Database {database_id}",
            properties={},
        )

    async def _get_database_real(self, database_id: str) -> NotionDatabase:
        """Get database metadata using Notion API asynchronously."""
        client = await self._get_client()

        response = await client.get(f"/databases/{database_id}")
        response.raise_for_status()
        data = response.json()

        # Extract title
        title_array = data.get("title", [])
        title = "".join(t.get("plain_text", "") for t in title_array)

        return NotionDatabase(
            id=data.get("id", database_id),
            title=title,
            url=data.get("url"),
            created_time=data.get("created_time"),
            last_edited_time=data.get("last_edited_time"),
            properties=data.get("properties", {}),
        )

    async def stream_samples(
        self,
        database_id: str,
        mapping: Dict[str, str],
        filter_obj: Optional[Dict] = None,
        include_content: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream training samples from a Notion database asynchronously.

        Each page in the database is converted to a training sample using
        the property mapping.

        Args:
            database_id: ID of the database to query.
            mapping: Property mapping dict with "input" and "output" keys.
                     Values should be Notion property names.
            filter_obj: Optional filter for the database query.
            include_content: If True, include page content in metadata.

        Yields:
            Training sample dicts with "input", "output", and "metadata" keys.
        """
        pages = await self.list_pages(database_id, filter_obj)

        for page in pages:
            try:
                # Extract values using mapping
                row = {}
                for sample_key, prop_name in mapping.items():
                    value = self._extract_property_value(page.properties, prop_name)
                    if value is None and prop_name.lower() == "title":
                        value = page.title
                    row[sample_key] = value or ""

                sample = row_to_sample(row, {"input": "input", "output": "output"})
                sample["metadata"]["source"] = "notion"
                sample["metadata"]["page_id"] = page.id
                sample["metadata"]["page_title"] = page.title
                sample["metadata"]["page_url"] = page.url

                # Optionally include page content
                if include_content:
                    try:
                        sample["metadata"]["content"] = await self.get_page_content(page.id)
                    except Exception as e:
                        logger.warning(f"Failed to get content for page {page.id}: {e}")

                yield sample

            except Exception as e:
                logger.warning(f"Error processing page {page.id}: {e}")
                continue

    def _extract_property_value(
        self,
        properties: Dict[str, Any],
        prop_name: str,
    ) -> Optional[str]:
        """Extract a string value from a Notion property."""
        if prop_name not in properties:
            return None

        prop = properties[prop_name]
        prop_type = prop.get("type", "")

        # Handle different property types
        if prop_type == "title":
            title_array = prop.get("title", [])
            return "".join(t.get("plain_text", "") for t in title_array)

        if prop_type == "rich_text":
            rich_text = prop.get("rich_text", [])
            return "".join(t.get("plain_text", "") for t in rich_text)

        if prop_type == "number":
            return str(prop.get("number", ""))

        if prop_type == "select":
            select = prop.get("select")
            return select.get("name") if select else None

        if prop_type == "multi_select":
            items = prop.get("multi_select", [])
            return ", ".join(item.get("name", "") for item in items)

        if prop_type == "checkbox":
            return str(prop.get("checkbox", False))

        if prop_type == "url":
            return prop.get("url")

        if prop_type == "email":
            return prop.get("email")

        if prop_type == "phone_number":
            return prop.get("phone_number")

        if prop_type == "date":
            date = prop.get("date")
            if date:
                return date.get("start", "")
            return None

        if prop_type == "formula":
            formula = prop.get("formula", {})
            formula_type = formula.get("type")
            return str(formula.get(formula_type, "")) if formula_type else None

        if prop_type == "rollup":
            rollup = prop.get("rollup", {})
            rollup_type = rollup.get("type")
            return str(rollup.get(rollup_type, "")) if rollup_type else None

        return None

    async def fetch_pages_concurrent(
        self,
        page_ids: List[str],
        max_concurrency: int = 5,
    ) -> Dict[str, str]:
        """
        Fetch content for multiple pages concurrently.

        Args:
            page_ids: List of page IDs to fetch.
            max_concurrency: Maximum number of concurrent requests.

        Returns:
            Dictionary mapping page_id to content.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def fetch_one(page_id: str) -> tuple:
            async with semaphore:
                try:
                    content = await self.get_page_content(page_id)
                    return page_id, content
                except Exception as e:
                    logger.error(f"Failed to fetch page {page_id}: {e}")
                    return page_id, None

        tasks = [fetch_one(pid) for pid in page_ids]
        results = await asyncio.gather(*tasks)

        return {pid: content for pid, content in results if content is not None}
