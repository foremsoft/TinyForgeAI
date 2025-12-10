"""
Notion connector for TinyForgeAI.

Provides functionality to list pages, get content, and stream training
samples from Notion databases and pages.

Mock Mode:
    When NOTION_MOCK=true (default), the connector reads from local sample
    files in examples/notion_samples/ instead of making real API calls.

Real Mode:
    When NOTION_MOCK=false, the connector uses the Notion API with an
    integration token to access real pages and databases.

Usage:
    from connectors.notion_connector import NotionConnector

    # Create connector (uses mock mode by default)
    connector = NotionConnector()

    # List pages in a database
    pages = connector.list_pages(database_id="my_database")

    # Get page content
    content = connector.get_page_content(page_id="my_page")

    # Stream training samples from database
    for sample in connector.stream_samples(database_id="my_database", mapping={...}):
        print(sample)
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# Add project root to path when run as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from connectors.mappers import row_to_sample

logger = logging.getLogger(__name__)

# Check for requests library
REQUESTS_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class NotionConfig:
    """Configuration for Notion connector."""

    # Authentication
    api_token: Optional[str] = None  # Notion integration token
    api_version: str = "2022-06-28"  # Notion API version

    # Mock mode settings
    mock_mode: bool = True  # Use mock mode by default
    samples_dir: Optional[str] = None  # Custom samples directory

    # Query settings
    page_size: int = 100  # Items per page when listing

    # Content extraction
    include_children: bool = True  # Include child blocks when getting content
    max_depth: int = 3  # Maximum depth for nested blocks


@dataclass
class NotionPage:
    """Represents a page in Notion."""

    id: str
    title: str
    url: Optional[str] = None
    created_time: Optional[str] = None
    last_edited_time: Optional[str] = None
    parent_type: Optional[str] = None  # database_id, page_id, or workspace
    parent_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    archived: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "created_time": self.created_time,
            "last_edited_time": self.last_edited_time,
            "parent_type": self.parent_type,
            "parent_id": self.parent_id,
            "properties": self.properties,
            "archived": self.archived,
        }


@dataclass
class NotionDatabase:
    """Represents a database in Notion."""

    id: str
    title: str
    url: Optional[str] = None
    created_time: Optional[str] = None
    last_edited_time: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "created_time": self.created_time,
            "last_edited_time": self.last_edited_time,
            "properties": self.properties,
        }


class NotionConnector:
    """
    Notion connector for accessing pages and streaming training samples.

    Example:
        connector = NotionConnector()

        # List pages
        pages = connector.list_pages(database_id="database_id")

        # Get page content
        content = connector.get_page_content(page_id="page_id")

        # Stream samples
        mapping = {"input": "Question", "output": "Answer"}
        for sample in connector.stream_samples("database_id", mapping):
            print(sample)
    """

    BASE_URL = "https://api.notion.com/v1"

    def __init__(self, config: Optional[NotionConfig] = None):
        """
        Initialize the Notion connector.

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

        self._session = None
        logger.debug(f"NotionConnector initialized (mock_mode={self.config.mock_mode})")

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

    def _get_session(self) -> "requests.Session":
        """Get or create a requests session with authentication."""
        if self._session is not None:
            return self._session

        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library not installed. "
                "Install with: pip install requests"
            )

        if not self.config.api_token:
            raise RuntimeError(
                "No Notion API token provided. Either:\n"
                "- Set NOTION_API_TOKEN environment variable\n"
                "- Pass api_token in NotionConfig\n"
                "Or set NOTION_MOCK=true for mock mode."
            )

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.config.api_token}",
            "Notion-Version": self.config.api_version,
            "Content-Type": "application/json",
        })
        return self._session

    def list_pages(
        self,
        database_id: Optional[str] = None,
        filter_obj: Optional[Dict] = None,
        sorts: Optional[List[Dict]] = None,
    ) -> List[NotionPage]:
        """
        List pages from a Notion database.

        Args:
            database_id: ID of the database to query.
            filter_obj: Optional filter object for the query.
            sorts: Optional list of sort objects.

        Returns:
            List of NotionPage objects.
        """
        if self.config.mock_mode:
            return self._list_pages_mock(database_id)
        return self._list_pages_real(database_id, filter_obj, sorts)

    def _list_pages_mock(self, database_id: Optional[str] = None) -> List[NotionPage]:
        """List pages from mock samples directory."""
        samples_dir = self._get_samples_dir()

        if not samples_dir.exists():
            logger.warning(f"Samples directory not found: {samples_dir}")
            return []

        # Look for a database file
        db_file = samples_dir / f"{database_id}.json" if database_id else None
        if db_file and db_file.exists():
            with open(db_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [self._parse_page(p) for p in data.get("pages", data) if isinstance(p, dict)]

        # Fallback: treat each JSON file as a page
        pages = []
        for file_path in sorted(samples_dir.iterdir()):
            if file_path.is_file() and file_path.suffix == ".json" and not file_path.name.startswith("."):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "id" in data:
                            pages.append(self._parse_page(data))
                        elif isinstance(data, list):
                            pages.extend([self._parse_page(p) for p in data if isinstance(p, dict)])
                except (json.JSONDecodeError, KeyError):
                    continue

        return pages

    def _list_pages_real(
        self,
        database_id: Optional[str] = None,
        filter_obj: Optional[Dict] = None,
        sorts: Optional[List[Dict]] = None,
    ) -> List[NotionPage]:
        """List pages using Notion API."""
        if not database_id:
            raise ValueError("database_id is required for real API queries")

        session = self._get_session()
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

            response = session.post(
                f"{self.BASE_URL}/databases/{database_id}/query",
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

    def get_page_content(self, page_id: str) -> str:
        """
        Get the text content of a page.

        Args:
            page_id: ID of the page to retrieve.

        Returns:
            Text content of the page.

        Raises:
            FileNotFoundError: If page not found (mock mode).
            RuntimeError: If retrieval fails.
        """
        if self.config.mock_mode:
            return self._get_page_content_mock(page_id)
        return self._get_page_content_real(page_id)

    def _get_page_content_mock(self, page_id: str) -> str:
        """Get page content from mock samples."""
        samples_dir = self._get_samples_dir()

        # Try to find the page file
        for ext in [".json", ".txt", ".md"]:
            file_path = samples_dir / f"{page_id}{ext}"
            if file_path.exists():
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

    def _get_page_content_real(self, page_id: str) -> str:
        """Get page content using Notion API."""
        session = self._get_session()

        # Get all blocks from the page
        blocks = []
        start_cursor = None

        while True:
            url = f"{self.BASE_URL}/blocks/{page_id}/children"
            params = {"page_size": 100}
            if start_cursor:
                params["start_cursor"] = start_cursor

            response = session.get(url, params=params)
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

    def get_database(self, database_id: str) -> NotionDatabase:
        """
        Get database metadata.

        Args:
            database_id: ID of the database.

        Returns:
            NotionDatabase object.
        """
        if self.config.mock_mode:
            return self._get_database_mock(database_id)
        return self._get_database_real(database_id)

    def _get_database_mock(self, database_id: str) -> NotionDatabase:
        """Get database metadata from mock samples."""
        samples_dir = self._get_samples_dir()

        # Try to find database file
        db_file = samples_dir / f"{database_id}.json"
        if db_file.exists():
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

    def _get_database_real(self, database_id: str) -> NotionDatabase:
        """Get database metadata using Notion API."""
        session = self._get_session()

        response = session.get(f"{self.BASE_URL}/databases/{database_id}")
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

    def stream_samples(
        self,
        database_id: str,
        mapping: Dict[str, str],
        filter_obj: Optional[Dict] = None,
        include_content: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream training samples from a Notion database.

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
        pages = self.list_pages(database_id, filter_obj)

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
                        sample["metadata"]["content"] = self.get_page_content(page.id)
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

        return None


def main() -> int:
    """CLI entry point for the Notion connector."""
    parser = argparse.ArgumentParser(
        description="Access pages and databases from Notion (or mock samples)."
    )
    parser.add_argument(
        "--database-id",
        help="Notion database ID to query.",
    )
    parser.add_argument(
        "--page-id",
        help="Notion page ID to retrieve.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List pages in database.",
    )
    parser.add_argument(
        "--content",
        action="store_true",
        help="Get page content.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream training samples from database.",
    )
    parser.add_argument(
        "--mapping",
        default='{"input": "Question", "output": "Answer"}',
        help='JSON mapping string for streaming samples.',
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples to output.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode.",
    )

    args = parser.parse_args()

    config = NotionConfig(mock_mode=args.mock or True)
    connector = NotionConnector(config)

    try:
        if args.list:
            pages = connector.list_pages(database_id=args.database_id)
            for p in pages:
                print(json.dumps(p.to_dict()))
            return 0

        if args.page_id and args.content:
            content = connector.get_page_content(args.page_id)
            print(content)
            return 0

        if args.stream:
            if not args.database_id:
                print("Error: --database-id required for streaming", file=sys.stderr)
                return 1

            mapping = json.loads(args.mapping)
            count = 0
            for sample in connector.stream_samples(args.database_id, mapping):
                print(json.dumps(sample))
                count += 1
                if args.limit and count >= args.limit:
                    break
            return 0

        # Default: show help
        parser.print_help()
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
