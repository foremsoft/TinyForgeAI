"""
Confluence connector for TinyForgeAI.

Provides functionality to list spaces, get pages, and stream training
samples from Atlassian Confluence wikis.

Mock Mode:
    When CONFLUENCE_MOCK=true (default), the connector reads from local sample
    files in examples/confluence_samples/ instead of making real API calls.

Real Mode:
    When CONFLUENCE_MOCK=false, the connector uses the Confluence REST API with
    API token authentication to access real spaces and pages.

Usage:
    from connectors.confluence_connector import ConfluenceConnector

    # Create connector (uses mock mode by default)
    connector = ConfluenceConnector()

    # List spaces
    spaces = connector.list_spaces()

    # Get pages in a space
    pages = connector.list_pages(space_key="DOCS")

    # Get page content
    content = connector.get_page_content(page_id="123456")

    # Stream training samples
    for sample in connector.stream_samples(space_key="DOCS", mapping={...}):
        print(sample)
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from html.parser import HTMLParser

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


class HTMLTextExtractor(HTMLParser):
    """Simple HTML parser to extract text content."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._skip_data = False
        self._skip_tags = {"script", "style", "noscript"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip_data = True
        elif tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip_data = False

    def handle_data(self, data):
        if not self._skip_data:
            self.text_parts.append(data)

    def get_text(self) -> str:
        text = "".join(self.text_parts)
        # Normalize whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()


def html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    parser = HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


@dataclass
class ConfluenceConfig:
    """Configuration for Confluence connector."""

    # Authentication
    base_url: Optional[str] = None  # Confluence base URL (e.g., https://your-domain.atlassian.net/wiki)
    username: Optional[str] = None  # Email for Atlassian Cloud
    api_token: Optional[str] = None  # API token

    # Mock mode settings
    mock_mode: bool = True  # Use mock mode by default
    samples_dir: Optional[str] = None  # Custom samples directory

    # Query settings
    limit: int = 25  # Items per page (max 100 for Confluence)
    expand: List[str] = field(default_factory=lambda: ["body.storage", "version", "ancestors"])

    # Content settings
    include_archived: bool = False
    include_attachments: bool = False


@dataclass
class ConfluenceSpace:
    """Represents a Confluence space."""

    key: str
    name: str
    id: Optional[str] = None
    type: Optional[str] = None  # global, personal
    status: Optional[str] = None  # current, archived
    description: Optional[str] = None
    homepage_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "name": self.name,
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "description": self.description,
            "homepage_id": self.homepage_id,
        }


@dataclass
class ConfluencePage:
    """Represents a Confluence page."""

    id: str
    title: str
    space_key: Optional[str] = None
    type: str = "page"  # page, blogpost
    status: str = "current"
    version: int = 1
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    created_by: Optional[str] = None
    modified_by: Optional[str] = None
    parent_id: Optional[str] = None
    ancestors: List[str] = field(default_factory=list)
    url: Optional[str] = None
    body_html: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "space_key": self.space_key,
            "type": self.type,
            "status": self.status,
            "version": self.version,
            "created_date": self.created_date,
            "modified_date": self.modified_date,
            "created_by": self.created_by,
            "modified_by": self.modified_by,
            "parent_id": self.parent_id,
            "ancestors": self.ancestors,
            "url": self.url,
        }


class ConfluenceConnector:
    """
    Confluence connector for accessing pages and streaming training samples.

    Example:
        connector = ConfluenceConnector()

        # List spaces
        spaces = connector.list_spaces()

        # List pages
        pages = connector.list_pages(space_key="DOCS")

        # Get page content
        content = connector.get_page_content(page_id="123456")

        # Stream samples
        mapping = {"input": "title", "output": "content"}
        for sample in connector.stream_samples("DOCS", mapping):
            print(sample)
    """

    def __init__(self, config: Optional[ConfluenceConfig] = None):
        """
        Initialize the Confluence connector.

        Args:
            config: Configuration object. If None, uses defaults with mock mode.
        """
        self.config = config or ConfluenceConfig()

        # Check environment variables for mock mode override
        env_mock = os.getenv("CONFLUENCE_MOCK", "").lower()
        if env_mock in ("true", "1", "yes"):
            self.config.mock_mode = True
        elif env_mock in ("false", "0", "no"):
            self.config.mock_mode = False

        # Also check global connector mock setting
        global_mock = os.getenv("CONNECTOR_MOCK", "").lower()
        if global_mock in ("true", "1", "yes"):
            self.config.mock_mode = True

        # Check for credentials in environment
        if not self.config.base_url:
            self.config.base_url = os.getenv("CONFLUENCE_URL") or os.getenv("CONFLUENCE_BASE_URL")
        if not self.config.username:
            self.config.username = os.getenv("CONFLUENCE_USERNAME") or os.getenv("CONFLUENCE_EMAIL")
        if not self.config.api_token:
            self.config.api_token = os.getenv("CONFLUENCE_API_TOKEN") or os.getenv("CONFLUENCE_TOKEN")

        self._session = None
        logger.debug(f"ConfluenceConnector initialized (mock_mode={self.config.mock_mode})")

    def _get_samples_dir(self) -> Path:
        """Get the path to the sample files directory."""
        if self.config.samples_dir:
            return Path(self.config.samples_dir)

        # Try relative to this file first
        connector_dir = Path(__file__).parent
        samples_dir = connector_dir.parent / "examples" / "confluence_samples"
        if samples_dir.exists():
            return samples_dir

        # Fallback to current working directory
        return Path("examples") / "confluence_samples"

    def _get_session(self) -> "requests.Session":
        """Get or create a requests session with authentication."""
        if self._session is not None:
            return self._session

        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library not installed. "
                "Install with: pip install requests"
            )

        if not self.config.base_url:
            raise RuntimeError(
                "No Confluence URL provided. Either:\n"
                "- Set CONFLUENCE_URL environment variable\n"
                "- Pass base_url in ConfluenceConfig\n"
                "Or set CONFLUENCE_MOCK=true for mock mode."
            )

        if not self.config.username or not self.config.api_token:
            raise RuntimeError(
                "No Confluence credentials provided. Either:\n"
                "- Set CONFLUENCE_USERNAME and CONFLUENCE_API_TOKEN environment variables\n"
                "- Pass username and api_token in ConfluenceConfig\n"
                "Or set CONFLUENCE_MOCK=true for mock mode."
            )

        self._session = requests.Session()
        self._session.auth = (self.config.username, self.config.api_token)
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        return self._session

    def _api_url(self, endpoint: str) -> str:
        """Build API URL."""
        base = self.config.base_url.rstrip("/")
        return f"{base}/rest/api/{endpoint}"

    def list_spaces(
        self,
        type_filter: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ConfluenceSpace]:
        """
        List Confluence spaces.

        Args:
            type_filter: Filter by space type (global, personal).
            status: Filter by status (current, archived).

        Returns:
            List of ConfluenceSpace objects.
        """
        if self.config.mock_mode:
            return self._list_spaces_mock(type_filter)
        return self._list_spaces_real(type_filter, status)

    def _list_spaces_mock(self, type_filter: Optional[str] = None) -> List[ConfluenceSpace]:
        """List spaces from mock samples directory."""
        samples_dir = self._get_samples_dir()

        if not samples_dir.exists():
            logger.warning(f"Samples directory not found: {samples_dir}")
            return []

        # Look for spaces.json file
        spaces_file = samples_dir / "spaces.json"
        if spaces_file.exists():
            with open(spaces_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                spaces = data if isinstance(data, list) else data.get("results", [])
                parsed = [self._parse_space(s) for s in spaces]
                if type_filter:
                    parsed = [s for s in parsed if s.type == type_filter]
                return parsed

        # Fallback: treat each subdirectory as a space
        spaces = []
        for subdir in sorted(samples_dir.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith("."):
                spaces.append(ConfluenceSpace(
                    key=subdir.name.upper(),
                    name=subdir.name.replace("_", " ").title(),
                    type="global",
                ))

        return spaces

    def _list_spaces_real(
        self,
        type_filter: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ConfluenceSpace]:
        """List spaces using Confluence API."""
        session = self._get_session()
        spaces = []
        start = 0

        while True:
            params = {
                "start": start,
                "limit": self.config.limit,
            }
            if type_filter:
                params["type"] = type_filter
            if status:
                params["status"] = status

            response = session.get(self._api_url("space"), params=params)
            response.raise_for_status()
            data = response.json()

            for space_data in data.get("results", []):
                spaces.append(self._parse_space(space_data))

            # Check for more results
            if data.get("size", 0) < self.config.limit:
                break
            start += self.config.limit

        return spaces

    def _parse_space(self, data: Dict[str, Any]) -> ConfluenceSpace:
        """Parse space data into ConfluenceSpace object."""
        description = data.get("description", {})
        if isinstance(description, dict):
            description = description.get("plain", {}).get("value", "")

        homepage = data.get("homepage", {})
        homepage_id = homepage.get("id") if isinstance(homepage, dict) else None

        return ConfluenceSpace(
            key=data.get("key", ""),
            name=data.get("name", ""),
            id=data.get("id"),
            type=data.get("type"),
            status=data.get("status"),
            description=description,
            homepage_id=homepage_id,
        )

    def list_pages(
        self,
        space_key: Optional[str] = None,
        title_filter: Optional[str] = None,
        page_type: str = "page",
    ) -> List[ConfluencePage]:
        """
        List pages in a space.

        Args:
            space_key: Key of the space to list pages from.
            title_filter: Filter pages by title (contains).
            page_type: Type of content (page, blogpost).

        Returns:
            List of ConfluencePage objects.
        """
        if self.config.mock_mode:
            return self._list_pages_mock(space_key, title_filter)
        return self._list_pages_real(space_key, title_filter, page_type)

    def _list_pages_mock(
        self,
        space_key: Optional[str] = None,
        title_filter: Optional[str] = None,
    ) -> List[ConfluencePage]:
        """List pages from mock samples."""
        samples_dir = self._get_samples_dir()

        # Try to find pages file in space directory
        if space_key:
            space_dir = samples_dir / space_key.lower()
            if space_dir.exists():
                pages_file = space_dir / "pages.json"
                if pages_file.exists():
                    with open(pages_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        pages = data if isinstance(data, list) else data.get("results", [])
                        parsed = [self._parse_page(p, space_key) for p in pages]
                        if title_filter:
                            parsed = [p for p in parsed if title_filter.lower() in p.title.lower()]
                        return parsed

                # Fallback: each JSON file is a page
                pages = []
                for json_file in space_dir.glob("*.json"):
                    if json_file.name != "pages.json":
                        with open(json_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, dict) and "id" in data:
                                pages.append(self._parse_page(data, space_key))
                return pages

        # No space_key: search all spaces
        pages = []
        for subdir in samples_dir.iterdir():
            if subdir.is_dir():
                pages.extend(self._list_pages_mock(subdir.name.upper(), title_filter))

        return pages

    def _list_pages_real(
        self,
        space_key: Optional[str] = None,
        title_filter: Optional[str] = None,
        page_type: str = "page",
    ) -> List[ConfluencePage]:
        """List pages using Confluence API."""
        session = self._get_session()
        pages = []
        start = 0

        # Build CQL query
        cql_parts = [f"type={page_type}"]
        if space_key:
            cql_parts.append(f"space={space_key}")
        if title_filter:
            cql_parts.append(f"title~\"{title_filter}\"")

        cql = " AND ".join(cql_parts)

        while True:
            params = {
                "cql": cql,
                "start": start,
                "limit": self.config.limit,
                "expand": ",".join(self.config.expand),
            }

            response = session.get(self._api_url("content/search"), params=params)
            response.raise_for_status()
            data = response.json()

            for page_data in data.get("results", []):
                pages.append(self._parse_page(page_data, space_key))

            # Check for more results
            if data.get("size", 0) < self.config.limit:
                break
            start += self.config.limit

        return pages

    def _parse_page(self, data: Dict[str, Any], space_key: Optional[str] = None) -> ConfluencePage:
        """Parse page data into ConfluencePage object."""
        # Extract version info
        version = data.get("version", {})
        version_num = version.get("number", 1) if isinstance(version, dict) else 1

        # Extract history/dates
        history = data.get("history", {})
        created_date = history.get("createdDate")
        created_by = history.get("createdBy", {}).get("displayName")

        # Get last update info from version
        modified_date = version.get("when") if isinstance(version, dict) else None
        modified_by = version.get("by", {}).get("displayName") if isinstance(version, dict) else None

        # Extract ancestors
        ancestors = data.get("ancestors", [])
        ancestor_ids = [a.get("id") for a in ancestors if isinstance(a, dict)]

        # Get parent
        parent_id = ancestor_ids[-1] if ancestor_ids else None

        # Get body
        body = data.get("body", {})
        body_storage = body.get("storage", {}) if isinstance(body, dict) else {}
        body_html = body_storage.get("value", "") if isinstance(body_storage, dict) else ""

        # Get space key from data if not provided
        if not space_key:
            space = data.get("space", {})
            space_key = space.get("key") if isinstance(space, dict) else None

        # Build URL
        base_links = data.get("_links", {})
        web_ui = base_links.get("webui", "")

        return ConfluencePage(
            id=data.get("id", ""),
            title=data.get("title", ""),
            space_key=space_key,
            type=data.get("type", "page"),
            status=data.get("status", "current"),
            version=version_num,
            created_date=created_date,
            modified_date=modified_date,
            created_by=created_by,
            modified_by=modified_by,
            parent_id=parent_id,
            ancestors=ancestor_ids,
            url=web_ui,
            body_html=body_html,
        )

    def get_page_content(self, page_id: str, format: str = "text") -> str:
        """
        Get the content of a page.

        Args:
            page_id: ID of the page to retrieve.
            format: Output format (text, html).

        Returns:
            Page content as text or HTML.

        Raises:
            FileNotFoundError: If page not found (mock mode).
            RuntimeError: If retrieval fails.
        """
        if self.config.mock_mode:
            return self._get_page_content_mock(page_id, format)
        return self._get_page_content_real(page_id, format)

    def _get_page_content_mock(self, page_id: str, format: str = "text") -> str:
        """Get page content from mock samples."""
        samples_dir = self._get_samples_dir()

        # Search for page file
        for subdir in samples_dir.iterdir():
            if subdir.is_dir():
                # Try JSON file
                page_file = subdir / f"{page_id}.json"
                if page_file.exists():
                    with open(page_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        html = data.get("body", {}).get("storage", {}).get("value", "")
                        if not html:
                            html = data.get("content", data.get("body", ""))
                        if format == "html":
                            return html
                        return html_to_text(html)

                # Try text/markdown file
                for ext in [".txt", ".md", ".html"]:
                    content_file = subdir / f"{page_id}{ext}"
                    if content_file.exists():
                        with open(content_file, "r", encoding="utf-8") as f:
                            content = f.read()
                            if format == "text" and ext == ".html":
                                return html_to_text(content)
                            return content

                # Search in pages.json
                pages_file = subdir / "pages.json"
                if pages_file.exists():
                    with open(pages_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        pages = data if isinstance(data, list) else data.get("results", [])
                        for page in pages:
                            if str(page.get("id")) == str(page_id):
                                html = page.get("body", {}).get("storage", {}).get("value", "")
                                if not html:
                                    html = page.get("content", "")
                                if format == "html":
                                    return html
                                return html_to_text(html)

        raise FileNotFoundError(f"Mock page '{page_id}' not found")

    def _get_page_content_real(self, page_id: str, format: str = "text") -> str:
        """Get page content using Confluence API."""
        session = self._get_session()

        params = {"expand": "body.storage"}
        response = session.get(self._api_url(f"content/{page_id}"), params=params)
        response.raise_for_status()
        data = response.json()

        body = data.get("body", {})
        storage = body.get("storage", {})
        html = storage.get("value", "")

        if format == "html":
            return html
        return html_to_text(html)

    def get_page(self, page_id: str) -> ConfluencePage:
        """
        Get page metadata.

        Args:
            page_id: ID of the page.

        Returns:
            ConfluencePage object.
        """
        if self.config.mock_mode:
            return self._get_page_mock(page_id)
        return self._get_page_real(page_id)

    def _get_page_mock(self, page_id: str) -> ConfluencePage:
        """Get page from mock samples."""
        samples_dir = self._get_samples_dir()

        for subdir in samples_dir.iterdir():
            if subdir.is_dir():
                # Try JSON file
                page_file = subdir / f"{page_id}.json"
                if page_file.exists():
                    with open(page_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return self._parse_page(data, subdir.name.upper())

                # Search in pages.json
                pages_file = subdir / "pages.json"
                if pages_file.exists():
                    with open(pages_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        pages = data if isinstance(data, list) else data.get("results", [])
                        for page in pages:
                            if str(page.get("id")) == str(page_id):
                                return self._parse_page(page, subdir.name.upper())

        raise FileNotFoundError(f"Mock page '{page_id}' not found")

    def _get_page_real(self, page_id: str) -> ConfluencePage:
        """Get page using Confluence API."""
        session = self._get_session()

        params = {"expand": ",".join(self.config.expand)}
        response = session.get(self._api_url(f"content/{page_id}"), params=params)
        response.raise_for_status()
        data = response.json()

        return self._parse_page(data)

    def stream_samples(
        self,
        space_key: str,
        mapping: Dict[str, str],
        include_content: bool = True,
        title_filter: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream training samples from Confluence pages.

        Each page is converted to a training sample using the field mapping.

        Args:
            space_key: Key of the space to stream from.
            mapping: Field mapping dict with "input" and "output" keys.
            include_content: If True, include page content in samples.
            title_filter: Optional title filter.

        Yields:
            Training sample dicts with "input", "output", and "metadata" keys.
        """
        pages = self.list_pages(space_key, title_filter)

        for page in pages:
            try:
                # Build row from page data
                row = {
                    "title": page.title,
                    "id": page.id,
                    "space": space_key,
                    "url": page.url or "",
                }

                # Get content if requested
                if include_content:
                    try:
                        content = self.get_page_content(page.id)
                        row["content"] = content
                    except Exception as e:
                        logger.warning(f"Failed to get content for page {page.id}: {e}")
                        row["content"] = ""
                else:
                    row["content"] = ""

                # Map fields
                sample_row = {}
                for sample_key, source_field in mapping.items():
                    sample_row[sample_key] = row.get(source_field, "")

                sample = row_to_sample(sample_row, {"input": "input", "output": "output"})
                sample["metadata"]["source"] = "confluence"
                sample["metadata"]["page_id"] = page.id
                sample["metadata"]["page_title"] = page.title
                sample["metadata"]["space_key"] = space_key
                sample["metadata"]["url"] = page.url

                yield sample

            except Exception as e:
                logger.warning(f"Error processing page {page.id}: {e}")
                continue

    def search(
        self,
        query: str,
        space_key: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ConfluencePage]:
        """
        Search for pages using CQL or text search.

        Args:
            query: Search query (CQL or text).
            space_key: Optional space to limit search.
            limit: Maximum results to return.

        Returns:
            List of matching ConfluencePage objects.
        """
        if self.config.mock_mode:
            return self._search_mock(query, space_key, limit)
        return self._search_real(query, space_key, limit)

    def _search_mock(
        self,
        query: str,
        space_key: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ConfluencePage]:
        """Search pages in mock samples."""
        pages = self.list_pages(space_key)
        query_lower = query.lower()

        # Simple text search
        matches = []
        for page in pages:
            if query_lower in page.title.lower():
                matches.append(page)
            elif page.body_html and query_lower in page.body_html.lower():
                matches.append(page)

        if limit:
            matches = matches[:limit]

        return matches

    def _search_real(
        self,
        query: str,
        space_key: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ConfluencePage]:
        """Search using Confluence API."""
        session = self._get_session()

        # Build CQL query
        cql = f"text~\"{query}\""
        if space_key:
            cql = f"space={space_key} AND {cql}"

        params = {
            "cql": cql,
            "limit": limit or self.config.limit,
            "expand": ",".join(self.config.expand),
        }

        response = session.get(self._api_url("content/search"), params=params)
        response.raise_for_status()
        data = response.json()

        return [self._parse_page(p) for p in data.get("results", [])]


def main() -> int:
    """CLI entry point for the Confluence connector."""
    parser = argparse.ArgumentParser(
        description="Access pages and spaces from Confluence (or mock samples)."
    )
    parser.add_argument(
        "--space-key",
        help="Confluence space key.",
    )
    parser.add_argument(
        "--page-id",
        help="Confluence page ID.",
    )
    parser.add_argument(
        "--list-spaces",
        action="store_true",
        help="List accessible spaces.",
    )
    parser.add_argument(
        "--list-pages",
        action="store_true",
        help="List pages in a space.",
    )
    parser.add_argument(
        "--content",
        action="store_true",
        help="Get page content.",
    )
    parser.add_argument(
        "--search",
        help="Search for pages.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream training samples from space.",
    )
    parser.add_argument(
        "--mapping",
        default='{"input": "title", "output": "content"}',
        help='JSON mapping string for streaming samples.',
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of results.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode.",
    )

    args = parser.parse_args()

    config = ConfluenceConfig(mock_mode=args.mock or True)
    connector = ConfluenceConnector(config)

    try:
        if args.list_spaces:
            spaces = connector.list_spaces()
            for space in spaces:
                print(json.dumps(space.to_dict()))
            return 0

        if args.list_pages:
            pages = connector.list_pages(space_key=args.space_key)
            count = 0
            for page in pages:
                print(json.dumps(page.to_dict()))
                count += 1
                if args.limit and count >= args.limit:
                    break
            return 0

        if args.page_id and args.content:
            content = connector.get_page_content(args.page_id)
            print(content)
            return 0

        if args.search:
            results = connector.search(args.search, args.space_key, args.limit)
            for page in results:
                print(json.dumps(page.to_dict()))
            return 0

        if args.stream:
            if not args.space_key:
                print("Error: --space-key required for streaming", file=sys.stderr)
                return 1

            mapping = json.loads(args.mapping)
            count = 0
            for sample in connector.stream_samples(args.space_key, mapping):
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
