"""
URL Content Extractor.

Extracts and converts content from various URL sources into training data.
Supports automatic detection of content type and Q&A pair extraction.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import html

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


@dataclass
class ExtractedData:
    """Represents extracted training data from a URL."""
    url: str
    source_type: str  # notion, google_docs, github, website, etc.
    title: Optional[str] = None
    samples: List[Dict[str, str]] = field(default_factory=list)
    raw_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class URLExtractor:
    """
    Extract training data from URLs.

    Supports:
    - Notion pages and databases
    - Google Docs/Sheets
    - GitHub README files
    - Generic websites (FAQ pages, documentation)
    - Raw JSONL/JSON/CSV files

    Usage:
        extractor = URLExtractor()
        data = extractor.extract("https://notion.so/my-workspace/FAQ-123456")
        print(f"Extracted {len(data.samples)} training samples")
    """

    # URL patterns for different sources
    URL_PATTERNS = {
        "notion": r"notion\.so|notion\.site",
        "google_docs": r"docs\.google\.com/document",
        "google_sheets": r"docs\.google\.com/spreadsheets",
        "github": r"github\.com/.+/blob/.+|raw\.githubusercontent\.com",
        "github_readme": r"github\.com/[^/]+/[^/]+/?$",
        "confluence": r"atlassian\.net/wiki|confluence\.",
        "raw_json": r"\.jsonl?$|\.csv$",
    }

    def __init__(
        self,
        notion_token: Optional[str] = None,
        google_credentials: Optional[str] = None,
        github_token: Optional[str] = None,
    ):
        """
        Initialize the URL extractor.

        Args:
            notion_token: Notion integration token.
            google_credentials: Path to Google service account JSON.
            github_token: GitHub personal access token.
        """
        self.notion_token = notion_token
        self.google_credentials = google_credentials
        self.github_token = github_token

        if not HTTPX_AVAILABLE:
            logger.warning("httpx not installed. Some features may be limited.")

    def detect_source_type(self, url: str) -> str:
        """Detect the type of source from URL."""
        for source_type, pattern in self.URL_PATTERNS.items():
            if re.search(pattern, url, re.IGNORECASE):
                return source_type
        return "website"

    def extract(self, url: str) -> ExtractedData:
        """
        Extract training data from a URL.

        Args:
            url: The URL to extract data from.

        Returns:
            ExtractedData with samples and metadata.
        """
        source_type = self.detect_source_type(url)
        logger.info(f"Extracting from {source_type}: {url}")

        extractors = {
            "notion": self._extract_notion,
            "google_docs": self._extract_google_docs,
            "google_sheets": self._extract_google_sheets,
            "github": self._extract_github,
            "github_readme": self._extract_github_readme,
            "confluence": self._extract_confluence,
            "raw_json": self._extract_raw_file,
            "website": self._extract_website,
        }

        extractor = extractors.get(source_type, self._extract_website)
        return extractor(url)

    def _fetch_url(self, url: str, headers: Optional[Dict] = None) -> str:
        """Fetch URL content."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required: pip install httpx")

        default_headers = {
            "User-Agent": "TinyForgeAI/1.0 (https://github.com/foremsoft/TinyForgeAI)",
        }
        if headers:
            default_headers.update(headers)

        with httpx.Client(follow_redirects=True, timeout=30) as client:
            response = client.get(url, headers=default_headers)
            response.raise_for_status()
            return response.text

    def _extract_notion(self, url: str) -> ExtractedData:
        """Extract data from Notion page."""
        # Extract page ID from URL
        page_id = self._extract_notion_page_id(url)

        if self.notion_token:
            # Use Notion API if token available
            return self._extract_notion_api(page_id)

        # Fallback: scrape public page
        logger.info("No Notion token, attempting to scrape public page")
        return self._extract_website(url)

    def _extract_notion_page_id(self, url: str) -> str:
        """Extract page ID from Notion URL."""
        # Notion URLs: https://notion.so/workspace/Page-Title-abc123def456
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        # The last part after the final dash is usually the ID
        if "-" in path:
            parts = path.split("-")
            potential_id = parts[-1]
            # Notion IDs are 32 hex chars
            if len(potential_id) == 32 and all(c in "0123456789abcdef" for c in potential_id.lower()):
                return potential_id

        # Try the last path segment
        segments = path.split("/")
        return segments[-1] if segments else path

    def _extract_notion_api(self, page_id: str) -> ExtractedData:
        """Extract Notion content via API."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required: pip install httpx")

        headers = {
            "Authorization": f"Bearer {self.notion_token}",
            "Notion-Version": "2022-06-28",
        }

        with httpx.Client(timeout=30) as client:
            # Get page
            page_url = f"https://api.notion.com/v1/pages/{page_id}"
            page_response = client.get(page_url, headers=headers)
            page_data = page_response.json()

            # Get page content (blocks)
            blocks_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
            blocks_response = client.get(blocks_url, headers=headers)
            blocks_data = blocks_response.json()

        # Extract title
        title = self._extract_notion_title(page_data)

        # Extract text content
        content = self._notion_blocks_to_text(blocks_data.get("results", []))

        # Convert to training samples
        samples = self._content_to_samples(content)

        return ExtractedData(
            url=f"https://notion.so/{page_id}",
            source_type="notion",
            title=title,
            samples=samples,
            raw_content=content,
            metadata={"page_id": page_id},
        )

    def _extract_notion_title(self, page_data: dict) -> str:
        """Extract title from Notion page data."""
        properties = page_data.get("properties", {})
        for prop in properties.values():
            if prop.get("type") == "title":
                title_parts = prop.get("title", [])
                return "".join(t.get("plain_text", "") for t in title_parts)
        return "Untitled"

    def _notion_blocks_to_text(self, blocks: list) -> str:
        """Convert Notion blocks to plain text."""
        text_parts = []

        for block in blocks:
            block_type = block.get("type")

            if block_type in ["paragraph", "heading_1", "heading_2", "heading_3"]:
                rich_text = block.get(block_type, {}).get("rich_text", [])
                text = "".join(t.get("plain_text", "") for t in rich_text)
                if text:
                    text_parts.append(text)

            elif block_type == "bulleted_list_item":
                rich_text = block.get(block_type, {}).get("rich_text", [])
                text = "".join(t.get("plain_text", "") for t in rich_text)
                if text:
                    text_parts.append(f"• {text}")

            elif block_type == "numbered_list_item":
                rich_text = block.get(block_type, {}).get("rich_text", [])
                text = "".join(t.get("plain_text", "") for t in rich_text)
                if text:
                    text_parts.append(f"- {text}")

            elif block_type == "toggle":
                rich_text = block.get(block_type, {}).get("rich_text", [])
                text = "".join(t.get("plain_text", "") for t in rich_text)
                if text:
                    text_parts.append(text)

        return "\n".join(text_parts)

    def _extract_google_docs(self, url: str) -> ExtractedData:
        """Extract data from Google Docs."""
        # Extract doc ID
        doc_id = self._extract_google_doc_id(url)

        # Try to get published version
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"

        try:
            content = self._fetch_url(export_url)
            samples = self._content_to_samples(content)

            return ExtractedData(
                url=url,
                source_type="google_docs",
                title=f"Google Doc {doc_id[:8]}",
                samples=samples,
                raw_content=content,
                metadata={"doc_id": doc_id},
            )
        except Exception as e:
            logger.warning(f"Could not export Google Doc: {e}")
            # Fallback to website extraction
            return self._extract_website(url)

    def _extract_google_doc_id(self, url: str) -> str:
        """Extract document ID from Google Docs URL."""
        # URLs: https://docs.google.com/document/d/DOC_ID/...
        match = re.search(r"/document/d/([a-zA-Z0-9_-]+)", url)
        return match.group(1) if match else ""

    def _extract_google_sheets(self, url: str) -> ExtractedData:
        """Extract data from Google Sheets."""
        # Extract sheet ID
        sheet_id = self._extract_google_sheet_id(url)

        # Try CSV export
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

        try:
            content = self._fetch_url(export_url)
            samples = self._csv_to_samples(content)

            return ExtractedData(
                url=url,
                source_type="google_sheets",
                title=f"Google Sheet {sheet_id[:8]}",
                samples=samples,
                raw_content=content,
                metadata={"sheet_id": sheet_id},
            )
        except Exception as e:
            logger.warning(f"Could not export Google Sheet: {e}")
            return ExtractedData(
                url=url,
                source_type="google_sheets",
                samples=[],
                metadata={"error": str(e)},
            )

    def _extract_google_sheet_id(self, url: str) -> str:
        """Extract spreadsheet ID from Google Sheets URL."""
        match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
        return match.group(1) if match else ""

    def _extract_github(self, url: str) -> ExtractedData:
        """Extract data from GitHub file."""
        # Convert blob URL to raw URL
        raw_url = url.replace("github.com", "raw.githubusercontent.com")
        raw_url = raw_url.replace("/blob/", "/")

        headers = {}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        content = self._fetch_url(raw_url, headers)

        # Detect file type
        if url.endswith(".jsonl"):
            samples = self._jsonl_to_samples(content)
        elif url.endswith(".json"):
            samples = self._json_to_samples(content)
        elif url.endswith(".csv"):
            samples = self._csv_to_samples(content)
        else:
            samples = self._content_to_samples(content)

        return ExtractedData(
            url=url,
            source_type="github",
            title=url.split("/")[-1],
            samples=samples,
            raw_content=content,
            metadata={"raw_url": raw_url},
        )

    def _extract_github_readme(self, url: str) -> ExtractedData:
        """Extract data from GitHub README."""
        # Parse repo URL
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) >= 2:
            owner, repo = path_parts[0], path_parts[1]
            readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"

            try:
                content = self._fetch_url(readme_url)
            except:
                # Try master branch
                readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md"
                content = self._fetch_url(readme_url)

            samples = self._markdown_to_samples(content)

            return ExtractedData(
                url=url,
                source_type="github_readme",
                title=f"{owner}/{repo} README",
                samples=samples,
                raw_content=content,
                metadata={"owner": owner, "repo": repo},
            )

        return ExtractedData(url=url, source_type="github_readme", samples=[])

    def _extract_confluence(self, url: str) -> ExtractedData:
        """Extract data from Confluence page."""
        # Fallback to website extraction for now
        # Full Confluence API integration could be added
        return self._extract_website(url)

    def _extract_raw_file(self, url: str) -> ExtractedData:
        """Extract data from raw JSON/JSONL/CSV file."""
        content = self._fetch_url(url)

        if url.endswith(".jsonl"):
            samples = self._jsonl_to_samples(content)
        elif url.endswith(".json"):
            samples = self._json_to_samples(content)
        elif url.endswith(".csv"):
            samples = self._csv_to_samples(content)
        else:
            samples = self._content_to_samples(content)

        return ExtractedData(
            url=url,
            source_type="raw_file",
            title=url.split("/")[-1],
            samples=samples,
            raw_content=content,
        )

    def _extract_website(self, url: str) -> ExtractedData:
        """Extract data from generic website."""
        if not BS4_AVAILABLE:
            raise ImportError("BeautifulSoup required: pip install beautifulsoup4")

        content = self._fetch_url(url)
        soup = BeautifulSoup(content, "html.parser")

        # Remove scripts and styles
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Get title
        title = soup.title.string if soup.title else url

        # Get text content
        text = soup.get_text(separator="\n", strip=True)

        # Try to detect FAQ patterns
        samples = self._detect_faq_patterns(soup, text)

        if not samples:
            # Fallback: convert content to Q&A
            samples = self._content_to_samples(text)

        return ExtractedData(
            url=url,
            source_type="website",
            title=title,
            samples=samples,
            raw_content=text,
            metadata={"detected_faq": len(samples) > 0},
        )

    def _detect_faq_patterns(self, soup: "BeautifulSoup", text: str) -> List[Dict[str, str]]:
        """Detect FAQ patterns in HTML/text."""
        samples = []

        # Pattern 1: FAQ schema markup
        faq_schemas = soup.find_all(attrs={"itemtype": re.compile(r"FAQPage|Question", re.I)})
        for schema in faq_schemas:
            q = schema.find(attrs={"itemprop": "name"})
            a = schema.find(attrs={"itemprop": "acceptedAnswer"})
            if q and a:
                samples.append({
                    "input": q.get_text(strip=True),
                    "output": a.get_text(strip=True),
                })

        # Pattern 2: Accordion/toggle FAQ
        accordions = soup.find_all(class_=re.compile(r"faq|accordion|toggle", re.I))
        for acc in accordions:
            q = acc.find(class_=re.compile(r"question|title|header", re.I))
            a = acc.find(class_=re.compile(r"answer|content|body", re.I))
            if q and a:
                samples.append({
                    "input": q.get_text(strip=True),
                    "output": a.get_text(strip=True),
                })

        # Pattern 3: Q: ... A: ... format in text
        qa_pattern = re.compile(
            r'(?:Q:|Question:?)\s*(.+?)\s*(?:A:|Answer:?)\s*(.+?)(?=(?:Q:|Question:|$))',
            re.IGNORECASE | re.DOTALL
        )
        for match in qa_pattern.finditer(text):
            q, a = match.groups()
            if len(q.strip()) > 10 and len(a.strip()) > 10:
                samples.append({
                    "input": q.strip()[:500],
                    "output": a.strip()[:1000],
                })

        # Pattern 4: Heading followed by paragraph
        headings = soup.find_all(["h2", "h3", "h4"])
        for heading in headings:
            heading_text = heading.get_text(strip=True)
            # Check if heading looks like a question
            if "?" in heading_text or heading_text.lower().startswith(("how", "what", "why", "when", "where", "can", "do")):
                # Get following paragraph(s)
                answer_parts = []
                for sibling in heading.find_next_siblings():
                    if sibling.name in ["h1", "h2", "h3", "h4"]:
                        break
                    if sibling.name == "p":
                        answer_parts.append(sibling.get_text(strip=True))

                if answer_parts:
                    samples.append({
                        "input": heading_text,
                        "output": " ".join(answer_parts)[:1000],
                    })

        return samples

    def _content_to_samples(self, content: str) -> List[Dict[str, str]]:
        """Convert plain text content to training samples."""
        samples = []

        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        # Create Q&A pairs from consecutive paragraphs
        for i in range(0, len(paragraphs) - 1, 2):
            q = paragraphs[i]
            a = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""

            # Skip if too short or too long
            if len(q) < 10 or len(q) > 500:
                continue
            if len(a) < 10:
                continue

            # Check if first paragraph looks like a question/topic
            if "?" in q or q[0].isupper():
                samples.append({
                    "input": q[:500],
                    "output": a[:1000],
                })

        # If we didn't get pairs, create topic-based samples
        if not samples:
            for p in paragraphs:
                if len(p) > 50:
                    # Create a question about the paragraph
                    first_sentence = p.split(".")[0]
                    if len(first_sentence) > 20:
                        samples.append({
                            "input": f"What is {first_sentence[:50].lower()}?",
                            "output": p[:1000],
                        })

        return samples

    def _markdown_to_samples(self, content: str) -> List[Dict[str, str]]:
        """Convert Markdown content to training samples."""
        samples = []

        # Find heading + content pairs
        heading_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
        matches = list(heading_pattern.finditer(content))

        for i, match in enumerate(matches):
            heading_level, heading_text = match.groups()

            # Get content until next heading
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()

            if len(section_content) > 20:
                # Make the heading a question if it isn't already
                if "?" not in heading_text:
                    question = f"What is {heading_text}?"
                else:
                    question = heading_text

                samples.append({
                    "input": question,
                    "output": section_content[:1000],
                })

        return samples

    def _jsonl_to_samples(self, content: str) -> List[Dict[str, str]]:
        """Convert JSONL content to samples."""
        samples = []
        for line in content.strip().split("\n"):
            if line.strip():
                try:
                    record = json.loads(line)
                    if "input" in record and "output" in record:
                        samples.append({
                            "input": str(record["input"]),
                            "output": str(record["output"]),
                        })
                    elif "question" in record and "answer" in record:
                        samples.append({
                            "input": str(record["question"]),
                            "output": str(record["answer"]),
                        })
                except json.JSONDecodeError:
                    continue
        return samples

    def _json_to_samples(self, content: str) -> List[Dict[str, str]]:
        """Convert JSON content to samples."""
        data = json.loads(content)

        if isinstance(data, list):
            samples = []
            for item in data:
                if isinstance(item, dict):
                    if "input" in item and "output" in item:
                        samples.append({
                            "input": str(item["input"]),
                            "output": str(item["output"]),
                        })
                    elif "question" in item and "answer" in item:
                        samples.append({
                            "input": str(item["question"]),
                            "output": str(item["answer"]),
                        })
            return samples

        return self._content_to_samples(json.dumps(data, indent=2))

    def _csv_to_samples(self, content: str) -> List[Dict[str, str]]:
        """Convert CSV content to samples."""
        import csv
        from io import StringIO

        samples = []
        reader = csv.DictReader(StringIO(content))

        for row in reader:
            # Try common column names
            input_col = None
            output_col = None

            for col in row.keys():
                col_lower = col.lower()
                if col_lower in ["input", "question", "query", "prompt", "q"]:
                    input_col = col
                elif col_lower in ["output", "answer", "response", "a", "target"]:
                    output_col = col

            if input_col and output_col:
                samples.append({
                    "input": str(row[input_col]),
                    "output": str(row[output_col]),
                })
            elif len(row) >= 2:
                # Use first two columns
                keys = list(row.keys())
                samples.append({
                    "input": str(row[keys[0]]),
                    "output": str(row[keys[1]]),
                })

        return samples
