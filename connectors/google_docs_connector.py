"""
Google Docs connector for TinyForgeAI.

Provides functionality to fetch document text from Google Docs.
Includes a mock mode for offline testing and development.

Mock Mode:
    When GOOGLE_DOCS_MOCK=true or GOOGLE_OAUTH_DISABLED=true (default),
    the connector reads from local sample files in examples/google_docs_samples/
    instead of making real API calls.

Real Mode:
    When GOOGLE_DOCS_MOCK=false and GOOGLE_OAUTH_DISABLED=false, the connector
    uses the Google Docs API with OAuth or service account credentials to
    fetch real documents.

Usage:
    from connectors.google_docs_connector import GoogleDocsConnector

    # Create connector (uses mock mode by default)
    connector = GoogleDocsConnector()

    # Fetch document text
    text = connector.fetch_doc_text(doc_id="my_document_id")

    # List documents (requires Drive API access)
    docs = connector.list_docs_in_folder(folder_id="folder_id")

    # Stream training samples from documents
    for sample in connector.stream_samples(folder_id="folder_id"):
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

# Add project root to path when run as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from connectors.google_utils import normalize_text

logger = logging.getLogger(__name__)

# Check for Google API client
GOOGLE_API_AVAILABLE = False
try:
    from google.oauth2.credentials import Credentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    HttpError = Exception  # Fallback for type hints


# Scopes required for Google Docs and Drive access
SCOPES = [
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


@dataclass
class GoogleDocsConfig:
    """Configuration for Google Docs connector."""

    # Authentication
    credentials_file: Optional[str] = None  # Path to OAuth credentials JSON
    service_account_file: Optional[str] = None  # Path to service account JSON
    token_file: Optional[str] = None  # Path to token file for OAuth

    # Mock mode settings
    mock_mode: bool = True  # Use mock mode by default
    samples_dir: Optional[str] = None  # Custom samples directory

    # Content extraction settings
    include_headers: bool = True  # Include header/title text
    include_lists: bool = True  # Include list items
    include_tables: bool = True  # Include table content
    paragraph_separator: str = "\n\n"  # Separator between paragraphs


@dataclass
class GoogleDoc:
    """Represents a Google Doc."""

    id: str
    title: str
    revision_id: Optional[str] = None
    created_time: Optional[str] = None
    modified_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "revision_id": self.revision_id,
            "created_time": self.created_time,
            "modified_time": self.modified_time,
        }


class GoogleDocsConnector:
    """
    Google Docs connector for fetching document content.

    Example:
        connector = GoogleDocsConnector()

        # Fetch document text
        text = connector.fetch_doc_text(doc_id="document_id")

        # List documents in folder
        docs = connector.list_docs_in_folder(folder_id="folder_id")

        # Stream training samples
        for sample in connector.stream_samples(folder_id="folder_id"):
            print(sample)
    """

    def __init__(self, config: Optional[GoogleDocsConfig] = None):
        """
        Initialize the Google Docs connector.

        Args:
            config: Configuration object. If None, uses defaults with mock mode.
        """
        self.config = config or GoogleDocsConfig()

        # Check environment variables for mock mode override
        env_mock = os.getenv("GOOGLE_DOCS_MOCK", "").lower()
        if env_mock in ("true", "1", "yes"):
            self.config.mock_mode = True
        elif env_mock in ("false", "0", "no"):
            self.config.mock_mode = False

        # Legacy environment variable support
        oauth_disabled = os.getenv("GOOGLE_OAUTH_DISABLED", "true").lower()
        if oauth_disabled in ("true", "1", "yes") and not env_mock:
            self.config.mock_mode = True

        # Global connector mock setting
        global_mock = os.getenv("CONNECTOR_MOCK", "").lower()
        if global_mock in ("true", "1", "yes"):
            self.config.mock_mode = True

        self._docs_service = None
        self._drive_service = None
        self._credentials = None
        logger.debug(f"GoogleDocsConnector initialized (mock_mode={self.config.mock_mode})")

    def _get_samples_dir(self) -> Path:
        """Get the path to the sample docs directory."""
        if self.config.samples_dir:
            return Path(self.config.samples_dir)

        # Try relative to this file first
        connector_dir = Path(__file__).parent
        samples_dir = connector_dir.parent / "examples" / "google_docs_samples"
        if samples_dir.exists():
            return samples_dir

        # Fallback to current working directory
        return Path("examples") / "google_docs_samples"

    def _get_credentials(self):
        """Get or create Google API credentials."""
        if self._credentials is not None:
            return self._credentials

        if not GOOGLE_API_AVAILABLE:
            raise ImportError(
                "Google API client not installed. "
                "Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )

        credentials = None

        # Try service account first (preferred for server-side)
        if self.config.service_account_file:
            sa_path = Path(self.config.service_account_file)
            if sa_path.exists():
                credentials = ServiceAccountCredentials.from_service_account_file(
                    str(sa_path),
                    scopes=SCOPES,
                )
                logger.info("Using service account credentials")

        # Try OAuth token file
        if credentials is None and self.config.token_file:
            token_path = Path(self.config.token_file)
            if token_path.exists():
                credentials = Credentials.from_authorized_user_file(
                    str(token_path),
                    scopes=SCOPES,
                )
                logger.info("Using OAuth token credentials")

        # Try environment variable for service account
        if credentials is None:
            sa_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if sa_env and Path(sa_env).exists():
                credentials = ServiceAccountCredentials.from_service_account_file(
                    sa_env,
                    scopes=SCOPES,
                )
                logger.info("Using credentials from GOOGLE_APPLICATION_CREDENTIALS")

        if credentials is None:
            raise RuntimeError(
                "No valid Google credentials found. Provide one of:\n"
                "- service_account_file: Path to service account JSON\n"
                "- token_file: Path to OAuth token file\n"
                "- GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
                "Or set GOOGLE_DOCS_MOCK=true for mock mode."
            )

        self._credentials = credentials
        return credentials

    def _get_docs_service(self):
        """Get or create the Google Docs API service."""
        if self._docs_service is not None:
            return self._docs_service

        credentials = self._get_credentials()
        self._docs_service = build("docs", "v1", credentials=credentials)
        return self._docs_service

    def _get_drive_service(self):
        """Get or create the Google Drive API service."""
        if self._drive_service is not None:
            return self._drive_service

        credentials = self._get_credentials()
        self._drive_service = build("drive", "v3", credentials=credentials)
        return self._drive_service

    def fetch_doc_text(
        self,
        doc_id: str,
        oauth_credentials: Optional[dict] = None,
    ) -> str:
        """
        Fetch and extract text content from a Google Doc.

        In mock mode, reads from local sample files.
        In real mode, uses the Google Docs API.

        Args:
            doc_id: The Google Doc ID or sample doc name (in mock mode).
            oauth_credentials: Deprecated. Use config.token_file instead.

        Returns:
            The text content of the document.

        Raises:
            FileNotFoundError: If document not found (mock mode).
            RuntimeError: If API call fails.
        """
        if self.config.mock_mode:
            return self._fetch_doc_mock(doc_id)
        return self._fetch_doc_real(doc_id)

    def _fetch_doc_mock(self, doc_id: str) -> str:
        """Fetch document text from local sample files (mock mode)."""
        samples_dir = self._get_samples_dir()

        # Try various extensions
        extensions = ["", ".txt", ".md", ".html"]
        for ext in extensions:
            file_path = samples_dir / f"{doc_id}{ext}"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                return normalize_text(text)

        # List available samples for error message
        available = []
        if samples_dir.exists():
            available = [f.stem for f in samples_dir.glob("*") if f.is_file()]

        raise FileNotFoundError(
            f"Mock document '{doc_id}' not found. "
            f"Available samples: {available}. "
            f"Expected file: {samples_dir / doc_id}.txt"
        )

    def _fetch_doc_real(self, doc_id: str) -> str:
        """Fetch document text from Google Docs API (real mode)."""
        try:
            service = self._get_docs_service()
            document = service.documents().get(documentId=doc_id).execute()
            text = self._extract_text_from_document(document)
            return normalize_text(text)

        except HttpError as e:
            if e.resp.status == 404:
                raise FileNotFoundError(f"Document not found: {doc_id}")
            elif e.resp.status == 403:
                raise PermissionError(
                    f"Access denied for document: {doc_id}. "
                    "Ensure the document is shared with your service account or OAuth user."
                )
            else:
                raise RuntimeError(f"Google Docs API error: {e}")

    def _extract_text_from_document(self, document: dict) -> str:
        """
        Extract plain text from a Google Docs document structure.

        The Google Docs API returns a complex nested structure. This method
        traverses it to extract all text content.

        Args:
            document: Document object from Google Docs API.

        Returns:
            Extracted text content.
        """
        text_parts = []

        # Extract title if configured
        if self.config.include_headers and "title" in document:
            text_parts.append(document["title"])
            text_parts.append("")  # Add blank line after title

        # Process document body
        body = document.get("body", {})
        content = body.get("content", [])

        for element in content:
            extracted = self._extract_text_from_element(element)
            if extracted:
                text_parts.append(extracted)

        return self.config.paragraph_separator.join(text_parts)

    def _extract_text_from_element(self, element: dict) -> str:
        """Extract text from a document element."""
        text_parts = []

        # Paragraph
        if "paragraph" in element:
            paragraph = element["paragraph"]
            para_text = self._extract_text_from_paragraph(paragraph)
            if para_text:
                text_parts.append(para_text)

        # Table
        if "table" in element and self.config.include_tables:
            table = element["table"]
            table_text = self._extract_text_from_table(table)
            if table_text:
                text_parts.append(table_text)

        # Table of contents
        if "tableOfContents" in element:
            toc = element["tableOfContents"]
            toc_content = toc.get("content", [])
            for toc_element in toc_content:
                extracted = self._extract_text_from_element(toc_element)
                if extracted:
                    text_parts.append(extracted)

        # Section break (ignore)
        # Page break (ignore)

        return "\n".join(text_parts)

    def _extract_text_from_paragraph(self, paragraph: dict) -> str:
        """Extract text from a paragraph element."""
        text_parts = []

        elements = paragraph.get("elements", [])
        for elem in elements:
            if "textRun" in elem:
                content = elem["textRun"].get("content", "")
                text_parts.append(content)

            # Handle inline objects (images, etc.) - just note their presence
            if "inlineObjectElement" in elem:
                text_parts.append("[Image]")

        text = "".join(text_parts)

        # Handle lists if configured
        if self.config.include_lists:
            bullet = paragraph.get("bullet")
            if bullet:
                nesting_level = bullet.get("nestingLevel", 0)
                indent = "  " * nesting_level
                text = f"{indent}• {text.strip()}"

        return text.strip()

    def _extract_text_from_table(self, table: dict) -> str:
        """Extract text from a table element."""
        rows = []

        for table_row in table.get("tableRows", []):
            cells = []
            for table_cell in table_row.get("tableCells", []):
                cell_text_parts = []
                for content in table_cell.get("content", []):
                    if "paragraph" in content:
                        para_text = self._extract_text_from_paragraph(content["paragraph"])
                        if para_text:
                            cell_text_parts.append(para_text)
                cells.append(" ".join(cell_text_parts))
            rows.append(" | ".join(cells))

        return "\n".join(rows)

    def get_doc_metadata(self, doc_id: str) -> GoogleDoc:
        """
        Get metadata for a Google Doc.

        Args:
            doc_id: The Google Doc ID.

        Returns:
            GoogleDoc object with metadata.
        """
        if self.config.mock_mode:
            return self._get_doc_metadata_mock(doc_id)
        return self._get_doc_metadata_real(doc_id)

    def _get_doc_metadata_mock(self, doc_id: str) -> GoogleDoc:
        """Get document metadata from mock samples."""
        samples_dir = self._get_samples_dir()

        # Find the file
        for ext in ["", ".txt", ".md", ".html"]:
            file_path = samples_dir / f"{doc_id}{ext}"
            if file_path.exists():
                return GoogleDoc(
                    id=doc_id,
                    title=doc_id.replace("_", " ").title(),
                    modified_time=str(file_path.stat().st_mtime),
                )

        raise FileNotFoundError(f"Document not found: {doc_id}")

    def _get_doc_metadata_real(self, doc_id: str) -> GoogleDoc:
        """Get document metadata from Google Docs API."""
        try:
            service = self._get_docs_service()
            document = service.documents().get(documentId=doc_id).execute()

            return GoogleDoc(
                id=doc_id,
                title=document.get("title", "Untitled"),
                revision_id=document.get("revisionId"),
            )

        except HttpError as e:
            if e.resp.status == 404:
                raise FileNotFoundError(f"Document not found: {doc_id}")
            raise RuntimeError(f"Google Docs API error: {e}")

    def list_docs_in_folder(self, folder_id: str) -> List[GoogleDoc]:
        """
        List Google Docs in a Google Drive folder.

        Args:
            folder_id: The Google Drive folder ID.

        Returns:
            List of GoogleDoc objects.
        """
        if self.config.mock_mode:
            return self._list_docs_mock()
        return self._list_docs_real(folder_id)

    def _list_docs_mock(self) -> List[GoogleDoc]:
        """List available sample documents (mock mode)."""
        samples_dir = self._get_samples_dir()

        if not samples_dir.exists():
            return []

        docs = []
        for file_path in sorted(samples_dir.glob("*")):
            if file_path.is_file() and not file_path.name.startswith("."):
                docs.append(GoogleDoc(
                    id=file_path.stem,
                    title=file_path.stem.replace("_", " ").title(),
                    modified_time=str(file_path.stat().st_mtime),
                ))

        return docs

    def _list_docs_real(self, folder_id: str) -> List[GoogleDoc]:
        """List Google Docs in a folder using Drive API."""
        try:
            service = self._get_drive_service()

            # Query for Google Docs in the specified folder
            query = (
                f"'{folder_id}' in parents and "
                "mimeType='application/vnd.google-apps.document' and "
                "trashed=false"
            )

            docs = []
            page_token = None

            while True:
                results = service.files().list(
                    q=query,
                    pageSize=100,
                    pageToken=page_token,
                    fields="nextPageToken, files(id, name, createdTime, modifiedTime)",
                ).execute()

                for item in results.get("files", []):
                    docs.append(GoogleDoc(
                        id=item["id"],
                        title=item["name"],
                        created_time=item.get("createdTime"),
                        modified_time=item.get("modifiedTime"),
                    ))

                page_token = results.get("nextPageToken")
                if not page_token:
                    break

            return docs

        except HttpError as e:
            if e.resp.status == 404:
                raise FileNotFoundError(f"Folder not found: {folder_id}")
            raise RuntimeError(f"Google Drive API error: {e}")

    def stream_samples(
        self,
        folder_id: str,
        chunk_by: str = "paragraph",
        min_length: int = 50,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream training samples from Google Docs in a folder.

        Each document is split into chunks based on the chunk_by parameter.
        Useful for creating training data from documentation.

        Args:
            folder_id: Google Drive folder ID containing documents.
            chunk_by: How to split documents - "paragraph", "section", or "document".
            min_length: Minimum character length for a chunk to be included.

        Yields:
            Training sample dicts with "input", "output", and "metadata" keys.
        """
        docs = self.list_docs_in_folder(folder_id)

        for doc in docs:
            try:
                text = self.fetch_doc_text(doc.id)

                if chunk_by == "document":
                    # Whole document as one sample
                    if len(text) >= min_length:
                        yield {
                            "input": doc.title,
                            "output": text,
                            "metadata": {
                                "source": "google_docs",
                                "doc_id": doc.id,
                                "doc_title": doc.title,
                                "chunk_type": "document",
                            },
                        }

                elif chunk_by == "section":
                    # Split by headers (lines starting with multiple #)
                    sections = re.split(r'\n(?=#+\s)', text)
                    for i, section in enumerate(sections):
                        section = section.strip()
                        if len(section) >= min_length:
                            # Extract header as input
                            lines = section.split('\n', 1)
                            header = lines[0].lstrip('#').strip() if lines else f"Section {i+1}"
                            content = lines[1].strip() if len(lines) > 1 else section

                            yield {
                                "input": header,
                                "output": content,
                                "metadata": {
                                    "source": "google_docs",
                                    "doc_id": doc.id,
                                    "doc_title": doc.title,
                                    "chunk_type": "section",
                                    "section_index": i,
                                },
                            }

                else:  # paragraph
                    # Split by double newlines (paragraphs)
                    paragraphs = text.split("\n\n")
                    for i, para in enumerate(paragraphs):
                        para = para.strip()
                        if len(para) >= min_length:
                            yield {
                                "input": f"{doc.title} - Paragraph {i+1}",
                                "output": para,
                                "metadata": {
                                    "source": "google_docs",
                                    "doc_id": doc.id,
                                    "doc_title": doc.title,
                                    "chunk_type": "paragraph",
                                    "paragraph_index": i,
                                },
                            }

            except Exception as e:
                logger.warning(f"Error processing document {doc.id}: {e}")
                continue


# Legacy function interface for backward compatibility
def _is_mock_mode() -> bool:
    """Check if running in mock mode."""
    value = os.getenv("GOOGLE_OAUTH_DISABLED", "true").lower()
    mock_value = os.getenv("GOOGLE_DOCS_MOCK", "").lower()
    if mock_value:
        return mock_value in ("true", "1", "yes")
    return value in ("true", "1", "yes")


def fetch_doc_text(doc_id: str, oauth_credentials: Optional[dict] = None) -> str:
    """
    Fetch and extract text content from a Google Doc.

    Legacy function interface. Prefer using GoogleDocsConnector class.

    Args:
        doc_id: The Google Doc ID or sample doc name (in mock mode).
        oauth_credentials: Deprecated. Not used.

    Returns:
        The text content of the document.
    """
    connector = GoogleDocsConnector()
    return connector.fetch_doc_text(doc_id)


def list_docs_in_folder(folder_id: str) -> List[dict]:
    """
    List documents in a Google Drive folder.

    Legacy function interface. Prefer using GoogleDocsConnector class.

    Args:
        folder_id: The Google Drive folder ID.

    Returns:
        List of document metadata dicts with 'id' and 'title' keys.
    """
    connector = GoogleDocsConnector()
    docs = connector.list_docs_in_folder(folder_id)
    return [{"id": doc.id, "title": doc.title} for doc in docs]


def main() -> int:
    """CLI entry point for the Google Docs connector."""
    parser = argparse.ArgumentParser(
        description="Fetch text from Google Docs (or samples in mock mode)."
    )
    parser.add_argument(
        "--doc-id",
        help="Google Doc ID to fetch.",
    )
    parser.add_argument(
        "--folder-id",
        help="Google Drive folder ID to list documents from.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available documents.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream training samples from folder.",
    )
    parser.add_argument(
        "--chunk-by",
        choices=["paragraph", "section", "document"],
        default="paragraph",
        help="How to chunk documents for streaming.",
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
    parser.add_argument(
        "--service-account",
        help="Path to service account JSON file.",
    )

    args = parser.parse_args()

    # Build config
    config = GoogleDocsConfig(
        mock_mode=args.mock,
        service_account_file=args.service_account,
    )

    connector = GoogleDocsConnector(config)

    try:
        if args.list:
            folder_id = args.folder_id or "mock"
            docs = connector.list_docs_in_folder(folder_id)
            for doc in docs:
                print(json.dumps(doc.to_dict()))
            return 0

        if args.doc_id:
            text = connector.fetch_doc_text(args.doc_id)
            print(text)
            return 0

        if args.stream:
            if not args.folder_id:
                print("Error: --folder-id required for streaming", file=sys.stderr)
                return 1

            count = 0
            for sample in connector.stream_samples(args.folder_id, chunk_by=args.chunk_by):
                print(json.dumps(sample))
                count += 1
                if args.limit and count >= args.limit:
                    break
            return 0

        # Default: list available samples
        docs = connector.list_docs_in_folder("mock")
        print("Available documents (mock mode):")
        for doc in docs:
            print(f"  {doc.id}: {doc.title}")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except PermissionError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
