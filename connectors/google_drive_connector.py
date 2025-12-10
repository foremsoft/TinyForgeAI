"""
Google Drive connector for TinyForgeAI.

Provides functionality to list files, download content, and stream training
samples from Google Drive folders and files.

Mock Mode:
    When GOOGLE_DRIVE_MOCK=true (default), the connector reads from
    local sample files in examples/google_drive_samples/ instead of
    making real API calls.

Real Mode:
    When GOOGLE_DRIVE_MOCK=false, the connector uses the Google Drive API
    with OAuth credentials to access real files.

Usage:
    from connectors.google_drive_connector import GoogleDriveConnector

    # Create connector (uses mock mode by default)
    connector = GoogleDriveConnector()

    # List files in a folder
    files = connector.list_files(folder_id="my_folder")

    # Download file content
    content = connector.get_file_content(file_id="my_file")

    # Stream training samples from folder
    for sample in connector.stream_samples(folder_id="my_folder", mapping={...}):
        print(sample)
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

# Add project root to path when run as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from connectors.mappers import row_to_sample

logger = logging.getLogger(__name__)

# Check for Google API client
GOOGLE_API_AVAILABLE = False
try:
    from google.oauth2.credentials import Credentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    GOOGLE_API_AVAILABLE = True
except ImportError:
    pass


@dataclass
class GoogleDriveConfig:
    """Configuration for Google Drive connector."""

    # Authentication
    credentials_file: Optional[str] = None  # Path to credentials JSON
    service_account_file: Optional[str] = None  # Path to service account JSON
    token_file: Optional[str] = None  # Path to token file for OAuth

    # Mock mode settings
    mock_mode: bool = True  # Use mock mode by default
    samples_dir: Optional[str] = None  # Custom samples directory

    # Query settings
    page_size: int = 100  # Items per page when listing
    include_trashed: bool = False  # Include trashed files

    # File type filters
    mime_types: List[str] = field(default_factory=lambda: [
        "application/vnd.google-apps.document",  # Google Docs
        "application/vnd.google-apps.spreadsheet",  # Google Sheets
        "text/plain",
        "text/markdown",
        "application/json",
        "text/csv",
    ])

    # Export formats for Google Workspace files
    export_formats: Dict[str, str] = field(default_factory=lambda: {
        "application/vnd.google-apps.document": "text/plain",
        "application/vnd.google-apps.spreadsheet": "text/csv",
        "application/vnd.google-apps.presentation": "text/plain",
    })


@dataclass
class DriveFile:
    """Represents a file in Google Drive."""

    id: str
    name: str
    mime_type: str
    size: Optional[int] = None
    created_time: Optional[str] = None
    modified_time: Optional[str] = None
    parents: List[str] = field(default_factory=list)
    web_view_link: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "mime_type": self.mime_type,
            "size": self.size,
            "created_time": self.created_time,
            "modified_time": self.modified_time,
            "parents": self.parents,
            "web_view_link": self.web_view_link,
        }


class GoogleDriveConnector:
    """
    Google Drive connector for accessing files and streaming training samples.

    Example:
        connector = GoogleDriveConnector()

        # List files
        files = connector.list_files(folder_id="folder_id")

        # Get file content
        content = connector.get_file_content(file_id="file_id")

        # Stream samples
        mapping = {"input": "question", "output": "answer"}
        for sample in connector.stream_samples("folder_id", mapping):
            print(sample)
    """

    def __init__(self, config: Optional[GoogleDriveConfig] = None):
        """
        Initialize the Google Drive connector.

        Args:
            config: Configuration object. If None, uses defaults with mock mode.
        """
        self.config = config or GoogleDriveConfig()

        # Check environment variables for mock mode override
        env_mock = os.getenv("GOOGLE_DRIVE_MOCK", "").lower()
        if env_mock in ("true", "1", "yes"):
            self.config.mock_mode = True
        elif env_mock in ("false", "0", "no"):
            self.config.mock_mode = False

        # Also check global connector mock setting
        global_mock = os.getenv("CONNECTOR_MOCK", "").lower()
        if global_mock in ("true", "1", "yes"):
            self.config.mock_mode = True

        self._service = None
        logger.debug(f"GoogleDriveConnector initialized (mock_mode={self.config.mock_mode})")

    def _get_samples_dir(self) -> Path:
        """Get the path to the sample files directory."""
        if self.config.samples_dir:
            return Path(self.config.samples_dir)

        # Try relative to this file first
        connector_dir = Path(__file__).parent
        samples_dir = connector_dir.parent / "examples" / "google_drive_samples"
        if samples_dir.exists():
            return samples_dir

        # Fallback to current working directory
        return Path("examples") / "google_drive_samples"

    def _get_service(self):
        """Get or create the Google Drive API service."""
        if self._service is not None:
            return self._service

        if not GOOGLE_API_AVAILABLE:
            raise ImportError(
                "Google API client not installed. "
                "Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )

        credentials = None

        # Try service account first
        if self.config.service_account_file:
            credentials = ServiceAccountCredentials.from_service_account_file(
                self.config.service_account_file,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )

        # Try OAuth credentials
        elif self.config.token_file and Path(self.config.token_file).exists():
            credentials = Credentials.from_authorized_user_file(
                self.config.token_file,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )

        if credentials is None:
            raise RuntimeError(
                "No valid credentials found. Provide either:\n"
                "- service_account_file: Path to service account JSON\n"
                "- token_file: Path to OAuth token file\n"
                "Or set GOOGLE_DRIVE_MOCK=true for mock mode."
            )

        self._service = build("drive", "v3", credentials=credentials)
        return self._service

    def list_files(
        self,
        folder_id: Optional[str] = None,
        query: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> List[DriveFile]:
        """
        List files in Google Drive.

        Args:
            folder_id: ID of folder to list (None for root).
            query: Custom search query (overrides folder_id).
            page_size: Number of results per page.

        Returns:
            List of DriveFile objects.
        """
        if self.config.mock_mode:
            return self._list_files_mock(folder_id)
        return self._list_files_real(folder_id, query, page_size)

    def _list_files_mock(self, folder_id: Optional[str] = None) -> List[DriveFile]:
        """List files from mock samples directory."""
        samples_dir = self._get_samples_dir()

        if not samples_dir.exists():
            logger.warning(f"Samples directory not found: {samples_dir}")
            return []

        # If folder_id specified, look for subdirectory
        if folder_id:
            target_dir = samples_dir / folder_id
            if target_dir.exists():
                samples_dir = target_dir

        files = []
        for file_path in sorted(samples_dir.iterdir()):
            if file_path.is_file() and not file_path.name.startswith("."):
                mime_type = self._guess_mime_type(file_path.suffix)
                files.append(DriveFile(
                    id=file_path.stem,
                    name=file_path.name,
                    mime_type=mime_type,
                    size=file_path.stat().st_size,
                    modified_time=str(file_path.stat().st_mtime),
                ))

        return files

    def _list_files_real(
        self,
        folder_id: Optional[str] = None,
        query: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> List[DriveFile]:
        """List files using Google Drive API."""
        service = self._get_service()
        page_size = page_size or self.config.page_size

        # Build query
        if query is None:
            parts = []
            if folder_id:
                parts.append(f"'{folder_id}' in parents")
            if not self.config.include_trashed:
                parts.append("trashed = false")
            query = " and ".join(parts) if parts else None

        files = []
        page_token = None

        while True:
            results = service.files().list(
                q=query,
                pageSize=page_size,
                pageToken=page_token,
                fields="nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, parents, webViewLink)",
            ).execute()

            for item in results.get("files", []):
                files.append(DriveFile(
                    id=item["id"],
                    name=item["name"],
                    mime_type=item["mimeType"],
                    size=item.get("size"),
                    created_time=item.get("createdTime"),
                    modified_time=item.get("modifiedTime"),
                    parents=item.get("parents", []),
                    web_view_link=item.get("webViewLink"),
                ))

            page_token = results.get("nextPageToken")
            if not page_token:
                break

        return files

    def get_file_content(self, file_id: str) -> str:
        """
        Get the text content of a file.

        Args:
            file_id: ID of the file to download.

        Returns:
            Text content of the file.

        Raises:
            FileNotFoundError: If file not found (mock mode).
            RuntimeError: If download fails.
        """
        if self.config.mock_mode:
            return self._get_file_content_mock(file_id)
        return self._get_file_content_real(file_id)

    def _get_file_content_mock(self, file_id: str) -> str:
        """Get file content from mock samples."""
        samples_dir = self._get_samples_dir()

        # Extensions to try (including jsonl which is common for training data)
        extensions = ["", ".txt", ".json", ".jsonl", ".md", ".csv"]

        # Try exact match first
        for ext in extensions:
            file_path = samples_dir / f"{file_id}{ext}"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()

        # Try searching in subdirectories
        for subdir in samples_dir.iterdir():
            if subdir.is_dir():
                for ext in extensions:
                    file_path = subdir / f"{file_id}{ext}"
                    if file_path.exists():
                        with open(file_path, "r", encoding="utf-8") as f:
                            return f.read()

        available = [f.stem for f in samples_dir.rglob("*") if f.is_file()]
        raise FileNotFoundError(
            f"Mock file '{file_id}' not found. "
            f"Available samples: {available[:10]}{'...' if len(available) > 10 else ''}"
        )

    def _get_file_content_real(self, file_id: str) -> str:
        """Get file content using Google Drive API."""
        import io

        service = self._get_service()

        # Get file metadata to determine type
        file_meta = service.files().get(fileId=file_id, fields="mimeType,name").execute()
        mime_type = file_meta["mimeType"]

        # Handle Google Workspace files (need export)
        if mime_type in self.config.export_formats:
            export_mime = self.config.export_formats[mime_type]
            request = service.files().export_media(fileId=file_id, mimeType=export_mime)
        else:
            request = service.files().get_media(fileId=file_id)

        # Download content
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        buffer.seek(0)
        return buffer.read().decode("utf-8")

    def get_file_metadata(self, file_id: str) -> DriveFile:
        """
        Get metadata for a file.

        Args:
            file_id: ID of the file.

        Returns:
            DriveFile object with metadata.
        """
        if self.config.mock_mode:
            files = self._list_files_mock()
            for f in files:
                if f.id == file_id:
                    return f
            raise FileNotFoundError(f"File not found: {file_id}")

        service = self._get_service()
        item = service.files().get(
            fileId=file_id,
            fields="id, name, mimeType, size, createdTime, modifiedTime, parents, webViewLink",
        ).execute()

        return DriveFile(
            id=item["id"],
            name=item["name"],
            mime_type=item["mimeType"],
            size=item.get("size"),
            created_time=item.get("createdTime"),
            modified_time=item.get("modifiedTime"),
            parents=item.get("parents", []),
            web_view_link=item.get("webViewLink"),
        )

    def stream_samples(
        self,
        folder_id: str,
        mapping: Dict[str, str],
        file_pattern: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream training samples from files in a Google Drive folder.

        Each file is expected to contain JSON or JSONL data that can be
        mapped to training samples.

        Args:
            folder_id: ID of the folder containing training files.
            mapping: Column mapping dict with "input" and "output" keys.
            file_pattern: Optional filename pattern to filter files.

        Yields:
            Training sample dicts with "input", "output", and "metadata" keys.
        """
        files = self.list_files(folder_id)

        for file in files:
            # Filter by pattern if specified
            if file_pattern and file_pattern not in file.name:
                continue

            # Skip unsupported file types
            if file.mime_type not in self.config.mime_types and not file.name.endswith((".json", ".jsonl", ".txt")):
                continue

            try:
                content = self.get_file_content(file.id)

                # Parse content based on file type
                if file.name.endswith(".jsonl"):
                    # JSONL format - one JSON object per line
                    for line in content.strip().split("\n"):
                        if line.strip():
                            row = json.loads(line)
                            sample = row_to_sample(row, mapping)
                            sample["metadata"]["source_file"] = file.name
                            sample["metadata"]["source_id"] = file.id
                            yield sample

                elif file.name.endswith(".json"):
                    # JSON format - array or object
                    data = json.loads(content)
                    if isinstance(data, list):
                        for row in data:
                            sample = row_to_sample(row, mapping)
                            sample["metadata"]["source_file"] = file.name
                            sample["metadata"]["source_id"] = file.id
                            yield sample
                    else:
                        sample = row_to_sample(data, mapping)
                        sample["metadata"]["source_file"] = file.name
                        sample["metadata"]["source_id"] = file.id
                        yield sample

                else:
                    # Plain text - treat as single sample
                    yield {
                        "input": content,
                        "output": "",
                        "metadata": {
                            "source": "google_drive",
                            "source_file": file.name,
                            "source_id": file.id,
                        },
                    }

            except Exception as e:
                logger.warning(f"Error processing file {file.name}: {e}")
                continue

    def _guess_mime_type(self, extension: str) -> str:
        """Guess MIME type from file extension."""
        mime_map = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".json": "application/json",
            ".jsonl": "application/json",
            ".csv": "text/csv",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        return mime_map.get(extension.lower(), "application/octet-stream")


def main() -> int:
    """CLI entry point for the Google Drive connector."""
    parser = argparse.ArgumentParser(
        description="Access files from Google Drive (or mock samples)."
    )
    parser.add_argument(
        "--folder-id",
        help="Google Drive folder ID to list/stream from.",
    )
    parser.add_argument(
        "--file-id",
        help="Google Drive file ID to download.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List files in folder.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream training samples from folder.",
    )
    parser.add_argument(
        "--mapping",
        default='{"input": "input", "output": "output"}',
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

    config = GoogleDriveConfig(mock_mode=args.mock or True)
    connector = GoogleDriveConnector(config)

    try:
        if args.list:
            files = connector.list_files(folder_id=args.folder_id)
            for f in files:
                print(json.dumps(f.to_dict()))
            return 0

        if args.file_id:
            content = connector.get_file_content(args.file_id)
            print(content)
            return 0

        if args.stream:
            if not args.folder_id:
                print("Error: --folder-id required for streaming", file=sys.stderr)
                return 1

            mapping = json.loads(args.mapping)
            count = 0
            for sample in connector.stream_samples(args.folder_id, mapping):
                print(json.dumps(sample))
                count += 1
                if args.limit and count >= args.limit:
                    break
            return 0

        # Default: list available samples
        files = connector.list_files()
        print("Available files (mock mode):")
        for f in files:
            print(f"  {f.id}: {f.name} ({f.mime_type})")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
