"""
Async Google Drive connector for TinyForgeAI.

Provides asynchronous functionality to list files, download content, and stream
training samples from Google Drive folders and files.

Uses httpx for async HTTP requests when in real mode, and aiofiles for
async file operations in mock mode.

Mock Mode:
    When GOOGLE_DRIVE_MOCK=true (default), the connector reads from
    local sample files in examples/google_drive_samples/ instead of
    making real API calls.

Usage:
    from connectors.async_google_drive_connector import AsyncGoogleDriveConnector
    from connectors.google_drive_connector import GoogleDriveConfig

    # Create connector (uses mock mode by default)
    async with AsyncGoogleDriveConnector() as connector:
        # List files in a folder
        files = await connector.list_files(folder_id="my_folder")

        # Download file content
        content = await connector.get_file_content(file_id="my_file")

        # Stream training samples from folder
        async for sample in connector.stream_samples(folder_id="my_folder", mapping={...}):
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
from connectors.google_drive_connector import GoogleDriveConfig, DriveFile

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


class AsyncGoogleDriveConnector:
    """
    Async Google Drive connector for accessing files and streaming training samples.

    Supports both mock mode (for local development) and real mode (using Google Drive API).

    Example:
        async with AsyncGoogleDriveConnector() as connector:
            # List files
            files = await connector.list_files(folder_id="folder_id")

            # Get file content
            content = await connector.get_file_content(file_id="file_id")

            # Stream samples
            mapping = {"input": "question", "output": "answer"}
            async for sample in connector.stream_samples("folder_id", mapping):
                print(sample)
    """

    # Google Drive API base URL
    BASE_URL = "https://www.googleapis.com/drive/v3"
    UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3"

    def __init__(self, config: Optional[GoogleDriveConfig] = None):
        """
        Initialize the async Google Drive connector.

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

        self._client: Optional[httpx.AsyncClient] = None
        self._access_token: Optional[str] = None
        logger.debug(f"AsyncGoogleDriveConnector initialized (mock_mode={self.config.mock_mode})")

    async def __aenter__(self) -> "AsyncGoogleDriveConnector":
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
                "Async Google Drive connector requires httpx. "
                "Install with: pip install httpx"
            )

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
            logger.debug("Created new httpx AsyncClient")

        return self._client

    async def _get_access_token(self) -> str:
        """Get access token for API requests."""
        if self._access_token:
            return self._access_token

        # Try service account
        if self.config.service_account_file:
            self._access_token = await self._get_service_account_token()
        # Try OAuth token file
        elif self.config.token_file and Path(self.config.token_file).exists():
            self._access_token = await self._get_oauth_token()
        else:
            raise RuntimeError(
                "No valid credentials found. Provide either:\n"
                "- service_account_file: Path to service account JSON\n"
                "- token_file: Path to OAuth token file\n"
                "Or set GOOGLE_DRIVE_MOCK=true for mock mode."
            )

        return self._access_token

    async def _get_service_account_token(self) -> str:
        """Get access token using service account credentials."""
        import jwt
        import time

        # Read service account file
        with open(self.config.service_account_file, "r") as f:
            sa_info = json.load(f)

        # Create JWT
        now = int(time.time())
        payload = {
            "iss": sa_info["client_email"],
            "scope": "https://www.googleapis.com/auth/drive.readonly",
            "aud": "https://oauth2.googleapis.com/token",
            "iat": now,
            "exp": now + 3600,
        }

        signed_jwt = jwt.encode(
            payload,
            sa_info["private_key"],
            algorithm="RS256",
        )

        # Exchange JWT for access token
        client = await self._get_client()
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": signed_jwt,
            },
        )
        response.raise_for_status()

        return response.json()["access_token"]

    async def _get_oauth_token(self) -> str:
        """Get access token from OAuth token file."""
        with open(self.config.token_file, "r") as f:
            token_data = json.load(f)
        return token_data.get("access_token") or token_data.get("token")

    def _get_headers(self, token: str) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

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
        samples_dir = connector_dir.parent / "examples" / "google_drive_samples"
        if samples_dir.exists():
            return samples_dir

        # Fallback to current working directory
        return Path("examples") / "google_drive_samples"

    def _guess_mime_type(self, extension: str) -> str:
        """Guess MIME type from file extension."""
        mime_types = {
            ".txt": "text/plain",
            ".json": "application/json",
            ".jsonl": "application/jsonl",
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".pdf": "application/pdf",
        }
        return mime_types.get(extension.lower(), "application/octet-stream")

    async def list_files(
        self,
        folder_id: Optional[str] = None,
        query: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> List[DriveFile]:
        """
        List files in Google Drive asynchronously.

        Args:
            folder_id: ID of folder to list (None for root).
            query: Custom search query (overrides folder_id).
            page_size: Number of results per page.

        Returns:
            List of DriveFile objects.
        """
        if self.config.mock_mode:
            return await self._list_files_mock(folder_id)
        return await self._list_files_real(folder_id, query, page_size)

    async def _list_files_mock(self, folder_id: Optional[str] = None) -> List[DriveFile]:
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
        # Use async file listing if aiofiles available, otherwise sync
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

    async def _list_files_real(
        self,
        folder_id: Optional[str] = None,
        query: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> List[DriveFile]:
        """List files using Google Drive API asynchronously."""
        client = await self._get_client()
        token = await self._get_access_token()
        headers = self._get_headers(token)

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
            params = {
                "pageSize": page_size,
                "fields": "nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, parents, webViewLink)",
            }
            if query:
                params["q"] = query
            if page_token:
                params["pageToken"] = page_token

            response = await client.get(
                f"{self.BASE_URL}/files",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            results = response.json()

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

    async def get_file_content(self, file_id: str) -> str:
        """
        Get the text content of a file asynchronously.

        Args:
            file_id: ID of the file to download.

        Returns:
            Text content of the file.

        Raises:
            FileNotFoundError: If file not found (mock mode).
            RuntimeError: If download fails.
        """
        if self.config.mock_mode:
            return await self._get_file_content_mock(file_id)
        return await self._get_file_content_real(file_id)

    async def _get_file_content_mock(self, file_id: str) -> str:
        """Get file content from mock samples."""
        samples_dir = self._get_samples_dir()

        # Extensions to try (including jsonl which is common for training data)
        extensions = ["", ".txt", ".json", ".jsonl", ".md", ".csv"]

        # Try exact match first
        for ext in extensions:
            file_path = samples_dir / f"{file_id}{ext}"
            if file_path.exists():
                if _AIOFILES_AVAILABLE:
                    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                        return await f.read()
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return f.read()

        # Try searching in subdirectories
        for subdir in samples_dir.iterdir():
            if subdir.is_dir():
                for ext in extensions:
                    file_path = subdir / f"{file_id}{ext}"
                    if file_path.exists():
                        if _AIOFILES_AVAILABLE:
                            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                                return await f.read()
                        else:
                            with open(file_path, "r", encoding="utf-8") as f:
                                return f.read()

        available = [f.stem for f in samples_dir.rglob("*") if f.is_file()]
        raise FileNotFoundError(
            f"Mock file '{file_id}' not found. "
            f"Available samples: {available[:10]}{'...' if len(available) > 10 else ''}"
        )

    async def _get_file_content_real(self, file_id: str) -> str:
        """Get file content using Google Drive API asynchronously."""
        client = await self._get_client()
        token = await self._get_access_token()
        headers = self._get_headers(token)

        # Get file metadata to determine type
        response = await client.get(
            f"{self.BASE_URL}/files/{file_id}",
            params={"fields": "mimeType,name"},
            headers=headers,
        )
        response.raise_for_status()
        file_meta = response.json()
        mime_type = file_meta["mimeType"]

        # Handle Google Workspace files (need export)
        if mime_type in self.config.export_formats:
            export_mime = self.config.export_formats[mime_type]
            response = await client.get(
                f"{self.BASE_URL}/files/{file_id}/export",
                params={"mimeType": export_mime},
                headers=headers,
            )
        else:
            response = await client.get(
                f"{self.BASE_URL}/files/{file_id}",
                params={"alt": "media"},
                headers=headers,
            )

        response.raise_for_status()
        return response.text

    async def get_file_metadata(self, file_id: str) -> DriveFile:
        """
        Get metadata for a file asynchronously.

        Args:
            file_id: ID of the file.

        Returns:
            DriveFile object with metadata.
        """
        if self.config.mock_mode:
            files = await self._list_files_mock()
            for f in files:
                if f.id == file_id:
                    return f
            raise FileNotFoundError(f"File not found: {file_id}")

        client = await self._get_client()
        token = await self._get_access_token()
        headers = self._get_headers(token)

        response = await client.get(
            f"{self.BASE_URL}/files/{file_id}",
            params={"fields": "id, name, mimeType, size, createdTime, modifiedTime, parents, webViewLink"},
            headers=headers,
        )
        response.raise_for_status()
        item = response.json()

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

    async def stream_samples(
        self,
        folder_id: str,
        mapping: Dict[str, str],
        file_pattern: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream training samples from files in a Google Drive folder asynchronously.

        Each file is expected to contain JSON or JSONL data that can be
        mapped to training samples.

        Args:
            folder_id: ID of the folder containing training files.
            mapping: Column mapping dict with "input" and "output" keys.
            file_pattern: Optional filename pattern to filter files.

        Yields:
            Training sample dicts with "input", "output", and "metadata" keys.
        """
        files = await self.list_files(folder_id)

        for file in files:
            # Filter by pattern if specified
            if file_pattern and file_pattern not in file.name:
                continue

            # Skip unsupported file types
            if file.mime_type not in self.config.mime_types and not file.name.endswith((".json", ".jsonl", ".txt")):
                continue

            try:
                content = await self.get_file_content(file.id)

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
                logger.error(f"Error processing file {file.name}: {e}")
                continue

    async def fetch_files_concurrent(
        self,
        file_ids: List[str],
        max_concurrency: int = 5,
    ) -> Dict[str, str]:
        """
        Fetch multiple files concurrently.

        Args:
            file_ids: List of file IDs to fetch.
            max_concurrency: Maximum number of concurrent requests.

        Returns:
            Dictionary mapping file_id to content.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def fetch_one(file_id: str) -> tuple:
            async with semaphore:
                try:
                    content = await self.get_file_content(file_id)
                    return file_id, content
                except Exception as e:
                    logger.error(f"Failed to fetch {file_id}: {e}")
                    return file_id, None

        tasks = [fetch_one(fid) for fid in file_ids]
        results = await asyncio.gather(*tasks)

        return {fid: content for fid, content in results if content is not None}
