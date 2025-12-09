"""
Google Docs connector for TinyForgeAI.

Provides functionality to fetch document text from Google Docs.
Includes a mock mode for offline testing and development.

Mock Mode:
    When GOOGLE_OAUTH_DISABLED=true (default), the connector reads from
    local sample files in examples/google_docs_samples/ instead of
    making real API calls.

Real Mode:
    When GOOGLE_OAUTH_DISABLED=false, the connector uses the Google Docs API
    with OAuth credentials to fetch real documents.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path when run as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from connectors.google_utils import normalize_text


def _is_mock_mode() -> bool:
    """Check if running in mock mode (GOOGLE_OAUTH_DISABLED is truthy)."""
    value = os.getenv("GOOGLE_OAUTH_DISABLED", "true").lower()
    return value in ("true", "1", "yes")


def _get_samples_dir() -> Path:
    """Get the path to the sample docs directory."""
    # Try relative to this file first
    connector_dir = Path(__file__).parent
    samples_dir = connector_dir.parent / "examples" / "google_docs_samples"
    if samples_dir.exists():
        return samples_dir

    # Fallback to current working directory
    return Path("examples") / "google_docs_samples"


def fetch_doc_text(doc_id: str, oauth_credentials: Optional[dict] = None) -> str:
    """
    Fetch and extract text content from a Google Doc.

    In mock mode (GOOGLE_OAUTH_DISABLED=true), reads from local sample files.
    In real mode, uses the Google Docs API with OAuth credentials.

    Args:
        doc_id: The Google Doc ID or sample doc name (in mock mode).
        oauth_credentials: Optional OAuth credentials dict for real mode.
                          Not used in mock mode.

    Returns:
        The text content of the document.

    Raises:
        FileNotFoundError: If document not found (mock mode).
        RuntimeError: If real mode called without proper setup.
    """
    if _is_mock_mode():
        return _fetch_doc_mock(doc_id)
    else:
        return _fetch_doc_real(doc_id, oauth_credentials)


def _fetch_doc_mock(doc_id: str) -> str:
    """
    Fetch document text from local sample files (mock mode).

    Args:
        doc_id: The sample document name (without .txt extension).

    Returns:
        The text content of the sample file.

    Raises:
        FileNotFoundError: If sample file not found.
    """
    samples_dir = _get_samples_dir()
    file_path = samples_dir / f"{doc_id}.txt"

    if not file_path.exists():
        available = [f.stem for f in samples_dir.glob("*.txt")] if samples_dir.exists() else []
        raise FileNotFoundError(
            f"Mock document '{doc_id}' not found. "
            f"Available samples: {available}. "
            f"Expected file: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return normalize_text(text)


def _fetch_doc_real(doc_id: str, oauth_credentials: Optional[dict] = None) -> str:
    """
    Fetch document text from Google Docs API (real mode).

    This is a placeholder implementation. To use real Google Docs API:

    1. Create OAuth credentials in Google Cloud Console
    2. Enable the Google Docs API
    3. Install google-api-python-client: pip install google-api-python-client
    4. Pass credentials to this function

    Example implementation (not active):
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_info(oauth_credentials)
        service = build('docs', 'v1', credentials=creds)
        document = service.documents().get(documentId=doc_id).execute()
        # Extract text from document['body']['content']

    Args:
        doc_id: The Google Doc ID.
        oauth_credentials: OAuth credentials dict.

    Raises:
        RuntimeError: Always raises in placeholder mode.
    """
    raise RuntimeError(
        "Real Google Docs API mode is not implemented. "
        "Set GOOGLE_OAUTH_DISABLED=true to use mock mode, or implement "
        "the Google Docs API integration with proper OAuth credentials. "
        "See docs/connectors.md for setup instructions."
    )


def list_docs_in_folder(folder_id: str) -> List[dict]:
    """
    List documents in a Google Drive folder.

    In mock mode, returns a list of sample documents.
    In real mode, would query the Google Drive API.

    Args:
        folder_id: The Google Drive folder ID (ignored in mock mode).

    Returns:
        List of document metadata dicts with 'id' and 'title' keys.
    """
    if _is_mock_mode():
        return _list_docs_mock()
    else:
        raise RuntimeError(
            "Real Google Drive API mode is not implemented. "
            "Set GOOGLE_OAUTH_DISABLED=true to use mock mode."
        )


def _list_docs_mock() -> List[dict]:
    """
    List available sample documents (mock mode).

    Returns:
        List of sample document metadata.
    """
    samples_dir = _get_samples_dir()

    if not samples_dir.exists():
        return []

    docs = []
    for file_path in sorted(samples_dir.glob("*.txt")):
        docs.append({
            "id": file_path.stem,
            "title": file_path.stem.replace("_", " ").title(),
        })

    return docs


def main() -> int:
    """CLI entry point for the Google Docs connector."""
    parser = argparse.ArgumentParser(
        description="Fetch text from a Google Doc (or sample in mock mode)."
    )
    parser.add_argument(
        "--doc-id",
        required=True,
        help="Google Doc ID or sample doc name (in mock mode).",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="List available sample documents (mock mode only).",
    )

    args = parser.parse_args()

    if args.list_samples:
        docs = list_docs_in_folder("mock")
        for doc in docs:
            print(f"{doc['id']}: {doc['title']}")
        return 0

    try:
        text = fetch_doc_text(args.doc_id)
        print(text)
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
