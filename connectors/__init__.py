"""TinyForgeAI Connectors package for data source integrations."""

from connectors.api_connector import APIConnector, APIConnectorConfig, PaginationConfig
from connectors.db_connector import DBConnector
from connectors.file_ingest import ingest_file, get_supported_formats
from connectors.google_docs_connector import fetch_doc_text, list_docs_in_folder
from connectors.google_drive_connector import GoogleDriveConnector, GoogleDriveConfig, DriveFile
from connectors.notion_connector import NotionConnector, NotionConfig, NotionPage, NotionDatabase

# Async connectors - import conditionally to avoid import errors if dependencies missing
try:
    from connectors.async_api_connector import AsyncAPIConnector
except ImportError:
    AsyncAPIConnector = None  # type: ignore

try:
    from connectors.async_db_connector import AsyncDBConnector
except ImportError:
    AsyncDBConnector = None  # type: ignore

__all__ = [
    # Sync connectors
    "APIConnector",
    "APIConnectorConfig",
    "PaginationConfig",
    "DBConnector",
    "ingest_file",
    "get_supported_formats",
    "fetch_doc_text",
    "list_docs_in_folder",
    # Google Drive
    "GoogleDriveConnector",
    "GoogleDriveConfig",
    "DriveFile",
    # Notion
    "NotionConnector",
    "NotionConfig",
    "NotionPage",
    "NotionDatabase",
    # Async connectors
    "AsyncAPIConnector",
    "AsyncDBConnector",
]
