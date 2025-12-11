"""
MCP Connector Tools

Tools for fetching data from various sources.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("tinyforge-mcp.connectors")


class ConnectorTools:
    """Data connector tools."""

    async def get_available_connectors(self) -> list[dict[str, Any]]:
        """Get list of available connectors."""
        return [
            {
                "type": "database",
                "name": "Database Connector",
                "description": "Connect to SQLite, PostgreSQL, or MySQL databases",
                "config_options": ["connection_string", "query", "table"]
            },
            {
                "type": "api",
                "name": "REST API Connector",
                "description": "Fetch data from REST APIs with pagination and auth support",
                "config_options": ["base_url", "api_key", "auth_type", "endpoints"]
            },
            {
                "type": "google_docs",
                "name": "Google Docs Connector",
                "description": "Extract text from Google Docs (mock mode available)",
                "config_options": ["document_id", "mock_mode"]
            },
            {
                "type": "google_drive",
                "name": "Google Drive Connector",
                "description": "Fetch files from Google Drive folders",
                "config_options": ["folder_id", "file_types", "mock_mode"]
            },
            {
                "type": "notion",
                "name": "Notion Connector",
                "description": "Extract content from Notion pages and databases",
                "config_options": ["api_key", "page_id", "database_id"]
            },
            {
                "type": "slack",
                "name": "Slack Connector",
                "description": "Fetch messages from Slack channels",
                "config_options": ["token", "channel_id", "limit"]
            },
            {
                "type": "confluence",
                "name": "Confluence Connector",
                "description": "Extract content from Confluence spaces and pages",
                "config_options": ["base_url", "username", "api_token", "space_key"]
            },
            {
                "type": "file",
                "name": "File Ingestion",
                "description": "Process local files (PDF, DOCX, TXT, MD, JSONL)",
                "config_options": ["file_path", "chunk_size"]
            }
        ]

    async def fetch_data(
        self,
        connector_type: str,
        output_path: str,
        config: Optional[dict] = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """
        Fetch data from a connector source.

        Args:
            connector_type: Type of connector to use
            output_path: Path to save fetched data
            config: Connector-specific configuration
            limit: Maximum records to fetch

        Returns:
            Fetch operation result
        """
        config = config or {}
        mock_mode = config.get("mock_mode", True)

        try:
            if connector_type == "database":
                return await self._fetch_database(output_path, config, limit)
            elif connector_type == "api":
                return await self._fetch_api(output_path, config, limit)
            elif connector_type == "google_docs":
                return await self._fetch_google_docs(output_path, config, mock_mode)
            elif connector_type == "google_drive":
                return await self._fetch_google_drive(output_path, config, mock_mode)
            elif connector_type == "notion":
                return await self._fetch_notion(output_path, config, limit)
            elif connector_type == "slack":
                return await self._fetch_slack(output_path, config, limit)
            elif connector_type == "confluence":
                return await self._fetch_confluence(output_path, config, limit)
            elif connector_type == "file":
                return await self.ingest_files(
                    config.get("file_path", "."),
                    output_path,
                    config.get("chunk_size", 512)
                )
            else:
                return {
                    "success": False,
                    "error": f"Unknown connector type: {connector_type}"
                }
        except Exception as e:
            logger.error(f"Connector error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _fetch_database(
        self,
        output_path: str,
        config: dict,
        limit: int
    ) -> dict[str, Any]:
        """Fetch data from database."""
        try:
            from connectors.db_connector import DatabaseConnector, DBConnectorConfig

            db_config = DBConnectorConfig(
                connection_string=config.get("connection_string", "sqlite:///data.db"),
                query=config.get("query"),
                table=config.get("table"),
            )

            connector = DatabaseConnector(db_config)
            records = []

            for i, sample in enumerate(connector.stream_samples()):
                if i >= limit:
                    break
                records.append(sample)

            # Write to output file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            return {
                "success": True,
                "connector": "database",
                "records_fetched": len(records),
                "output_path": str(output_path)
            }

        except ImportError:
            return {
                "success": False,
                "error": "Database connector not available"
            }

    async def _fetch_api(
        self,
        output_path: str,
        config: dict,
        limit: int
    ) -> dict[str, Any]:
        """Fetch data from REST API."""
        try:
            from connectors.api_connector import APIConnector, APIConnectorConfig

            api_config = APIConnectorConfig(
                base_url=config.get("base_url"),
                auth_type=config.get("auth_type", "none"),
                auth_token=config.get("api_key"),
            )

            connector = APIConnector(api_config)
            records = []

            endpoint = config.get("endpoint", "/")
            mapping = config.get("mapping", {"input": "title", "output": "body"})

            for i, sample in enumerate(connector.stream_samples(endpoint, mapping)):
                if i >= limit:
                    break
                records.append(sample)

            # Write to output file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            return {
                "success": True,
                "connector": "api",
                "records_fetched": len(records),
                "output_path": str(output_path)
            }

        except ImportError:
            return {
                "success": False,
                "error": "API connector not available"
            }

    async def _fetch_google_docs(
        self,
        output_path: str,
        config: dict,
        mock_mode: bool
    ) -> dict[str, Any]:
        """Fetch data from Google Docs."""
        try:
            from connectors.google_docs import GoogleDocsConnector

            connector = GoogleDocsConnector(mock_mode=mock_mode)
            doc_id = config.get("document_id", "sample")

            content = connector.fetch_document(doc_id)

            # Write to output file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                f.write(json.dumps({
                    "input": f"Document: {doc_id}",
                    "output": content,
                    "metadata": {"source": "google_docs", "doc_id": doc_id}
                }) + "\n")

            return {
                "success": True,
                "connector": "google_docs",
                "mode": "mock" if mock_mode else "real",
                "records_fetched": 1,
                "output_path": str(output_path)
            }

        except ImportError:
            return {
                "success": False,
                "error": "Google Docs connector not available"
            }

    async def _fetch_google_drive(
        self,
        output_path: str,
        config: dict,
        mock_mode: bool
    ) -> dict[str, Any]:
        """Fetch files from Google Drive."""
        try:
            from connectors.google_drive_connector import GoogleDriveConnector

            connector = GoogleDriveConnector(mock_mode=mock_mode)
            folder_id = config.get("folder_id", "root")

            records = list(connector.stream_samples(folder_id=folder_id))

            # Write to output file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            return {
                "success": True,
                "connector": "google_drive",
                "mode": "mock" if mock_mode else "real",
                "records_fetched": len(records),
                "output_path": str(output_path)
            }

        except ImportError:
            return {
                "success": False,
                "error": "Google Drive connector not available"
            }

    async def _fetch_notion(
        self,
        output_path: str,
        config: dict,
        limit: int
    ) -> dict[str, Any]:
        """Fetch content from Notion."""
        try:
            from connectors.notion_connector import NotionConnector, NotionConfig

            notion_config = NotionConfig(
                api_key=config.get("api_key", ""),
                mock_mode=config.get("mock_mode", True)
            )

            connector = NotionConnector(notion_config)
            records = []

            for i, sample in enumerate(connector.stream_samples()):
                if i >= limit:
                    break
                records.append(sample)

            # Write to output file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            return {
                "success": True,
                "connector": "notion",
                "records_fetched": len(records),
                "output_path": str(output_path)
            }

        except ImportError:
            return {
                "success": False,
                "error": "Notion connector not available"
            }

    async def _fetch_slack(
        self,
        output_path: str,
        config: dict,
        limit: int
    ) -> dict[str, Any]:
        """Fetch messages from Slack."""
        try:
            from connectors.slack_connector import SlackConnector, SlackConfig

            slack_config = SlackConfig(
                token=config.get("token", ""),
                mock_mode=config.get("mock_mode", True)
            )

            connector = SlackConnector(slack_config)
            records = []

            channel_id = config.get("channel_id")
            for i, sample in enumerate(connector.stream_samples(channel_id=channel_id)):
                if i >= limit:
                    break
                records.append(sample)

            # Write to output file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            return {
                "success": True,
                "connector": "slack",
                "records_fetched": len(records),
                "output_path": str(output_path)
            }

        except ImportError:
            return {
                "success": False,
                "error": "Slack connector not available"
            }

    async def _fetch_confluence(
        self,
        output_path: str,
        config: dict,
        limit: int
    ) -> dict[str, Any]:
        """Fetch pages from Confluence."""
        try:
            from connectors.confluence_connector import ConfluenceConnector, ConfluenceConfig

            conf_config = ConfluenceConfig(
                base_url=config.get("base_url", ""),
                username=config.get("username", ""),
                api_token=config.get("api_token", ""),
                mock_mode=config.get("mock_mode", True)
            )

            connector = ConfluenceConnector(conf_config)
            records = []

            space_key = config.get("space_key")
            for i, sample in enumerate(connector.stream_samples(space_key=space_key)):
                if i >= limit:
                    break
                records.append(sample)

            # Write to output file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            return {
                "success": True,
                "connector": "confluence",
                "records_fetched": len(records),
                "output_path": str(output_path)
            }

        except ImportError:
            return {
                "success": False,
                "error": "Confluence connector not available"
            }

    async def ingest_files(
        self,
        input_path: str,
        output_path: str,
        chunk_size: int = 512,
    ) -> dict[str, Any]:
        """
        Ingest documents and convert to training data format.

        Args:
            input_path: Path to file or directory
            output_path: Path to save converted data
            chunk_size: Size of text chunks

        Returns:
            Ingestion result
        """
        try:
            from backend.data_processing.file_ingest import FileIngest

            ingest = FileIngest()
            input_dir = Path(input_path)

            records = []

            if input_dir.is_file():
                files = [input_dir]
            else:
                files = list(input_dir.glob("**/*"))
                files = [f for f in files if f.suffix.lower() in [".pdf", ".docx", ".txt", ".md", ".jsonl"]]

            for file_path in files:
                try:
                    content = ingest.process(str(file_path))

                    # Chunk content
                    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

                    for i, chunk in enumerate(chunks):
                        records.append({
                            "input": f"Document: {file_path.name} (chunk {i+1})",
                            "output": chunk,
                            "metadata": {
                                "source": str(file_path),
                                "chunk_index": i,
                                "total_chunks": len(chunks)
                            }
                        })
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")

            # Write to output file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            return {
                "success": True,
                "files_processed": len(files),
                "records_created": len(records),
                "output_path": str(output_path)
            }

        except ImportError:
            return {
                "success": False,
                "error": "File ingestion not available"
            }

    async def index_documents(
        self,
        input_path: str,
        index_name: str = "default",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> dict[str, Any]:
        """
        Index documents for RAG search.

        Args:
            input_path: Path to documents
            index_name: Name for the index
            embedding_model: Sentence transformer model

        Returns:
            Indexing result
        """
        try:
            from connectors.indexer import DocumentIndexer, IndexerConfig

            config = IndexerConfig(
                index_name=index_name,
                embedding_model=embedding_model,
            )

            indexer = DocumentIndexer(config)

            input_dir = Path(input_path)
            if input_dir.is_file():
                indexer.index_file(str(input_dir))
                files_indexed = 1
            else:
                files = list(input_dir.glob("**/*"))
                files = [f for f in files if f.is_file()]
                for f in files:
                    try:
                        indexer.index_file(str(f))
                    except Exception as e:
                        logger.warning(f"Failed to index {f}: {e}")
                files_indexed = len(files)

            return {
                "success": True,
                "index_name": index_name,
                "files_indexed": files_indexed,
                "embedding_model": embedding_model
            }

        except ImportError:
            return {
                "success": False,
                "error": "RAG indexer not available. Install with: pip install sentence-transformers"
            }

    async def search_documents(
        self,
        query: str,
        index_name: str = "default",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Search indexed documents.

        Args:
            query: Search query
            index_name: Name of index to search
            top_k: Number of results

        Returns:
            Search results
        """
        try:
            from connectors.indexer import DocumentIndexer, IndexerConfig

            config = IndexerConfig(index_name=index_name)
            indexer = DocumentIndexer(config)

            results = indexer.search(query, top_k=top_k)

            return {
                "success": True,
                "query": query,
                "results": [
                    {
                        "score": r.score,
                        "content": r.document.content[:500],
                        "metadata": r.document.metadata
                    }
                    for r in results
                ]
            }

        except ImportError:
            return {
                "success": False,
                "error": "RAG indexer not available"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
