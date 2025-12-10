#!/usr/bin/env python3
"""
Document Indexing Script for Internal Knowledge Search

This script indexes documents from various sources (local files, Google Drive,
Notion, Confluence) into a vector index for semantic search.

Usage:
    python index_documents.py --source local --path ./data/sample_documents
    python index_documents.py --source gdrive --folder-id YOUR_FOLDER_ID
    python index_documents.py --source notion --database-id YOUR_DB_ID
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import yaml
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Using mock embeddings.")


class DocumentChunker:
    """Split documents into chunks for indexing."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        chunks = []
        text = text.strip()

        if len(text) <= self.chunk_size:
            return [{
                "text": text,
                "metadata": metadata or {},
                "chunk_index": 0
            }]

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at paragraph or sentence boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sep in [". ", "! ", "? ", "\n"]:
                        sent_break = text.rfind(sep, start, end)
                        if sent_break > start + self.chunk_size // 2:
                            end = sent_break + len(sep)
                            break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {**(metadata or {}), "chunk_index": chunk_index},
                    "chunk_index": chunk_index
                })
                chunk_index += 1

            start = end - self.overlap

        return chunks


class DocumentLoader:
    """Load documents from various sources."""

    @staticmethod
    def load_local(path: str, extensions: List[str] = None) -> List[Dict[str, Any]]:
        """Load documents from local filesystem."""
        extensions = extensions or [".txt", ".md", ".pdf", ".docx"]
        documents = []

        path = Path(path)
        if path.is_file():
            files = [path]
        else:
            files = []
            for ext in extensions:
                files.extend(path.rglob(f"*{ext}"))

        for file_path in files:
            try:
                content = DocumentLoader._read_file(file_path)
                if content:
                    documents.append({
                        "id": str(file_path),
                        "content": content,
                        "metadata": {
                            "source": "local",
                            "filename": file_path.name,
                            "path": str(file_path),
                            "extension": file_path.suffix
                        }
                    })
                    print(f"  Loaded: {file_path.name}")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")

        return documents

    @staticmethod
    def _read_file(path: Path) -> Optional[str]:
        """Read content from a file based on its extension."""
        ext = path.suffix.lower()

        if ext in [".txt", ".md"]:
            return path.read_text(encoding="utf-8")

        elif ext == ".pdf":
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(path))
                text = []
                for page in reader.pages:
                    text.append(page.extract_text() or "")
                return "\n\n".join(text)
            except ImportError:
                print(f"    PyPDF2 not installed, skipping {path.name}")
                return None

        elif ext == ".docx":
            try:
                from docx import Document
                doc = Document(str(path))
                return "\n\n".join([p.text for p in doc.paragraphs])
            except ImportError:
                print(f"    python-docx not installed, skipping {path.name}")
                return None

        return None

    @staticmethod
    def load_gdrive(folder_id: str, credentials_path: str = None) -> List[Dict[str, Any]]:
        """Load documents from Google Drive folder."""
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseDownload
            import io
        except ImportError:
            print("Google API libraries not installed. Run: pip install google-api-python-client")
            return []

        # This is a placeholder - implement actual GDrive loading
        print(f"Loading from Google Drive folder: {folder_id}")
        print("Note: Implement Google Drive authentication for production use")
        return []

    @staticmethod
    def load_notion(database_id: str, token: str = None) -> List[Dict[str, Any]]:
        """Load documents from Notion database."""
        try:
            from notion_client import Client
        except ImportError:
            print("Notion client not installed. Run: pip install notion-client")
            return []

        # This is a placeholder - implement actual Notion loading
        print(f"Loading from Notion database: {database_id}")
        print("Note: Implement Notion authentication for production use")
        return []

    @staticmethod
    def load_confluence(url: str, space_key: str, username: str = None, api_token: str = None) -> List[Dict[str, Any]]:
        """Load documents from Confluence space."""
        try:
            from atlassian import Confluence
        except ImportError:
            print("Atlassian library not installed. Run: pip install atlassian-python-api")
            return []

        # This is a placeholder - implement actual Confluence loading
        print(f"Loading from Confluence space: {space_key}")
        print("Note: Implement Confluence authentication for production use")
        return []


class VectorIndex:
    """Simple vector index for semantic search."""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embeddings = []
        self.documents = []
        self.model = None

        if HAS_SENTENCE_TRANSFORMERS:
            print(f"Loading embedding model: {embedding_model}")
            self.model = SentenceTransformer(embedding_model)
        else:
            print("Using mock embeddings (install sentence-transformers for real embeddings)")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if self.model:
            return self.model.encode(text, convert_to_numpy=True)
        else:
            # Mock embedding for testing
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(384).astype(np.float32)

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to the index."""
        print(f"Indexing {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            embedding = self._get_embedding(text)

            self.embeddings.append(embedding)
            self.documents.append({
                "id": f"doc_{len(self.documents)}",
                "text": text,
                "metadata": chunk.get("metadata", {})
            })

            if (i + 1) % 10 == 0:
                print(f"  Indexed {i + 1}/{len(chunks)} chunks")

        print(f"Total documents in index: {len(self.documents)}")

    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        embeddings_array = np.array(self.embeddings)
        np.save(path / "embeddings.npy", embeddings_array)

        # Save documents
        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)

        # Save metadata
        metadata = {
            "embedding_model": self.embedding_model_name,
            "num_documents": len(self.documents),
            "embedding_dim": embeddings_array.shape[1] if len(embeddings_array) > 0 else 0,
            "created_at": datetime.utcnow().isoformat()
        }
        with open(path / "index_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Index saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        """Load index from disk."""
        path = Path(path)

        with open(path / "index_metadata.json") as f:
            metadata = json.load(f)

        index = cls(embedding_model=metadata["embedding_model"])
        index.embeddings = list(np.load(path / "embeddings.npy"))

        with open(path / "documents.json", encoding="utf-8") as f:
            index.documents = json.load(f)

        print(f"Loaded index with {len(index.documents)} documents")
        return index


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "search_config.yaml"

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Default config
    return {
        "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        "chunking": {"size": 500, "overlap": 50},
        "index": {"path": "./data/knowledge_index"}
    }


def main():
    parser = argparse.ArgumentParser(description="Index documents for semantic search")
    parser.add_argument("--source", choices=["local", "gdrive", "notion", "confluence"],
                        default="local", help="Document source")
    parser.add_argument("--path", help="Path to local documents")
    parser.add_argument("--folder-id", help="Google Drive folder ID")
    parser.add_argument("--credentials", help="Path to credentials file")
    parser.add_argument("--database-id", help="Notion database ID")
    parser.add_argument("--token", help="API token (Notion)")
    parser.add_argument("--url", help="Confluence URL")
    parser.add_argument("--space-key", help="Confluence space key")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--output", help="Output index path")

    args = parser.parse_args()
    config = load_config(args.config)

    # Load documents based on source
    print(f"\n=== Loading documents from {args.source} ===\n")

    if args.source == "local":
        path = args.path or "./data/sample_documents"
        documents = DocumentLoader.load_local(path)
    elif args.source == "gdrive":
        if not args.folder_id:
            print("Error: --folder-id required for Google Drive")
            sys.exit(1)
        documents = DocumentLoader.load_gdrive(args.folder_id, args.credentials)
    elif args.source == "notion":
        if not args.database_id:
            print("Error: --database-id required for Notion")
            sys.exit(1)
        token = args.token or os.environ.get("NOTION_TOKEN")
        documents = DocumentLoader.load_notion(args.database_id, token)
    elif args.source == "confluence":
        if not args.url or not args.space_key:
            print("Error: --url and --space-key required for Confluence")
            sys.exit(1)
        documents = DocumentLoader.load_confluence(
            args.url, args.space_key,
            os.environ.get("CONFLUENCE_USER"),
            os.environ.get("CONFLUENCE_TOKEN")
        )

    if not documents:
        print("No documents loaded. Exiting.")
        sys.exit(1)

    print(f"\nLoaded {len(documents)} documents")

    # Chunk documents
    print(f"\n=== Chunking documents ===\n")

    chunker = DocumentChunker(
        chunk_size=config["chunking"]["size"],
        overlap=config["chunking"]["overlap"]
    )

    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_text(doc["content"], doc["metadata"])
        all_chunks.extend(chunks)
        print(f"  {doc['metadata'].get('filename', 'doc')}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Create and populate index
    print(f"\n=== Building vector index ===\n")

    index = VectorIndex(embedding_model=config["embedding"]["model"])
    index.add_documents(all_chunks)

    # Save index
    output_path = args.output or config["index"]["path"]
    index.save(output_path)

    print(f"\n=== Indexing complete! ===")
    print(f"Index location: {output_path}")
    print(f"Total chunks indexed: {len(all_chunks)}")


if __name__ == "__main__":
    main()
