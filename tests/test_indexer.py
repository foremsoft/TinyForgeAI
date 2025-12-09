"""
Tests for the RAG document indexer.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from connectors.indexer import (
    DocumentIndexer,
    IndexerConfig,
    TextChunker,
    Document,
    SearchResult,
)
from connectors import indexer as indexer_module


class TestIndexerConfig:
    """Test IndexerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IndexerConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding_dimension == 384

    def test_custom_config(self):
        """Test custom configuration."""
        config = IndexerConfig(
            chunk_size=256,
            chunk_overlap=25,
            embedding_model="custom-model",
        )
        assert config.chunk_size == 256
        assert config.chunk_overlap == 25
        assert config.embedding_model == "custom-model"


class TestTextChunker:
    """Test TextChunker class."""

    def test_basic_chunking(self):
        """Test basic text chunking."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "A" * 100

        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk("")
        assert chunks == []

    def test_short_text(self):
        """Test chunking text shorter than chunk_size."""
        chunker = TextChunker(chunk_size=100, overlap=10)
        text = "Short text"

        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_sentence_boundary_breaking(self):
        """Test that chunker tries to break at sentence boundaries."""
        chunker = TextChunker(chunk_size=50, overlap=5)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunker.chunk(text)
        # Should have multiple chunks
        assert len(chunks) >= 1

    def test_chunk_with_metadata(self):
        """Test chunk_with_metadata creates Document objects."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "This is a test document with enough text to create multiple chunks."

        docs = chunker.chunk_with_metadata(
            text,
            source="test.txt",
            metadata={"author": "test"}
        )

        assert len(docs) >= 1
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.id is not None
            assert doc.content is not None
            assert doc.metadata["source"] == "test.txt"
            assert doc.metadata["author"] == "test"
            assert "chunk_index" in doc.metadata


class TestDocument:
    """Test Document dataclass."""

    def test_document_creation(self):
        """Test creating a Document."""
        doc = Document(
            id="test123",
            content="Test content",
            metadata={"source": "test"},
        )
        assert doc.id == "test123"
        assert doc.content == "Test content"
        assert doc.metadata["source"] == "test"
        assert doc.embedding is None

    def test_document_with_embedding(self):
        """Test Document with embedding."""
        doc = Document(
            id="test123",
            content="Test",
            embedding=[0.1, 0.2, 0.3]
        )
        assert doc.embedding == [0.1, 0.2, 0.3]


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating SearchResult."""
        doc = Document(id="test", content="content")
        result = SearchResult(document=doc, score=0.95)

        assert result.document.id == "test"
        assert result.score == 0.95


class TestDocumentIndexerBasic:
    """Basic tests for DocumentIndexer without dependencies."""

    def test_indexer_creation(self):
        """Test indexer creation."""
        indexer = DocumentIndexer()
        assert indexer.config is not None
        assert indexer.documents == {}

    def test_indexer_with_custom_config(self, tmp_path):
        """Test indexer with custom config."""
        config = IndexerConfig(
            chunk_size=256,
            index_path=str(tmp_path / "index")
        )
        indexer = DocumentIndexer(config)

        assert indexer.config.chunk_size == 256
        assert Path(tmp_path / "index").exists()

    def test_availability_flags(self):
        """Test that availability flags exist."""
        assert hasattr(indexer_module, "EMBEDDINGS_AVAILABLE")
        assert hasattr(indexer_module, "NUMPY_AVAILABLE")


class TestDocumentIndexerWithMocks:
    """Tests with mocked dependencies."""

    @pytest.fixture
    def mock_embeddings(self):
        """Mock embedding model."""
        with patch.object(indexer_module, "EMBEDDINGS_AVAILABLE", True):
            with patch.object(indexer_module, "NUMPY_AVAILABLE", True):
                mock_model = MagicMock()
                mock_model.encode.return_value = [
                    [0.1] * 384,
                    [0.2] * 384,
                ]
                yield mock_model

    def test_indexer_index_text_without_embeddings(self, tmp_path):
        """Test that indexing fails without embedding library."""
        # Save original
        original = indexer_module.EMBEDDINGS_AVAILABLE

        try:
            indexer_module.EMBEDDINGS_AVAILABLE = False
            config = IndexerConfig(index_path=str(tmp_path / "index"))
            indexer = DocumentIndexer(config)

            with pytest.raises(RuntimeError) as excinfo:
                indexer.index_text("Test text")

            assert "sentence-transformers" in str(excinfo.value)
        finally:
            indexer_module.EMBEDDINGS_AVAILABLE = original


class TestDocumentIndexerFile:
    """Test file handling."""

    @pytest.fixture
    def sample_files(self, tmp_path):
        """Create sample files for testing."""
        # Text file
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("This is a sample text file for testing.")

        # JSON file
        json_file = tmp_path / "sample.json"
        json_file.write_text('{"key": "value", "nested": {"a": 1}}')

        # JSONL file
        jsonl_file = tmp_path / "sample.jsonl"
        jsonl_file.write_text(
            '{"input": "hello", "output": "world"}\n'
            '{"input": "foo", "output": "bar"}\n'
        )

        return {
            "txt": txt_file,
            "json": json_file,
            "jsonl": jsonl_file,
        }

    def test_file_not_found(self, tmp_path):
        """Test handling of missing file."""
        config = IndexerConfig(index_path=str(tmp_path / "index"))
        indexer = DocumentIndexer(config)

        with pytest.raises(FileNotFoundError):
            indexer.index_file("/nonexistent/file.txt")

    def test_unsupported_file_type(self, tmp_path):
        """Test handling of unsupported file type."""
        config = IndexerConfig(index_path=str(tmp_path / "index"))
        indexer = DocumentIndexer(config)

        # Create a file with unsupported extension
        weird_file = tmp_path / "file.xyz"
        weird_file.write_text("content")

        with pytest.raises(ValueError) as excinfo:
            indexer.index_file(weird_file)

        assert "Unsupported file type" in str(excinfo.value)


class TestDocumentIndexerPersistence:
    """Test index persistence."""

    def test_save_and_load_empty_index(self, tmp_path):
        """Test saving and loading empty index."""
        config = IndexerConfig(index_path=str(tmp_path / "index"))
        indexer = DocumentIndexer(config)

        # Add a document manually (without embeddings)
        doc = Document(
            id="test1",
            content="Test content",
            metadata={"source": "test"},
            embedding=[0.1, 0.2, 0.3]
        )
        indexer.documents["test1"] = doc

        # Save
        indexer.save_index()

        # Verify files exist
        assert (tmp_path / "index" / "documents.json").exists()
        assert (tmp_path / "index" / "config.json").exists()

        # Load into new indexer
        config2 = IndexerConfig(index_path=str(tmp_path / "index"))
        indexer2 = DocumentIndexer(config2)
        indexer2.load_index()

        assert len(indexer2.documents) == 1
        assert "test1" in indexer2.documents
        assert indexer2.documents["test1"].content == "Test content"

    def test_load_nonexistent_index(self, tmp_path):
        """Test loading from nonexistent path."""
        config = IndexerConfig(index_path=str(tmp_path / "nonexistent"))
        indexer = DocumentIndexer(config)

        with pytest.raises(FileNotFoundError):
            indexer.load_index()


class TestDocumentIndexerStats:
    """Test statistics methods."""

    def test_get_stats(self, tmp_path):
        """Test get_stats method."""
        config = IndexerConfig(
            chunk_size=256,
            index_path=str(tmp_path / "index")
        )
        indexer = DocumentIndexer(config)

        stats = indexer.get_stats()

        assert stats["total_documents"] == 0
        assert stats["chunk_size"] == 256
        assert stats["embedding_dimension"] == 384

    def test_clear(self, tmp_path):
        """Test clear method."""
        config = IndexerConfig(index_path=str(tmp_path / "index"))
        indexer = DocumentIndexer(config)

        # Add document
        indexer.documents["test"] = Document(id="test", content="content")

        # Clear
        indexer.clear()

        assert len(indexer.documents) == 0
        assert indexer.embeddings is None


class TestCLI:
    """Test CLI entry point."""

    def test_main_function_exists(self):
        """Test main function exists."""
        from connectors.indexer import main
        assert callable(main)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    not indexer_module.EMBEDDINGS_AVAILABLE or not indexer_module.NUMPY_AVAILABLE,
    reason="Embedding dependencies not installed"
)
class TestDocumentIndexerIntegration:
    """Integration tests requiring actual dependencies."""

    def test_full_index_and_search(self, tmp_path):
        """Test full indexing and search flow."""
        config = IndexerConfig(
            chunk_size=100,
            chunk_overlap=20,
            index_path=str(tmp_path / "index")
        )
        indexer = DocumentIndexer(config)

        # Index some text
        text = """
        Machine learning is a subset of artificial intelligence.
        It enables computers to learn from data without being explicitly programmed.
        Deep learning is a specialized form of machine learning using neural networks.
        """

        doc_ids = indexer.index_text(text, source="ml_intro.txt")
        assert len(doc_ids) >= 1

        # Search
        results = indexer.search("What is machine learning?", top_k=3)
        assert len(results) >= 1
        assert results[0].score > 0

    def test_search_with_context(self, tmp_path):
        """Test search_with_context formatting."""
        config = IndexerConfig(
            chunk_size=50,
            index_path=str(tmp_path / "index")
        )
        indexer = DocumentIndexer(config)

        indexer.index_text("Test content for search.", source="test.txt")

        context = indexer.search_with_context("test", top_k=1)
        assert "score:" in context
        assert "Test content" in context
