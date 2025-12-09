"""Tests for the database connector module."""

import sqlite3

import pytest

from connectors.db_connector import DBConnector


@pytest.fixture
def sample_db(tmp_path):
    """Create a temporary SQLite database with sample data."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create table
    cursor.execute("""
        CREATE TABLE qa (
            question TEXT,
            answer TEXT
        )
    """)

    # Insert sample data (same as examples/sample_qna.jsonl)
    samples = [
        ("How do I reset my password?", "Go to Settings > Reset Password."),
        ("What is your refund policy?", "Refunds within 30 days with receipt."),
        ("How to contact support?", "Email support@example.com or open a ticket."),
    ]
    cursor.executemany("INSERT INTO qa VALUES (?, ?)", samples)
    conn.commit()
    conn.close()

    return f"sqlite:///{db_path}"


def test_db_connector_stream_samples_yields_3_samples(sample_db):
    """Test that stream_samples yields 3 samples from test DB."""
    connector = DBConnector(db_url=sample_db)
    query = "SELECT question, answer FROM qa"
    mapping = {"input": "question", "output": "answer"}

    samples = list(connector.stream_samples(query, mapping))

    assert len(samples) == 3


def test_db_connector_samples_have_required_keys(sample_db):
    """Test that each sample has input, output, and metadata keys."""
    connector = DBConnector(db_url=sample_db)
    query = "SELECT question, answer FROM qa"
    mapping = {"input": "question", "output": "answer"}

    for sample in connector.stream_samples(query, mapping):
        assert "input" in sample
        assert "output" in sample
        assert "metadata" in sample


def test_db_connector_metadata_source_is_db(sample_db):
    """Test that metadata.source is 'db'."""
    connector = DBConnector(db_url=sample_db)
    query = "SELECT question, answer FROM qa"
    mapping = {"input": "question", "output": "answer"}

    for sample in connector.stream_samples(query, mapping):
        assert sample["metadata"]["source"] == "db"


def test_db_connector_test_connection_returns_true(sample_db):
    """Test that test_connection returns True for valid DB."""
    connector = DBConnector(db_url=sample_db)

    assert connector.test_connection() is True


def test_db_connector_test_connection_returns_false_for_invalid_db():
    """Test that test_connection returns False for invalid DB."""
    connector = DBConnector(db_url="sqlite:///nonexistent/path/db.sqlite")

    assert connector.test_connection() is False


def test_db_connector_with_column_aliases(sample_db):
    """Test that column aliases work correctly."""
    connector = DBConnector(db_url=sample_db)
    query = "SELECT question AS q, answer AS a FROM qa"
    mapping = {"input": "q", "output": "a"}

    samples = list(connector.stream_samples(query, mapping))

    assert len(samples) == 3
    assert samples[0]["input"] == "How do I reset my password?"


def test_db_connector_batch_size(sample_db):
    """Test that batch_size parameter works."""
    connector = DBConnector(db_url=sample_db)
    query = "SELECT question, answer FROM qa"
    mapping = {"input": "question", "output": "answer"}

    # Use batch_size=1 to test batching
    samples = list(connector.stream_samples(query, mapping, batch_size=1))

    assert len(samples) == 3


def test_db_connector_uses_default_db_url():
    """Test that connector uses default db_settings.DB_URL when None."""
    connector = DBConnector(db_url=None)

    # Should not raise, just use default
    assert connector.db_url is not None


def test_db_connector_raw_row_in_metadata(sample_db):
    """Test that metadata includes raw_row with original data."""
    connector = DBConnector(db_url=sample_db)
    query = "SELECT question, answer FROM qa LIMIT 1"
    mapping = {"input": "question", "output": "answer"}

    samples = list(connector.stream_samples(query, mapping))

    assert len(samples) == 1
    raw_row = samples[0]["metadata"]["raw_row"]
    assert "question" in raw_row
    assert "answer" in raw_row
