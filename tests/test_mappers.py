"""Tests for the database mappers module."""

import pytest

from connectors.mappers import row_to_sample


def test_row_to_sample_basic():
    """Test basic row to sample conversion."""
    row = {"question": "What is 2+2?", "answer": "4"}
    mapping = {"input": "question", "output": "answer"}

    sample = row_to_sample(row, mapping)

    assert sample["input"] == "What is 2+2?"
    assert sample["output"] == "4"


def test_row_to_sample_includes_metadata():
    """Test that sample includes metadata with source and raw_row."""
    row = {"q": "Hello?", "a": "Hi!"}
    mapping = {"input": "q", "output": "a"}

    sample = row_to_sample(row, mapping)

    assert "metadata" in sample
    assert sample["metadata"]["source"] == "db"
    assert sample["metadata"]["raw_row"] == {"q": "Hello?", "a": "Hi!"}


def test_row_to_sample_preserves_extra_columns_in_raw_row():
    """Test that extra columns are preserved in raw_row."""
    row = {"question": "Q?", "answer": "A!", "category": "test", "id": 123}
    mapping = {"input": "question", "output": "answer"}

    sample = row_to_sample(row, mapping)

    assert sample["metadata"]["raw_row"]["category"] == "test"
    assert sample["metadata"]["raw_row"]["id"] == 123


def test_row_to_sample_converts_to_string():
    """Test that non-string values are converted to strings."""
    row = {"num": 42, "result": 3.14}
    mapping = {"input": "num", "output": "result"}

    sample = row_to_sample(row, mapping)

    assert sample["input"] == "42"
    assert sample["output"] == "3.14"


def test_row_to_sample_raises_on_missing_input_column():
    """Test that KeyError is raised for missing input column."""
    row = {"answer": "A!"}
    mapping = {"input": "question", "output": "answer"}

    with pytest.raises(KeyError, match="missing required input column"):
        row_to_sample(row, mapping)


def test_row_to_sample_raises_on_missing_output_column():
    """Test that KeyError is raised for missing output column."""
    row = {"question": "Q?"}
    mapping = {"input": "question", "output": "answer"}

    with pytest.raises(KeyError, match="missing required output column"):
        row_to_sample(row, mapping)


def test_row_to_sample_raises_on_missing_input_key_in_mapping():
    """Test that KeyError is raised when mapping lacks 'input' key."""
    row = {"question": "Q?", "answer": "A!"}
    mapping = {"output": "answer"}

    with pytest.raises(KeyError, match="Mapping must contain 'input' key"):
        row_to_sample(row, mapping)


def test_row_to_sample_raises_on_missing_output_key_in_mapping():
    """Test that KeyError is raised when mapping lacks 'output' key."""
    row = {"question": "Q?", "answer": "A!"}
    mapping = {"input": "question"}

    with pytest.raises(KeyError, match="Mapping must contain 'output' key"):
        row_to_sample(row, mapping)
