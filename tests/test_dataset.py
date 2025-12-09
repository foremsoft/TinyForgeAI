"""Tests for the dataset loading utilities."""

import os

import pytest

from backend.training.dataset import load_jsonl, stream_jsonl, summarize_dataset

SAMPLE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "examples", "sample_qna.jsonl"
)


def test_load_jsonl_returns_3_records():
    """Test that load_jsonl loads exactly 3 records from sample file."""
    records = load_jsonl(SAMPLE_FILE)
    assert len(records) == 3


def test_load_jsonl_records_have_required_keys():
    """Test that all records have input and output keys."""
    records = load_jsonl(SAMPLE_FILE)
    for record in records:
        assert "input" in record
        assert "output" in record


def test_stream_jsonl_yields_3_records():
    """Test that stream_jsonl yields exactly 3 records."""
    records = list(stream_jsonl(SAMPLE_FILE))
    assert len(records) == 3


def test_stream_jsonl_records_have_required_keys():
    """Test that streamed records have input and output keys."""
    for record in stream_jsonl(SAMPLE_FILE):
        assert "input" in record
        assert "output" in record


def test_summarize_dataset_returns_correct_n_records():
    """Test that summarize_dataset returns n_records == 3."""
    records = load_jsonl(SAMPLE_FILE)
    summary = summarize_dataset(records)
    assert summary["n_records"] == 3


def test_summarize_dataset_returns_positive_avg_lengths():
    """Test that summarize_dataset returns positive average lengths."""
    records = load_jsonl(SAMPLE_FILE)
    summary = summarize_dataset(records)
    assert summary["avg_input_len"] > 0
    assert summary["avg_output_len"] > 0


def test_summarize_dataset_empty_returns_zeros():
    """Test that summarize_dataset handles empty dataset."""
    summary = summarize_dataset([])
    assert summary["n_records"] == 0
    assert summary["avg_input_len"] == 0.0
    assert summary["avg_output_len"] == 0.0


def test_load_jsonl_raises_on_missing_file():
    """Test that load_jsonl raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_jsonl("/nonexistent/path/data.jsonl")


def test_load_jsonl_raises_on_invalid_record(tmp_path):
    """Test that load_jsonl raises ValueError for records missing keys."""
    invalid_file = tmp_path / "invalid.jsonl"
    invalid_file.write_text('{"input": "test"}\n')  # Missing output key

    with pytest.raises(ValueError, match="missing keys"):
        load_jsonl(str(invalid_file))
