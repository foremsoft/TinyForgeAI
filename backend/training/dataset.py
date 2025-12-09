"""
Dataset loading utilities for TinyForgeAI training.

Provides functions to load and validate JSONL training data files
containing input/output pairs for model training.
"""

import json
from typing import Dict, Generator, Iterable, List


def load_jsonl(path: str) -> List[Dict]:
    """
    Load a JSONL file and return all records as a list.

    Each line must be a JSON object with at least 'input' and 'output' keys.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dictionaries parsed from the file.

    Raises:
        ValueError: If a record is missing required 'input' or 'output' keys.
        FileNotFoundError: If the file does not exist.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")

            if "input" not in record or "output" not in record:
                raise ValueError(
                    f"Invalid record at line {line_num}: missing keys 'input' or 'output'"
                )
            records.append(record)
    return records


def stream_jsonl(path: str) -> Generator[Dict, None, None]:
    """
    Stream records from a JSONL file one by one.

    Useful for processing large files without loading everything into memory.

    Args:
        path: Path to the JSONL file.

    Yields:
        Dictionary records parsed from each line.

    Raises:
        ValueError: If a record is missing required 'input' or 'output' keys.
        FileNotFoundError: If the file does not exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")

            if "input" not in record or "output" not in record:
                raise ValueError(
                    f"Invalid record at line {line_num}: missing keys 'input' or 'output'"
                )
            yield record


def summarize_dataset(records: Iterable[Dict]) -> Dict:
    """
    Compute summary statistics for a dataset.

    Args:
        records: Iterable of record dictionaries with 'input' and 'output' keys.

    Returns:
        Dictionary with:
        - n_records: Total number of records
        - avg_input_len: Average input length (whitespace-tokenized word count)
        - avg_output_len: Average output length (whitespace-tokenized word count)
    """
    n_records = 0
    total_input_len = 0
    total_output_len = 0

    for record in records:
        n_records += 1
        total_input_len += len(record["input"].split())
        total_output_len += len(record["output"].split())

    if n_records == 0:
        return {"n_records": 0, "avg_input_len": 0.0, "avg_output_len": 0.0}

    return {
        "n_records": n_records,
        "avg_input_len": total_input_len / n_records,
        "avg_output_len": total_output_len / n_records,
    }
