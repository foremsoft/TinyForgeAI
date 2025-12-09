"""
Mapping utilities for converting database rows to training samples.

Provides functions to transform database rows into the standard
training sample format used by TinyForgeAI.
"""

from typing import Any, Mapping


def row_to_sample(row: Mapping[str, Any], mapping: Mapping[str, str]) -> dict:
    """
    Convert a database row to a training sample dict.

    Args:
        row: A mapping (dict-like) representing a database row.
             Supports dict, sqlite3.Row, and similar mapping types.
        mapping: A mapping defining column names for input/output.
                 Must contain keys "input" and "output" with column name values.
                 Example: {"input": "question_col", "output": "answer_col"}

    Returns:
        A training sample dict with structure:
        {
            "input": <str from input column>,
            "output": <str from output column>,
            "metadata": {
                "source": "db",
                "raw_row": <original row as dict>
            }
        }

    Raises:
        KeyError: If required columns are missing from the row or mapping.
    """
    # Validate mapping has required keys
    if "input" not in mapping:
        raise KeyError("Mapping must contain 'input' key specifying the input column")
    if "output" not in mapping:
        raise KeyError("Mapping must contain 'output' key specifying the output column")

    input_col = mapping["input"]
    output_col = mapping["output"]

    # Convert row to dict for consistent access and validation
    # This handles sqlite3.Row and other mapping types
    row_dict = dict(row)

    # Validate row has required columns
    if input_col not in row_dict:
        raise KeyError(f"Row missing required input column: '{input_col}'")
    if output_col not in row_dict:
        raise KeyError(f"Row missing required output column: '{output_col}'")

    return {
        "input": str(row_dict[input_col]),
        "output": str(row_dict[output_col]),
        "metadata": {
            "source": "db",
            "raw_row": row_dict,
        },
    }
