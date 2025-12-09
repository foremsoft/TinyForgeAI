"""
Utility functions for the training module.

Provides common helpers for file operations and timestamp generation.
"""

import os
from datetime import datetime, timezone


def ensure_dir(path: str) -> None:
    """
    Create directory and parent directories if they don't exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def iso_now_utc() -> str:
    """
    Get current UTC timestamp in ISO8601 format.

    Returns:
        ISO8601 formatted UTC timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()
