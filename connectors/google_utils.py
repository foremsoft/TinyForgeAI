"""
Utility functions for Google Docs connector.

Provides helpers for text extraction and normalization from Google Docs content.
"""

import re
from html.parser import HTMLParser
from typing import List


class _TagStripper(HTMLParser):
    """Simple HTML parser that strips tags and collects text content."""

    def __init__(self):
        super().__init__()
        self.text_parts: List[str] = []

    def handle_data(self, data: str) -> None:
        self.text_parts.append(data)

    def get_text(self) -> str:
        return "".join(self.text_parts)


def extract_text_from_html(html: str) -> str:
    """
    Extract plain text from HTML content.

    Strips all HTML tags and returns the text content.
    Useful for processing Google Docs content that may be returned as HTML.

    Args:
        html: HTML string to extract text from.

    Returns:
        Plain text with all HTML tags removed.
    """
    stripper = _TagStripper()
    stripper.feed(html)
    return stripper.get_text()


def normalize_text(text: str) -> str:
    """
    Normalize text by cleaning up whitespace and removing control characters.

    - Replaces multiple consecutive whitespace characters with a single space
    - Removes control characters (except newlines and tabs)
    - Strips leading and trailing whitespace

    Args:
        text: Text string to normalize.

    Returns:
        Normalized text string.
    """
    # Remove control characters except newlines (\n), tabs (\t), and carriage returns (\r)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Replace multiple spaces/tabs with single space (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    return text.strip()
