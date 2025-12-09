"""
File ingestion utilities for TinyForgeAI.

Provides functions to extract text content from various file formats:
- TXT: Plain text files
- MD: Markdown files (returned as-is, no rendering)
- DOCX: Microsoft Word documents (requires python-docx)
- PDF: PDF documents (requires PyMuPDF or pdfminer.six)

Usage:
    from connectors.file_ingest import ingest_file

    text = ingest_file("path/to/document.pdf")
"""

import os
from pathlib import Path
from typing import Optional

# Track optional dependency availability
_DOCX_AVAILABLE = False
_PYMUPDF_AVAILABLE = False
_PDFMINER_AVAILABLE = False

try:
    from docx import Document

    _DOCX_AVAILABLE = True
except ImportError:
    pass

try:
    import fitz  # PyMuPDF

    _PYMUPDF_AVAILABLE = True
except ImportError:
    pass

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text

    _PDFMINER_AVAILABLE = True
except ImportError:
    pass


def ingest_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Extract text content from a file.

    Supports TXT, MD, DOCX, and PDF formats. For DOCX and PDF files,
    optional dependencies are required.

    Args:
        file_path: Path to the file to ingest.
        encoding: Text encoding for TXT/MD files (default: utf-8).

    Returns:
        Extracted text content from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
        RuntimeError: If required optional dependency is not installed.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return _ingest_txt(path, encoding)
    elif suffix == ".md":
        return _ingest_md(path, encoding)
    elif suffix == ".docx":
        return _ingest_docx(path)
    elif suffix == ".pdf":
        return _ingest_pdf(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .txt, .md, .docx, .pdf"
        )


def _ingest_txt(path: Path, encoding: str = "utf-8") -> str:
    """
    Ingest a plain text file.

    Args:
        path: Path to the TXT file.
        encoding: Text encoding (default: utf-8).

    Returns:
        Text content of the file.
    """
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def _ingest_md(path: Path, encoding: str = "utf-8") -> str:
    """
    Ingest a Markdown file.

    Returns the raw markdown content without rendering.

    Args:
        path: Path to the MD file.
        encoding: Text encoding (default: utf-8).

    Returns:
        Raw markdown content of the file.
    """
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def _ingest_docx(path: Path) -> str:
    """
    Ingest a Microsoft Word document.

    Requires the python-docx library.

    Args:
        path: Path to the DOCX file.

    Returns:
        Extracted text content from the document.

    Raises:
        RuntimeError: If python-docx is not installed.
    """
    if not _DOCX_AVAILABLE:
        raise RuntimeError(
            "DOCX support requires the python-docx library. "
            "Install it with: pip install python-docx"
        )

    doc = Document(str(path))
    paragraphs = [para.text for para in doc.paragraphs]
    return "\n".join(paragraphs)


def _ingest_pdf(path: Path) -> str:
    """
    Ingest a PDF document.

    Tries PyMuPDF first, then falls back to pdfminer.six.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text content from the PDF.

    Raises:
        RuntimeError: If no PDF library is installed.
    """
    if _PYMUPDF_AVAILABLE:
        return _ingest_pdf_pymupdf(path)
    elif _PDFMINER_AVAILABLE:
        return _ingest_pdf_pdfminer(path)
    else:
        raise RuntimeError(
            "PDF support requires PyMuPDF or pdfminer.six. "
            "Install with: pip install PyMuPDF  (or: pip install pdfminer.six)"
        )


def _ingest_pdf_pymupdf(path: Path) -> str:
    """
    Ingest a PDF using PyMuPDF (fitz).

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text content.
    """
    import fitz

    doc = fitz.open(str(path))
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def _ingest_pdf_pdfminer(path: Path) -> str:
    """
    Ingest a PDF using pdfminer.six.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text content.
    """
    from pdfminer.high_level import extract_text

    return extract_text(str(path))


def get_supported_formats() -> dict:
    """
    Get information about supported file formats and their availability.

    Returns:
        Dictionary with format extensions as keys and availability info as values.
    """
    return {
        ".txt": {"available": True, "requires": None},
        ".md": {"available": True, "requires": None},
        ".docx": {"available": _DOCX_AVAILABLE, "requires": "python-docx"},
        ".pdf": {
            "available": _PYMUPDF_AVAILABLE or _PDFMINER_AVAILABLE,
            "requires": "PyMuPDF or pdfminer.six",
            "pymupdf": _PYMUPDF_AVAILABLE,
            "pdfminer": _PDFMINER_AVAILABLE,
        },
    }


def check_dependencies() -> dict:
    """
    Check availability of optional dependencies.

    Returns:
        Dictionary with library names as keys and availability as values.
    """
    return {
        "python-docx": _DOCX_AVAILABLE,
        "PyMuPDF": _PYMUPDF_AVAILABLE,
        "pdfminer.six": _PDFMINER_AVAILABLE,
    }
