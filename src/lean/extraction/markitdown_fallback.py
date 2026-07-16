"""Markitdown-based PDF → markdown extraction (fallback path).

Used when the Unlimited-OCR server is unavailable. Microsoft MarkItDown
provides lightweight PDF-to-markdown via PDFMiner under the hood.
Lower fidelity than vision-based OCR but no GPU required.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF, for page count
from markitdown import MarkItDown

logger = logging.getLogger(__name__)


def extract_markdown(pdf_path: Path) -> tuple[str, int]:
    """Extract markdown from a PDF using Microsoft MarkItDown.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Tuple of (markdown_text, page_count).

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError: If extraction fails.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("extracting markdown via markitdown: %s", pdf_path)
    md = MarkItDown()
    result = md.convert(str(pdf_path))
    markdown: str = result.text_content

    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    logger.info("markitdown extracted %d chars from %d pages", len(markdown), page_count)
    return markdown, page_count
