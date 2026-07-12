"""Tests for markitdown fallback extractor."""

from __future__ import annotations

from pathlib import Path

import pytest


def _write_minimal_pdf(path: Path) -> None:
    """Write a 1-page PDF with 'Hello Lean Six Sigma' text."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 10, "Hello Lean Six Sigma")
    pdf.output(str(path))


def test_extract_markdown_from_simple_pdf(tmp_path: Path) -> None:
    """markitdown extracts text from a minimal PDF."""
    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf_path)

    from lean.extraction.markitdown_fallback import extract_markdown

    markdown, page_count = extract_markdown(pdf_path)

    assert isinstance(markdown, str)
    assert len(markdown) > 0
    assert page_count == 1


def test_extract_markdown_missing_file(tmp_path: Path) -> None:
    """Missing file raises FileNotFoundError."""
    from lean.extraction.markitdown_fallback import extract_markdown

    with pytest.raises(FileNotFoundError):
        extract_markdown(tmp_path / "nonexistent.pdf")
