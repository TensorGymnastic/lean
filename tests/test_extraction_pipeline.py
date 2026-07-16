"""Tests for extraction pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def _write_minimal_pdf(path: Path) -> None:
    """Write a minimal PDF."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 10, "Test")
    pdf.output(str(path))


def test_pipeline_uses_ocr_when_available(tmp_path: Path) -> None:
    """When OCR server is available, pipeline uses Unlimited-OCR."""
    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf_path)

    with (
        patch(
            "lean.extraction.pipeline.extract_ocr",
            return_value=("# OCR markdown\n", 1),
        ),
        patch(
            "lean.extraction.pipeline.extract_markitdown",
            return_value=("# markitdown\n", 1),
        ),
    ):
        from lean.extraction.pipeline import ExtractionMethod, extract_pdf_markdown

        markdown, page_count, method = extract_pdf_markdown(
            pdf_path,
            ocr_base_url="http://fake:8000",
            hf_token="x",
        )

    assert method == ExtractionMethod.UNLIMITED_OCR
    assert "OCR" in markdown


def test_pipeline_falls_back_when_ocr_unavailable(tmp_path: Path) -> None:
    """When OCR server is down, pipeline falls back to markitdown."""
    from lean.extraction.unlimited_ocr import OCRBackendUnavailable

    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf_path)

    with (
        patch(
            "lean.extraction.pipeline.extract_ocr",
            side_effect=OCRBackendUnavailable("OCR server down"),
        ),
        patch(
            "lean.extraction.pipeline.extract_markitdown",
            return_value=("# markitdown fallback\n", 1),
        ) as md_mock,
    ):
        from lean.extraction.pipeline import ExtractionMethod, extract_pdf_markdown

        markdown, page_count, method = extract_pdf_markdown(
            pdf_path,
            ocr_base_url="http://fake:8000",
            hf_token="x",
        )

    assert method == ExtractionMethod.MARKITDOWN
    assert "markitdown" in markdown
    md_mock.assert_called_once_with(pdf_path)
