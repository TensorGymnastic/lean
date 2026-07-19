"""Tests for Unlimited-OCR HTTP client."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx


def _write_minimal_pdf(path: Path) -> None:
    """Write a minimal PDF (needed because the client renders pages)."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 10, "Test page")
    pdf.output(str(path))


_OCR_TUNABLES = {
    "model": "baidu/Unlimited-OCR",
    "dpi": 300,
    "timeout": 1800.0,
    "max_tokens": 32768,
    "batch_size": 20,
}


@respx.mock
def test_extract_markdown_success(tmp_path: Path) -> None:
    """Successful OCR call returns markdown."""
    respx.post("http://fake-ocr:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "# Chapter 1\n\nDMAIC is the methodology."}}]
            },
        )
    )

    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf_path)

    from lean.extraction.unlimited_ocr import extract_markdown

    markdown, page_count = extract_markdown(
        pdf_path, ocr_base_url="http://fake-ocr:8000", **_OCR_TUNABLES
    )
    assert "DMAIC" in markdown
    assert page_count == 1


@respx.mock
def test_extract_markdown_raises_on_503(tmp_path: Path) -> None:
    """503 response raises OCRBackendUnavailable."""
    respx.post("http://fake-ocr:8000/v1/chat/completions").mock(return_value=httpx.Response(503))
    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf_path)

    from lean.extraction.unlimited_ocr import OCRBackendUnavailable, extract_markdown

    with pytest.raises(OCRBackendUnavailable):
        extract_markdown(pdf_path, ocr_base_url="http://fake-ocr:8000", **_OCR_TUNABLES)


def test_extract_markdown_raises_on_connection_error(tmp_path: Path) -> None:
    """Connection error (OCR server not running) raises OCRBackendUnavailable."""
    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf_path)

    from lean.extraction.unlimited_ocr import OCRBackendUnavailable, extract_markdown

    with pytest.raises(OCRBackendUnavailable):
        extract_markdown(
            pdf_path,
            ocr_base_url="http://127.0.0.1:59999",
            **{**_OCR_TUNABLES, "timeout": 2.0},
        )


@respx.mock
def test_extract_markdown_missing_file() -> None:
    """Missing PDF raises FileNotFoundError before hitting HTTP."""
    from lean.extraction.unlimited_ocr import extract_markdown

    with pytest.raises(FileNotFoundError):
        extract_markdown(
            Path("/nonexistent.pdf"),
            ocr_base_url="http://fake-ocr:8000",
            **_OCR_TUNABLES,
        )


def test_extract_markdown_requires_ocr_tunables(tmp_path: Path) -> None:
    """No function-level defaults: caller must pass model/dpi/timeout/max_tokens/batch_size.

    The pipeline orchestrator in extraction/pipeline.py is the only
    production caller and threads from Settings. Function-level
    defaults were an unused YAGNI surface (see
    audit-architecture-2026-07-19 finding 2.2).
    """
    from lean.extraction.unlimited_ocr import extract_markdown

    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf_path)
    with pytest.raises(TypeError):
        extract_markdown(pdf_path, ocr_base_url="http://fake-ocr:8000")
