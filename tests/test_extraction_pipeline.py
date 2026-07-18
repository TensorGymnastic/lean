"""Tests for extraction pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from lean.extraction.marker_converter import MarkerNotInstalled


def _write_minimal_pdf(path: Path) -> None:
    """Write a minimal PDF."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 10, "Test")
    pdf.output(str(path))


def test_pipeline_uses_marker_when_available(tmp_path: Path) -> None:
    """When marker is installed, pipeline uses it (highest priority)."""
    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf_path)

    with (
        patch(
            "lean.extraction.marker_converter.extract_markdown",
            return_value=("# marker markdown\n", 3),
        ),
        patch(
            "lean.extraction.pipeline.extract_ocr",
            return_value=("# OCR markdown\n", 1),
        ) as ocr_mock,
    ):
        from lean.extraction.pipeline import ExtractionMethod, extract_pdf_markdown

        markdown, page_count, method = extract_pdf_markdown(
            pdf_path,
            ocr_base_url="http://fake:8000",
            hf_token="x",
            ocr_model="baidu/Unlimited-OCR",
            ocr_dpi=300,
            ocr_timeout_s=1800.0,
            ocr_max_tokens=32768,
            ocr_batch_size=20,
        )

    assert method == ExtractionMethod.MARKER
    assert "marker" in markdown
    assert page_count == 3
    ocr_mock.assert_not_called()


def test_pipeline_uses_ocr_when_marker_unavailable(tmp_path: Path) -> None:
    """When marker is not installed, pipeline uses Unlimited-OCR."""
    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf_path)

    with (
        patch(
            "lean.extraction.marker_converter.extract_markdown",
            side_effect=MarkerNotInstalled("not installed"),
        ),
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
            ocr_model="baidu/Unlimited-OCR",
            ocr_dpi=300,
            ocr_timeout_s=1800.0,
            ocr_max_tokens=32768,
            ocr_batch_size=20,
        )

    assert method == ExtractionMethod.UNLIMITED_OCR
    assert "OCR" in markdown


def test_pipeline_falls_back_to_markitdown_when_both_fail(tmp_path: Path) -> None:
    """When marker not installed and OCR server down, pipeline uses markitdown."""
    from lean.extraction.unlimited_ocr import OCRBackendUnavailable

    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf_path)

    with (
        patch(
            "lean.extraction.marker_converter.extract_markdown",
            side_effect=MarkerNotInstalled("not installed"),
        ),
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
            ocr_model="baidu/Unlimited-OCR",
            ocr_dpi=300,
            ocr_timeout_s=1800.0,
            ocr_max_tokens=32768,
            ocr_batch_size=20,
        )

    assert method == ExtractionMethod.MARKITDOWN
    assert "markitdown" in markdown
    md_mock.assert_called_once_with(pdf_path)


def test_pipeline_uses_markitdown_when_no_ocr_configured(tmp_path: Path) -> None:
    """When marker not installed and no OCR URL, pipeline goes straight to markitdown."""
    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf_path)

    with (
        patch(
            "lean.extraction.marker_converter.extract_markdown",
            side_effect=MarkerNotInstalled("not installed"),
        ),
        patch(
            "lean.extraction.pipeline.extract_markitdown",
            return_value=("# markitdown\n", 1),
        ),
    ):
        from lean.extraction.pipeline import ExtractionMethod, extract_pdf_markdown

        markdown, page_count, method = extract_pdf_markdown(
            pdf_path,
            ocr_base_url="",
            hf_token=None,
            ocr_model="baidu/Unlimited-OCR",
            ocr_dpi=300,
            ocr_timeout_s=1800.0,
            ocr_max_tokens=32768,
            ocr_batch_size=20,
        )

    assert method == ExtractionMethod.MARKITDOWN


def test_pipeline_falls_through_when_marker_errors(tmp_path: Path) -> None:
    """When marker raises a non-MarkerNotInstalled error, pipeline tries OCR next."""
    pdf_path = tmp_path / "test.pdf"
    _write_minimal_pdf(pdf_path)

    with (
        patch(
            "lean.extraction.marker_converter.extract_markdown",
            side_effect=RuntimeError("model load failed"),
        ),
        patch(
            "lean.extraction.pipeline.extract_ocr",
            return_value=("# OCR markdown\n", 1),
        ),
    ):
        from lean.extraction.pipeline import ExtractionMethod, extract_pdf_markdown

        markdown, page_count, method = extract_pdf_markdown(
            pdf_path,
            ocr_base_url="http://fake:8000",
            hf_token="x",
            ocr_model="baidu/Unlimited-OCR",
            ocr_dpi=300,
            ocr_timeout_s=1800.0,
            ocr_max_tokens=32768,
            ocr_batch_size=20,
        )

    assert method == ExtractionMethod.UNLIMITED_OCR
