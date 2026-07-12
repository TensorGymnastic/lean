"""Orchestrator: try Unlimited-OCR, fall back to markitdown.

The pipeline tries vLLM first (best quality for tables, formulas, layout).
If vLLM is unreachable or errors, it falls back to markitdown (lighter
but no vision understanding). The chosen method is recorded on the document
so downstream code knows which path ran.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path

from lean.extraction.markitdown_fallback import (
    extract_markdown as extract_markitdown,
)
from lean.extraction.unlimited_ocr import (
    OCRBackendUnavailable,
)
from lean.extraction.unlimited_ocr import (
    extract_markdown as extract_ocr,
)

logger = logging.getLogger(__name__)


class ExtractionMethod(StrEnum):
    """Which extraction path produced this document's markdown."""

    UNLIMITED_OCR = "unlimited_ocr"
    MARKITDOWN = "markitdown"


def extract_pdf_markdown(
    pdf_path: Path,
    *,
    vllm_base_url: str,
    hf_token: str | None,
) -> tuple[str, int, ExtractionMethod]:
    """Extract markdown from a PDF, preferring Unlimited-OCR.

    Args:
        pdf_path: Path to the PDF file.
        vllm_base_url: Base URL of the vLLM server.
        hf_token: Optional HuggingFace token.

    Returns:
        Tuple of (markdown, page_count, method).
        ``method`` indicates which extractor succeeded.

    The function never raises OCRBackendUnavailable — if vLLM fails,
    it logs a warning and falls back to markitdown.
    """
    try:
        markdown, page_count = extract_ocr(
            pdf_path,
            vllm_base_url=vllm_base_url,
            hf_token=hf_token,
        )
        return markdown, page_count, ExtractionMethod.UNLIMITED_OCR
    except OCRBackendUnavailable as exc:
        logger.warning("vLLM unavailable (%s); falling back to markitdown", exc)
        markdown, page_count = extract_markitdown(pdf_path)
        return markdown, page_count, ExtractionMethod.MARKITDOWN
