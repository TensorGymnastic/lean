"""Orchestrator: try Unlimited-OCR, fall back to markitdown.

The pipeline tries vLLM first (best quality for tables, formulas, layout).
If vLLM is unreachable or errors, it falls back to markitdown (lighter
but no vision understanding). The chosen method is recorded on the document
so downstream code knows which path ran.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lean.extraction.markitdown_fallback import (
    extract_markdown as extract_markitdown,
)
from lean.extraction.ocr_postprocess import clean_ocr_output
from lean.extraction.unlimited_ocr import (
    OCRBackendUnavailable,
)
from lean.extraction.unlimited_ocr import (
    extract_markdown as extract_ocr,
)
from lean.models.schemas import ExtractionMethod

logger = logging.getLogger(__name__)


def extract_pdf_markdown(
    pdf_path: Path,
    *,
    vllm_base_url: str,
    hf_token: str | None,
    ocr_model: str = "baidu/Unlimited-OCR",
    ocr_dpi: int = 300,
    ocr_timeout_s: float = 600.0,
    ocr_max_tokens: int = 32768,
    ocr_batch_size: int = 20,
) -> tuple[str, int, ExtractionMethod]:
    """Extract markdown from a PDF, preferring Unlimited-OCR."""
    try:
        raw_markdown, page_count = extract_ocr(
            pdf_path,
            vllm_base_url=vllm_base_url,
            hf_token=hf_token,
            model=ocr_model,
            dpi=ocr_dpi,
            timeout=ocr_timeout_s,
            max_tokens=ocr_max_tokens,
            batch_size=ocr_batch_size,
        )
        markdown = clean_ocr_output(raw_markdown)
        return markdown, page_count, ExtractionMethod.UNLIMITED_OCR
    except OCRBackendUnavailable as exc:
        logger.warning("vLLM unavailable (%s); falling back to markitdown", exc)
        markdown, page_count = extract_markitdown(pdf_path)
        return markdown, page_count, ExtractionMethod.MARKITDOWN
