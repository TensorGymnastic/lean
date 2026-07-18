"""Orchestrator: try marker → Unlimited-OCR → markitdown.

Priority:
    1. Marker (datalab-to/marker) — best quality + speed, no server needed.
       Requires ``uv sync --extra marker``. Falls through if not installed.
    2. Unlimited-OCR (remote GPU server) — if ``ocr_base_url`` is set.
    3. markitdown — always available, last resort.

The chosen method is recorded on the document so downstream code knows
which path ran.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lean.extraction.marker_converter import MarkerNotInstalled
from lean.extraction.markitdown_fallback import (
    extract_markdown as extract_markitdown,
)
from lean.extraction.ocr_postprocess import clean_ocr_output
from lean.extraction.unlimited_ocr import OCRBackendUnavailable
from lean.extraction.unlimited_ocr import (
    extract_markdown as extract_ocr,
)
from lean.models.schemas import ExtractionMethod

logger = logging.getLogger(__name__)


def extract_pdf_markdown(
    pdf_path: Path,
    *,
    ocr_base_url: str,
    hf_token: str | None,
    ocr_model: str,
    ocr_dpi: int,
    ocr_timeout_s: float,
    ocr_max_tokens: int,
    ocr_batch_size: int,
    marker_force_ocr: bool = False,
) -> tuple[str, int, ExtractionMethod]:
    """Extract markdown from a PDF, preferring marker then OCR then markitdown."""
    # 1. Try marker (best quality + speed if installed)
    try:
        from lean.extraction.marker_converter import extract_markdown as extract_marker

        markdown, page_count = extract_marker(pdf_path, force_ocr=marker_force_ocr)
        return markdown, page_count, ExtractionMethod.MARKER
    except MarkerNotInstalled:
        pass  # marker not installed — try next backend
    except Exception as exc:
        logger.warning("marker extraction failed (%s); trying next backend", exc)

    # 2. Try Unlimited-OCR (if remote server configured)
    if ocr_base_url:
        try:
            raw_markdown, page_count = extract_ocr(
                pdf_path,
                ocr_base_url=ocr_base_url,
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
            logger.warning("OCR server unavailable (%s); falling back to markitdown", exc)

    # 3. Last resort: markitdown
    markdown, page_count = extract_markitdown(pdf_path)
    return markdown, page_count, ExtractionMethod.MARKITDOWN
