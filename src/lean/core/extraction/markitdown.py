"""Markitdown-based PDF → markdown extraction (fallback path).

Microsoft MarkItDown via PDFMiner under the hood. Lower fidelity than
vision-based OCR but no GPU required. Always available.

Implements the universal ``Extractor`` Protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz

from lean.core.extraction.base import ExtractionResult
from lean.core.models import ExtractionMethod

logger = logging.getLogger(__name__)


class MarkitdownExtractor:
    """``Extractor`` Protocol implementation: Microsoft MarkItDown.

    Pure-Python fallback. Always available when ``markitdown[pdf]`` is
    installed (declared as a ``lean-core`` core dependency).
    """

    method = ExtractionMethod.MARKITDOWN

    def __init__(self) -> None:
        pass

    def is_configured(self) -> bool:
        try:
            import markitdown  # noqa: F401

            return True
        except ImportError:
            return False

    def extract(self, pdf_path: Path) -> ExtractionResult:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        from markitdown import MarkItDown

        logger.info("extracting markdown via markitdown: %s", pdf_path)
        md = MarkItDown()
        result = md.convert(str(pdf_path))
        markdown: str = result.text_content

        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        doc.close()

        logger.info("markitdown extracted %d chars from %d pages", len(markdown), page_count)
        return ExtractionResult(
            markdown=markdown,
            page_count=page_count,
            method=ExtractionMethod.MARKITDOWN,
            images={},
            block_metas=[],
        )


__all__ = ["MarkitdownExtractor"]
