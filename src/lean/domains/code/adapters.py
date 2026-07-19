"""Markdown / source-file extractors for the `code` domain.

The code domain doesn't need a multi-backend extraction pipeline (marker
/ OCR) — its inputs are already plain text (markdown files, source code
files). The "extraction" step is just: read the file, wrap in markdown
if needed, and yield one section per file.

Each adapter returns the full file content + a synthetic page count.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lean.core.extraction.base import ExtractionResult
from lean.core.models import ExtractionMethod

logger = logging.getLogger(__name__)


class MarkdownFileExtractor:
    """Read a single markdown / text file as if it were a one-page PDF.

    Adapter for the code domain — reads ``.md``, ``.markdown``, ``.txt``,
    ``.rst`` files directly. Multi-file corpora (e.g. a git repo) ingest
    one file per call.

    This adapter is intentionally a no-op for PDF/OCR — it just
    produces markdown for chunking.
    """

    method = ExtractionMethod.MARKITDOWN  # closest analog in the enum

    def is_configured(self) -> bool:
        return True

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Read the file and wrap as a single 'page' extraction result."""
        if not pdf_path.is_file():
            raise FileNotFoundError(f"file not found: {pdf_path}")
        content = pdf_path.read_text(encoding="utf-8", errors="replace")

        relative = pdf_path.name
        title_match = _first_heading(content)
        markdown = f"# {title_match}\n\n" if title_match else f"# {relative}\n\n"

        ext = pdf_path.suffix.lower()
        if ext and ext not in (".md", ".markdown", ".txt"):
            markdown += f"```{ext.lstrip('.')}\n{content}\n```\n"
        else:
            markdown += content

        return ExtractionResult(
            markdown=markdown,
            page_count=1,
            method=self.method,
            images={},
            block_metas=[],
        )


def _first_heading(text: str) -> str | None:
    """Pick the first markdown heading from a file, if any."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return None


__all__ = ["MarkdownFileExtractor"]
