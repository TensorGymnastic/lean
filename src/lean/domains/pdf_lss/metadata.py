"""PDF metadata extractor for the pdf_lss domain.

Delegates to ``lean.core.extraction.metadata.extract_metadata`` for the
universal fields, then adds the domain-specific ``publisher`` heuristic.
"""

from __future__ import annotations

from pathlib import Path

from lean.core.extraction.metadata import extract_metadata as _core_extract

_PUBLISHER_KEYWORDS = ("Wiley", "Springer", "Press", "Publishing")


def _guess_publisher(title: str | None, authors: list[str]) -> str | None:
    text = f"{title or ''} {' '.join(authors)}"
    for kw in _PUBLISHER_KEYWORDS:
        if kw in text:
            return kw
    return None


def extract(pdf_path: Path) -> dict[str, object]:
    """Extract metadata from a PDF. Returns a dict for the ingestion service."""
    meta = _core_extract(pdf_path)
    publisher = _guess_publisher(meta.title, meta.authors)
    return {
        "title": meta.title,
        "authors": meta.authors,
        "year": meta.year,
        "publisher": publisher,
        "keywords": meta.keywords,
        "subject": meta.subject,
        "toc": meta.toc,
    }


__all__ = ["extract"]
