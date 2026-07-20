"""Universal PDF metadata extractor — fits in lean, free of domain heuristics.

Returns the document title / authors / year from the PDF's embedded
``/Info`` dict (via PyMuPDF). Domain-specific enrichment (e.g.
lean-lss's publisher heuristic) should subclass or compose this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

_DATE_RE = re.compile(r"D:(\d{4})")
_AUTHOR_SPLIT_RE = re.compile(r"[;,]| and |&|/(?!.*\d)")


@dataclass
class PdfMetadata:
    """Structured metadata extracted from a PDF."""

    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    subject: str | None = None
    keywords: list[str] = field(default_factory=list)
    toc: list[str] = field(default_factory=list)


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    return cleaned or None


def _parse_authors(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = _AUTHOR_SPLIT_RE.split(raw)
    return [p.strip() for p in parts if p.strip()]


def _parse_keywords(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[;,]", raw)
    return [p.strip() for p in parts if p.strip()]


def _parse_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    match = _DATE_RE.search(date_str)
    if match:
        return int(match.group(1))
    match = re.search(r"\d{4}", date_str)
    return int(match.group()) if match else None


def extract_metadata(pdf_path: Path) -> PdfMetadata:
    """Extract metadata from a PDF via PyMuPDF.

    Falls back gracefully — any field that fitz can't populate stays
    None/empty. Returns the universal subset of fields. Domains can
    wrap this and add their own enrichment.
    """
    doc = fitz.open(str(pdf_path))
    try:
        info = doc.metadata or {}
        toc_entries = doc.get_toc() or []
    finally:
        doc.close()

    title = _clean(info.get("title"))
    authors = _parse_authors(info.get("author"))
    subject = _clean(info.get("subject"))
    keywords = _parse_keywords(info.get("keywords"))
    year = _parse_year(info.get("creationDate") or info.get("modDate"))
    toc_titles = [entry[1] for entry in toc_entries if len(entry) >= 3 and entry[1]]

    return PdfMetadata(
        title=title,
        authors=authors,
        year=year,
        subject=subject,
        keywords=keywords,
        toc=toc_titles,
    )


__all__ = ["PdfMetadata", "extract_metadata"]
