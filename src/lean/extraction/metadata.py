"""Extract structured metadata from PDF files via PyMuPDF (fitz).

Reads the PDF's embedded /Info dict (title, author, subject, keywords,
creation date) and the table-of-contents outline. These populate the
``documents`` table columns that were previously left empty.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


@dataclass
class PdfMetadata:
    """Structured metadata extracted from a PDF."""

    title: str | None = None
    authors: list[str] = field(default_factory=list)
    publisher: str | None = None
    year: int | None = None
    keywords: list[str] = field(default_factory=list)
    subject: str | None = None
    toc: list[str] = field(default_factory=list)


def extract_metadata(pdf_path: Path) -> PdfMetadata:
    """Extract metadata from a PDF via PyMuPDF.

    Falls back gracefully — any field that fitz can't populate stays None/empty.
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
    publisher = _guess_publisher(title, authors)

    toc_titles = [entry[1] for entry in toc_entries if len(entry) >= 3 and entry[1]]

    return PdfMetadata(
        title=title,
        authors=authors,
        publisher=publisher,
        year=year,
        keywords=keywords,
        subject=subject,
        toc=toc_titles,
    )


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    return cleaned or None


_AUTHOR_SPLIT_RE = re.compile(r"[;,]| and |&|/(?!.*\d)")


def _parse_authors(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = _AUTHOR_SPLIT_RE.split(raw)
    return [p.strip() for p in parts if p.strip()]


_KEYWORD_SPLIT_RE = re.compile(r"[;,]")


def _parse_keywords(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = _KEYWORD_SPLIT_RE.split(raw)
    return [p.strip() for p in parts if p.strip()]


_DATE_RE = re.compile(r"D:(\d{4})")


def _parse_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    match = _DATE_RE.search(date_str)
    if match:
        return int(match.group(1))
    match = re.search(r"\d{4}", date_str)
    return int(match.group()) if match else None


_PUBLISHER_KEYWORDS = ("Wiley", "Springer", "Wiley ", "Springer ", "Press", "Publishing")


def _guess_publisher(title: str | None, authors: list[str]) -> str | None:
    text = f"{title or ''} {' '.join(authors)}"
    for kw in _PUBLISHER_KEYWORDS:
        if kw.strip() in text:
            return kw.strip()
    return None
