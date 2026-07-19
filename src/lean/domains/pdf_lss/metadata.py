"""PDF metadata extractor for the pdf_lss domain.

Pure-Python heuristic — reads the PDF /Info dict for title, authors,
year, publisher, keywords. Same shape as ``lean.core.extraction.metadata``
with one addition: a ``publisher`` field inferred from the title text.

Domains that need a different metadata shape should override this
module — declare a new dotted path in the YAML.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

_DATE_RE = re.compile(r"D:(\d{4})")
_AUTHOR_SPLIT_RE = re.compile(r"[;,]| and |&|/(?!.*\d)")
_KEYWORD_SPLIT_RE = re.compile(r"[;,]")
_PUBLISHER_KEYWORDS = ("Wiley", "Springer", "Press", "Publishing")


@dataclass
class PdfMetadata:
    """Document metadata extracted from a PDF."""

    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    publisher: str | None = None
    keywords: list[str] = field(default_factory=list)
    subject: str | None = None
    toc: list[str] = field(default_factory=list)


def _clean(value: object) -> str | None:
    if not value:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _parse_authors(raw: object) -> list[str]:
    if not raw:
        return []
    parts = _AUTHOR_SPLIT_RE.split(str(raw))
    return [p.strip() for p in parts if p.strip()]


def _parse_keywords(raw: object) -> list[str]:
    if not raw:
        return []
    parts = _KEYWORD_SPLIT_RE.split(str(raw))
    return [p.strip() for p in parts if p.strip()]


def _parse_year(date_str: object) -> int | None:
    if not date_str:
        return None
    match = _DATE_RE.search(str(date_str))
    if match:
        return int(match.group(1))
    match = re.search(r"\d{4}", str(date_str))
    return int(match.group()) if match else None


def _guess_publisher(title: object, authors: list[str]) -> str | None:
    text = f"{title or ''} {' '.join(authors)}"
    for kw in _PUBLISHER_KEYWORDS:
        if kw.strip() in text:
            return kw.strip()
    return None


def extract(pdf_path: Path) -> dict[str, object]:
    """Extract metadata from a PDF. Returns a dict (not a dataclass) so
    lean-core's ingestion service can pass it through to the document row.
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

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "publisher": publisher,
        "keywords": keywords,
        "subject": subject,
        "toc": toc_titles,
    }


__all__ = ["extract", "PdfMetadata"]
