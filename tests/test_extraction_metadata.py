"""Tests for lean.core.extraction.metadata."""

from __future__ import annotations

from pathlib import Path

from lean.core.extraction.metadata import PdfMetadata, extract_metadata


def test_extract_metadata_from_real_pdf() -> None:
    cssc = Path("data/Six-Sigma-White-Belt-Certification-Training-Manual-CSSC-2018-06b.pdf")
    if not cssc.is_file():
        return
    meta = extract_metadata(cssc)
    assert isinstance(meta, PdfMetadata)
    assert isinstance(meta.authors, list)
    assert isinstance(meta.keywords, list)
    assert isinstance(meta.toc, list)


def test_pdf_metadata_defaults() -> None:
    m = PdfMetadata()
    assert m.title is None
    assert m.authors == []
    assert getattr(m, "publisher", None) is None
    assert m.year is None
    assert m.keywords == []
    assert m.subject is None
    assert m.toc == []


def test_parse_authors_semicolon() -> None:
    from lean.core.extraction.metadata import _parse_authors

    assert _parse_authors("Smith; Jones; Brown") == ["Smith", "Jones", "Brown"]
    assert _parse_authors("Smith and Jones") == ["Smith", "Jones"]
    assert _parse_authors("Smith & Jones") == ["Smith", "Jones"]
    assert _parse_authors(None) == []
    assert _parse_authors("") == []


def test_parse_year_from_pdf_date() -> None:
    from lean.core.extraction.metadata import _parse_year

    assert _parse_year("D:20180601") == 2018
    assert _parse_year("D:2024") == 2024
    assert _parse_year("2023") == 2023
    assert _parse_year(None) is None
    assert _parse_year("") is None
