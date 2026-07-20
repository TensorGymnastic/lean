"""Tests for markdown AST section parser."""

from __future__ import annotations

from lean.core.chunker.markdown_ast import build_sections


def test_build_sections_simple_markdown() -> None:
    """Single chapter with two sections."""
    md = """\
# Chapter 1

Intro paragraph.

## Section 1.1

Detail paragraph.

## Section 1.2

More detail.
"""
    sections = build_sections(md)
    assert len(sections) == 3
    assert sections[0].path == "Chapter 1"
    assert sections[0].level == 1
    assert "Intro paragraph" in sections[0].content
    assert sections[1].path == "Chapter 1 > Section 1.1"
    assert sections[2].path == "Chapter 1 > Section 1.2"


def test_build_sections_deep_nesting() -> None:
    """H1 > H2 > H3 nesting produces hierarchical paths."""
    md = """\
# A

## A.1

### A.1.1

Deep content.

### A.1.2

More deep.
"""
    sections = build_sections(md)
    paths = [s.path for s in sections]
    assert "A" in paths
    assert "A > A.1" in paths
    assert "A > A.1 > A.1.1" in paths
    assert "A > A.1 > A.1.2" in paths


def test_build_sections_front_matter() -> None:
    """Text before the first heading goes into 'Front Matter'."""
    md = """\
This is front matter with no heading.

# Chapter 1

Body.
"""
    sections = build_sections(md)
    assert sections[0].path == "Front Matter"
    assert "front matter" in sections[0].content
    assert sections[1].path == "Chapter 1"


def test_build_sections_empty_markdown() -> None:
    """Empty markdown produces empty list."""
    sections = build_sections("")
    assert sections == []
