"""Unit tests for the code (markdown / source-file) domain adapter.

The audit's FYI-1 found these domains had zero unit tests. These tests
pin the behavior of ``MarkdownFileExtractor``: it reads .md / .txt /
source files and wraps them as a single-page ExtractionResult ready for
chunker.markdown_ast + recursive.split.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lean.core.models import ExtractionMethod


@pytest.fixture(autouse=True)
def _default_transport_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_HTTP_PORT", "8765")
    monkeypatch.setenv("API_PORT", "8766")


def test_markdown_extractor_method_is_text_file() -> None:
    """No extraction backend needed for plain text — uses text_file method."""
    from lean.domains.code.adapters import MarkdownFileExtractor

    assert MarkdownFileExtractor().method == ExtractionMethod.TEXT_FILE


def test_markdown_extractor_is_always_configured() -> None:
    """No external deps, is_configured() always returns True."""
    from lean.domains.code.adapters import MarkdownFileExtractor

    assert MarkdownFileExtractor().is_configured() is True


def test_markdown_extractor_reads_md_file(tmp_path: Path) -> None:
    """Reads .md directly, preserves body, no synthetic markdown wrap."""
    from lean.domains.code.adapters import MarkdownFileExtractor

    md = tmp_path / "doc.md"
    md.write_text("# Heading\n\nBody text here.\n", encoding="utf-8")
    result = MarkdownFileExtractor().extract(md)
    assert "# Heading" in result.markdown
    assert "Body text here." in result.markdown
    assert result.page_count == 1
    assert result.images == {}


def test_markdown_extractor_raises_on_missing_file(tmp_path: Path) -> None:
    """FileNotFoundError surfaces clearly so CLI can map to HTTP 404."""
    from lean.domains.code.adapters import MarkdownFileExtractor

    with pytest.raises(FileNotFoundError):
        MarkdownFileExtractor().extract(tmp_path / "nope.md")


def test_markdown_extractor_wraps_source_file_in_code_fence(tmp_path: Path) -> None:
    """Source files (not .md / .txt) get wrapped in a fenced code block."""
    from lean.domains.code.adapters import MarkdownFileExtractor

    py = tmp_path / "module.py"
    py.write_text("def foo():\n    return 42\n", encoding="utf-8")
    result = MarkdownFileExtractor().extract(py)
    assert "```py" in result.markdown
    assert "def foo():" in result.markdown


def test_markdown_extractor_picks_first_heading(tmp_path: Path) -> None:
    """The first markdown heading becomes the section title when present."""
    from lean.domains.code.adapters import MarkdownFileExtractor

    md = tmp_path / "intro.md"
    md.write_text("# Intro Title\n\nbody\n## Subheading\nbody 2\n", encoding="utf-8")
    result = MarkdownFileExtractor().extract(md)
    assert result.markdown.startswith("# Intro Title\n\n")


def test_markdown_extractor_falls_back_to_filename(tmp_path: Path) -> None:
    """When no heading is present, the filename is used as the title."""
    from lean.domains.code.adapters import MarkdownFileExtractor

    md = tmp_path / "no_heading.md"
    md.write_text("just body\n", encoding="utf-8")
    result = MarkdownFileExtractor().extract(md)
    assert result.markdown.startswith("# no_heading.md\n\n")


def test_markdown_extractor_uses_replacement_for_bad_utf8(tmp_path: Path) -> None:
    """Encoding errors are tolerated with the 'replace' error handler so
    a malformed text file does not abort ingestion."""
    from lean.domains.code.adapters import MarkdownFileExtractor

    md = tmp_path / "bad.md"
    md.write_bytes(b"# Heading\n\nBad \xff bytes\n")
    # Should not raise — read_text(errors='replace') substitutes U+FFFD.
    result = MarkdownFileExtractor().extract(md)
    assert "Heading" in result.markdown
