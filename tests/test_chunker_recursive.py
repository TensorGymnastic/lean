"""Tests for recursive chunker."""

from __future__ import annotations

from lean.chunker.markdown_ast import Section
from lean.chunker.recursive import chunk_sections


def test_chunk_short_section_is_one_chunk() -> None:
    """A short section produces exactly one chunk."""
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="A short paragraph.")]
    results = chunk_sections(sections, target_min=50, target_max=100, hard_cap=120)
    assert len(results) == 1
    assert results[0].section_path == "Ch 1"
    assert results[0].chunk_index == 0


def test_chunk_long_section_splits_into_multiple() -> None:
    """A long section splits into multiple chunks within hard_cap."""
    long_text = ". ".join(f"Sentence {i}" for i in range(200))
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content=long_text)]
    results = chunk_sections(sections, target_min=20, target_max=40, hard_cap=50)
    assert len(results) > 1
    for r in results:
        assert r.token_count <= 50
    indices = [r.chunk_index for r in results]
    assert indices == list(range(len(results)))


def test_chunk_multiple_sections_restart_index() -> None:
    """Each section restarts chunk_index at 0."""
    sections = [
        Section(path="A", level=1, heading="A", content="Para A."),
        Section(path="B", level=1, heading="B", content="Para B."),
    ]
    results = chunk_sections(sections, target_min=50, target_max=100, hard_cap=120)
    assert results[0].chunk_index == 0
    assert results[1].chunk_index == 0
    assert results[0].section_path != results[1].section_path


def test_chunk_empty_content() -> None:
    """Empty content produces no chunks."""
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="")]
    results = chunk_sections(sections, target_min=50, target_max=100, hard_cap=120)
    assert results == []


def test_chunk_preserves_section_metadata() -> None:
    """Chunk carries section_path and heading_text from its section."""
    sections = [
        Section(
            path="Chapter 3 > DMAIC > Measure",
            level=3,
            heading="Measure",
            content="Data collection.",
        )
    ]
    results = chunk_sections(sections, target_min=50, target_max=100, hard_cap=120)
    assert len(results) == 1
    assert results[0].section_path == "Chapter 3 > DMAIC > Measure"
    assert results[0].heading_text == "Measure"


def test_chunk_overlap_links_adjacent_chunks() -> None:
    """With overlap > 0, each chunk starts with tokens from the previous chunk's end."""
    long_text = ". ".join(f"Sentence {i}" for i in range(200))
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content=long_text)]
    results = chunk_sections(sections, target_min=20, target_max=40, hard_cap=50, overlap=10)
    assert len(results) > 1
    for i in range(1, len(results)):
        prev_tail = results[i - 1].content[-50:]
        curr_head = results[i].content[:50]
        assert any(word in curr_head for word in prev_tail.split()[-5:]), (
            f"chunk {i} should overlap with chunk {i - 1}'s tail"
        )


def test_chunk_overlap_zero_no_overlap() -> None:
    """With overlap=0, chunks are independent (backward compatible)."""
    long_text = ". ".join(f"Sentence {i}" for i in range(200))
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content=long_text)]
    no_overlap = chunk_sections(sections, target_min=20, target_max=40, hard_cap=50, overlap=0)
    assert len(no_overlap) > 1
