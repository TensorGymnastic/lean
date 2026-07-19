"""Tests for recursive chunker."""

from __future__ import annotations

from lean.chunker.markdown_ast import Section
from lean.chunker.recursive import chunk_sections


def test_chunk_short_section_is_one_chunk() -> None:
    """A short section produces exactly one chunk."""
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="A short paragraph.")]
    results = chunk_sections(sections, target_max=100, hard_cap=120)
    assert len(results) == 1
    assert results[0].section_path == "Ch 1"
    assert results[0].chunk_index == 0


def test_chunk_long_section_splits_into_multiple() -> None:
    """A long section splits into multiple chunks within hard_cap."""
    long_text = ". ".join(f"Sentence {i}" for i in range(200))
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content=long_text)]
    results = chunk_sections(sections, target_max=40, hard_cap=50)
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
    results = chunk_sections(sections, target_max=100, hard_cap=120)
    assert results[0].chunk_index == 0
    assert results[1].chunk_index == 0
    assert results[0].section_path != results[1].section_path


def test_chunk_empty_content() -> None:
    """Empty content produces no chunks."""
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="")]
    results = chunk_sections(sections, target_max=100, hard_cap=120)
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
    results = chunk_sections(sections, target_max=100, hard_cap=120)
    assert len(results) == 1
    assert results[0].section_path == "Chapter 3 > DMAIC > Measure"
    assert results[0].heading_text == "Measure"


def test_chunk_overlap_links_adjacent_chunks() -> None:
    """With overlap > 0, each chunk starts with tokens from the previous chunk's end."""
    long_text = ". ".join(f"Sentence {i}" for i in range(200))
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content=long_text)]
    results = chunk_sections(sections, target_max=40, hard_cap=50, overlap=10)
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
    no_overlap = chunk_sections(sections, target_max=40, hard_cap=50, overlap=0)
    assert len(no_overlap) > 1


def test_chunk_single_long_paragraph_splits_by_sentences() -> None:
    """A single paragraph exceeding hard_cap is split by sentence boundaries."""
    long_para = ". ".join(f"Sentence {i}" for i in range(80))
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content=long_para)]
    results = chunk_sections(sections, target_max=20, hard_cap=25)
    assert len(results) > 1
    for r in results:
        assert r.token_count <= 25


def test_chunk_sentence_longer_than_hard_cap_splits_by_words() -> None:
    """A single sentence longer than hard_cap is hard-split by words."""
    long_sentence = " ".join(f"word{i}" for i in range(200))
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content=long_sentence)]
    results = chunk_sections(sections, target_max=20, hard_cap=25)
    assert len(results) > 1
    for r in results:
        assert r.token_count <= 25


def test_chunk_overlap_passthrough_when_chunks_lt_2() -> None:
    """When overlap > 0 but only 1 chunk is produced, no overlap is applied."""
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="A short one.")]
    results = chunk_sections(sections, target_max=100, hard_cap=120, overlap=10)
    assert len(results) == 1
    assert "A short one." in results[0].content


def test_chunk_with_custom_encoding() -> None:
    """encoding= parameter is honored; chunks still bound by hard_cap."""
    long_text = ". ".join(f"Sentence {i}" for i in range(100))
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content=long_text)]
    results_cl100k = chunk_sections(sections, target_max=20, hard_cap=25, encoding="cl100k_base")
    assert len(results_cl100k) > 1
    for r in results_cl100k:
        assert r.token_count <= 25


def test_chunk_section_path_propagated_when_overlap_active() -> None:
    """With overlap, every chunk retains its parent section_path."""
    long_text = ". ".join(f"Sentence {i}" for i in range(200))
    sections = [
        Section(
            path="Chapter 1 > DMAIC > Define",
            level=3,
            heading="Define",
            content=long_text,
        )
    ]
    results = chunk_sections(sections, target_max=40, hard_cap=50, overlap=10)
    for r in results:
        assert r.section_path == "Chapter 1 > DMAIC > Define"
        assert r.heading_text == "Define"


def test_chunk_incremental_token_accounting() -> None:
    """Multiple paragraphs accumulate tokens correctly; flush happens at hard_cap."""
    paragraphs = []
    for i in range(10):
        para = f"Paragraph {i} with enough words to contribute to chunking."
        paragraphs.append(para)
    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="\n\n".join(paragraphs))]
    results = chunk_sections(sections, target_max=30, hard_cap=40)
    assert len(results) >= 1
    for r in results:
        assert r.token_count > 0
