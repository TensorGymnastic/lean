"""Unit tests for ``_find_best_block_match`` in services/ingestion.py.

This is the Jaccard-like word-overlap matcher that drives ``bbox`` and
``page_start``/``page_end`` enrichment. It has no DB or marker dependency —
pure string matching — so it can be unit-tested directly.
"""

from __future__ import annotations

from lean.extraction.marker_converter import BlockMeta
from lean.services.ingestion import _find_best_block_match


def test_exact_text_match_returns_block_page_and_bbox():
    """Identical content+block text yields score=1.0 → page+bbox returned."""
    content = "the quick brown fox jumps over the lazy dog"
    blocks = [
        BlockMeta(
            page=7,
            bbox=[10.0, 20.0, 30.0, 40.0],
            text=content,
        )
    ]

    page, bbox = _find_best_block_match(content, blocks)

    assert page == 7
    assert bbox == {"x0": 10.0, "y0": 20.0, "x1": 30.0, "y1": 40.0}


def test_no_overlap_returns_none_tuple():
    """Zero shared words → score=0.0 < 0.15 → (None, None)."""
    content = "alpha beta gamma"
    blocks = [BlockMeta(page=1, bbox=[1.0, 2.0, 3.0, 4.0], text="delta epsilon zeta")]

    page, bbox = _find_best_block_match(content, blocks)

    assert page is None
    assert bbox is None


def test_below_threshold_overlap_returns_none_tuple():
    """Overlap below the 0.15 minimum is rejected (would otherwise produce noisy matches)."""
    # 7 content words, 7 block words, 1 shared → score = 1/7 ≈ 0.143 < 0.15
    content = "a b c d e f g"
    blocks = [BlockMeta(page=2, bbox=[1.0, 2.0, 3.0, 4.0], text="a z y x w v u")]

    page, bbox = _find_best_block_match(content, blocks)

    assert page is None
    assert bbox is None


def test_bbox_with_wrong_arity_returns_page_without_bbox():
    """bbox list with len != 4 returns (page, None) — page is still trustworthy."""
    content = "the quick brown fox"
    blocks = [
        BlockMeta(
            page=5,
            bbox=[1.0, 2.0, 3.0],  # len=3, malformed
            text=content,
        )
    ]

    page, bbox = _find_best_block_match(content, blocks)

    assert page == 5
    assert bbox is None


def test_empty_content_returns_none_tuple():
    """Empty or whitespace-only content short-circuits before scanning blocks."""
    blocks = [BlockMeta(page=1, bbox=[1.0, 2.0, 3.0, 4.0], text="anything")]

    page, bbox = _find_best_block_match("", blocks)

    assert page is None
    assert bbox is None


def test_empty_block_text_is_skipped_not_crashing():
    """A block whose text normalizes to empty is skipped, not raised on."""
    content = "the quick brown fox jumps over the lazy dog"
    blocks = [
        BlockMeta(page=99, bbox=[1.0, 2.0, 3.0, 4.0], text=""),  # empty, skipped
        BlockMeta(page=3, bbox=[5.0, 6.0, 7.0, 8.0], text=content),  # exact match
    ]

    page, bbox = _find_best_block_match(content, blocks)

    # The empty block at page 99 was skipped; the exact-match block at page 3 wins.
    assert page == 3
    assert bbox == {"x0": 5.0, "y0": 6.0, "x1": 7.0, "y1": 8.0}


def test_empty_blocks_list_returns_none_tuple():
    """No blocks to match against → (None, None), no crash."""
    page, bbox = _find_best_block_match("anything", [])

    assert page is None
    assert bbox is None


def test_none_bbox_on_best_block_returns_page_without_bbox():
    """Best block has bbox=None → returned bbox is None, page is still set."""
    content = "the quick brown fox jumps over the lazy dog"
    blocks = [BlockMeta(page=4, bbox=None, text=content)]

    page, bbox = _find_best_block_match(content, blocks)

    assert page == 4
    assert bbox is None


def test_highest_scoring_block_wins_when_multiple_match():
    """When multiple blocks partially overlap, the best score wins."""
    content = "alpha beta gamma delta"
    blocks = [
        BlockMeta(page=1, bbox=[1.0, 1.0, 1.0, 1.0], text="alpha zzz"),  # 1/3 ≈ 0.33
        BlockMeta(page=2, bbox=[2.0, 2.0, 2.0, 2.0], text="alpha beta yyy"),  # 2/3 ≈ 0.67
        BlockMeta(page=3, bbox=[3.0, 3.0, 3.0, 3.0], text="zzz yyy www"),  # 0
    ]

    page, bbox = _find_best_block_match(content, blocks)

    assert page == 2
    assert bbox == {"x0": 2.0, "y0": 2.0, "x1": 2.0, "y1": 2.0}
