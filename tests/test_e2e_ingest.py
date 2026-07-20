"""E2E test: verifies the full pipeline works against the real corpus.

Marked @pytest.mark.e2e — requires:
- Local Supabase running with schema applied
- At least one PDF ingested

All tests in this file are read-only against the real corpus. They do NOT
ingest, delete, or mutate any persisted data. Use ``make verify-all`` plus
``-m e2e`` to run them when ``SUPABASE_DB_URL`` is set.

Run with:
    SUPABASE_DB_URL=postgresql://postgres:postgres@localhost:54322/postgres \\
        RUN_E2E=1 uv run pytest tests/test_e2e_ingest.py -v -m e2e
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

# Skip the entire module when no database is configured so bare ``pytest``
# doesn't crash with ``psycopg.OperationalError``. Integration tests use the
# same guard (see test_store_pgvector.py).
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("SUPABASE_DB_URL"),
        reason="SUPABASE_DB_URL not set — start Supabase first (make db-init)",
    ),
]


def test_corpus_has_documents() -> None:
    """Corpus has at least 1 document with chunks."""
    from lean.core.transports.mcp import corpus_stats

    stats = asyncio.run(corpus_stats())
    assert stats.document_count >= 1, "no documents ingested"
    assert stats.chunk_count > 0, "documents exist but no chunks"
    assert stats.embedding_dim == 1024


def test_canonical_queries_return_results() -> None:
    """Each canonical Lean Six Sigma query returns at least 1 result."""
    from lean.core.transports.mcp import search

    queries_file = Path(__file__).parent.parent / "scripts" / "canonical-queries.json"
    if not queries_file.exists():
        pytest.skip("canonical-queries.json not found")
    queries = json.loads(queries_file.read_text())

    for q in queries:
        results = asyncio.run(search(q["query"], k=3))
        assert len(results) >= 1, f"no results for query: {q['query']}"


def test_list_documents_works() -> None:
    """list_documents returns all ingested documents."""
    from lean.core.transports.mcp import list_documents

    docs = asyncio.run(list_documents())
    assert len(docs) >= 1
    for d in docs:
        assert d.chunk_count > 0
        assert d.extraction_method in ("unlimited_ocr", "markitdown")


def test_corpus_stats_fields_all_populated() -> None:
    """corpus_stats returns every documented field with sensible types."""
    from lean.core.transports.mcp import corpus_stats

    stats = asyncio.run(corpus_stats())
    assert stats.embedding_dim == 1024
    assert isinstance(stats.embedding_model, str) and stats.embedding_model
    assert isinstance(stats.total_tokens, int) and stats.total_tokens > 0
    assert isinstance(stats.extraction_method_breakdown, dict)
    assert isinstance(stats.duplicate_image_hashes, list)
    assert stats.last_ingested_at is not None
    for entry in stats.extraction_method_breakdown.values():
        assert entry >= 1


def test_get_chunk_returns_full_chunk_for_known_id() -> None:
    """Looking up a known chunk by ID returns its content + embedding metadata."""
    from lean.core.transports.mcp import get_chunk, search

    results = asyncio.run(search("DMAIC", k=1))
    assert len(results) >= 1
    chunk_id = str(results[0].id)

    fetched = asyncio.run(get_chunk(chunk_id))
    assert fetched is not None
    assert str(fetched.id) == chunk_id
    assert fetched.token_count > 0
    assert fetched.embedding_model is not None
    assert fetched.embedding_dim == 1024


def test_search_respects_k_limit() -> None:
    """Search with k=5 returns at most 5 hits."""
    from lean.core.transports.mcp import search

    results = asyncio.run(search("Lean Six Sigma", k=5))
    assert len(results) <= 5


def test_search_with_chunk_type_text_filter() -> None:
    """chunk_type='text' returns only text chunks (or empty list if corpus has none)."""
    from lean.core.transports.mcp import search

    results = asyncio.run(search("Lean Six Sigma", k=10, chunk_type="text"))
    for r in results:
        assert r.chunk_type == "text"


def test_search_with_chunk_type_image_filter() -> None:
    """chunk_type='image' returns only image chunks (or empty list if corpus has none)."""
    from lean.core.transports.mcp import search

    results = asyncio.run(search("Lean Six Sigma", k=10, chunk_type="image"))
    for r in results:
        assert r.chunk_type == "image"


def test_search_with_section_filter() -> None:
    """Filtering by section_substring narrows results to chunks in that section."""
    from lean.core.transports.mcp import search

    partial = asyncio.run(search("DMAIC", k=10, section="Measure"))
    for r in partial:
        assert "Measure" in r.section_path


def test_search_rejects_invalid_chunk_type() -> None:
    """Passing chunk_type='foo' raises ValueError via the service layer."""
    from lean.core.services.search import search as _search

    with pytest.raises(ValueError, match="chunk_type must be one of"):
        asyncio.run(_search("Lean Six Sigma", chunk_type="foo"))


def test_search_rejects_empty_query() -> None:
    """An empty/whitespace query raises ValueError."""
    from lean.core.services.search import search as _search

    with pytest.raises(ValueError, match="query must not be empty"):
        asyncio.run(_search("   "))


def test_search_rejects_overlong_query() -> None:
    """Query strings beyond settings.search_max_query_len raise ValueError."""
    from lean.core.services.search import search as _search

    long_query = "x" * 5000
    with pytest.raises(ValueError, match="exceeds max length"):
        asyncio.run(_search(long_query))


def test_get_document_markdown_returns_text() -> None:
    """get_document_markdown returns the joined markdown of stored chunks."""
    from lean.core.transports.mcp import get_document_markdown, list_documents

    docs = asyncio.run(list_documents())
    assert len(docs) >= 1

    md = asyncio.run(get_document_markdown(str(docs[0].id)))
    assert isinstance(md, str)
    assert len(md) > 0


def test_list_documents_chunk_counts_match_corpus_stats() -> None:
    """Sum of per-document chunk_count equals corpus_stats.chunk_count."""
    from lean.core.transports.mcp import corpus_stats, list_documents

    docs = asyncio.run(list_documents())
    stats = asyncio.run(corpus_stats())
    assert sum(d.chunk_count for d in docs) == stats.chunk_count


def test_search_results_have_populated_metadata() -> None:
    """Search hits carry section_path, heading_text, chunk_index metadata."""
    from lean.core.transports.mcp import search

    results = asyncio.run(search("DMAIC", k=3))
    for r in results:
        assert r.section_path
        assert r.token_count > 0
        assert r.chunk_index >= 0


def test_search_with_high_min_score_filters_results() -> None:
    """min_score near 1.0 returns at most a subset of unrestricted results."""
    from lean.core.transports.mcp import search

    unrestricted = asyncio.run(search("DMAIC", k=10))
    if not unrestricted:
        pytest.skip("no results to compare")
    high_threshold = max(r.score for r in unrestricted if r.score is not None)
    filtered = asyncio.run(search("DMAIC", k=10, min_score=high_threshold + 0.01))
    assert len(filtered) <= len(unrestricted)
