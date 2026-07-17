"""E2E test: verifies the full pipeline works against the real corpus.

Marked @pytest.mark.e2e — requires:
- Local Supabase running with schema applied
- At least one PDF ingested

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
    from lean.mcp_server.tools import corpus_stats

    stats = asyncio.run(corpus_stats())
    assert stats.document_count >= 1, "no documents ingested"
    assert stats.chunk_count > 0, "documents exist but no chunks"
    assert stats.embedding_dim == 1024


def test_canonical_queries_return_results() -> None:
    """Each canonical Lean Six Sigma query returns at least 1 result."""
    from lean.mcp_server.tools import search

    queries_file = Path(__file__).parent.parent / "scripts" / "canonical-queries.json"
    if not queries_file.exists():
        pytest.skip("canonical-queries.json not found")
    queries = json.loads(queries_file.read_text())

    for q in queries:
        results = asyncio.run(search(q["query"], k=3))
        assert len(results) >= 1, f"no results for query: {q['query']}"


def test_list_documents_works() -> None:
    """list_documents returns all ingested documents."""
    from lean.mcp_server.tools import list_documents

    docs = asyncio.run(list_documents())
    assert len(docs) >= 1
    for d in docs:
        assert d.chunk_count > 0
        assert d.extraction_method in ("unlimited_ocr", "markitdown")
