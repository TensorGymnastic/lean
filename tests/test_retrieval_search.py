"""Tests for retrieval search module."""

from __future__ import annotations

from unittest.mock import patch


@patch("lean.retrieval.search.PgVectorStore")
@patch("lean.retrieval.search.LiquidLMFEmbedder")
def test_search_returns_chunks_with_scores(MockEmbedder, MockStore) -> None:
    """search() embeds the query and returns pgvector results."""
    from lean.models.schemas import Chunk
    from lean.retrieval import search as search_mod
    from lean.store.pgvector import SearchHit

    # Reset module-level singletons
    search_mod._embedder = None  # noqa: SLF001
    search_mod._store = None  # noqa: SLF001

    mock_embedder = MockEmbedder.return_value
    mock_embedder.embed_query.return_value = [0.1] * 1024

    mock_store = MockStore.from_env.return_value
    mock_store.search.return_value = [
        SearchHit(
            chunk=Chunk(
                id="c1",
                document_id="d1",
                chunk_index=0,
                section_path="Ch 1",
                heading_text="DMAIC",
                page_start=1,
                page_end=2,
                token_count=400,
                content="DMAIC content",
                score=0.92,
            ),
            score=0.92,
        )
    ]

    results = search_mod.search("What is DMAIC?", k=5)

    assert len(results) == 1
    assert results[0].score == 0.92
    mock_embedder.embed_query.assert_called_once_with("What is DMAIC?")
    mock_store.search.assert_called_once()

    # Cleanup
    search_mod._embedder = None  # noqa: SLF001
    search_mod._store = None  # noqa: SLF001
