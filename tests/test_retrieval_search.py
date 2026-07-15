"""Tests for the search service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


@patch("lean.services.search.AnalyticsRepo")
@patch("lean.services.search.SearchEngine")
@patch("lean.services.search.StoreConnection")
def test_search_returns_chunks_with_scores(MockStoreConn, MockSearchEngine, MockAnalytics) -> None:
    """search() embeds the query and returns vector search results."""
    from lean.models.schemas import Chunk
    from lean.services import search as search_mod
    from lean.store.search import SearchHit

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 1024

    mock_engine = MockSearchEngine.return_value
    vector_hits = [
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
    mock_engine.vector_search.return_value = vector_hits
    mock_engine.bm25_search.return_value = []
    mock_engine.reciprocal_rank_fusion.return_value = vector_hits

    with patch("lean.services.search.get_embedder", return_value=mock_embedder):
        results = search_mod.search("What is DMAIC?", k=5)

    assert len(results) == 1
    mock_embedder.embed_query.assert_called_once_with("What is DMAIC?")
    mock_engine.vector_search.assert_called_once()
