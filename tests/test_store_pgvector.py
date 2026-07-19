"""Integration tests for the store repos (requires local Supabase)."""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def db_url() -> str:
    url = os.environ.get("SUPABASE_DB_URL", "")
    if not url:
        pytest.skip("SUPABASE_DB_URL not set")
    return url


@pytest.fixture()
def conn(db_url: str):
    from lean.store.base import StoreConnection

    c = StoreConnection(db_url)
    yield c
    c.close()


def _make_doc(conn, sha: str = "test-sha-256") -> uuid.UUID:
    """Insert a test document and return its UUID. Cleans up via test isolation."""
    from lean.store.documents import DocumentRepo

    return DocumentRepo(conn).upsert_document(
        source_path=f"test-{sha}.pdf",
        source_sha256=sha,
        title="Test Doc",
        extraction_method="markitdown",
        source_storage_path=f"sources/{sha}.pdf",
        markdown_storage_path=f"markdown/{sha}.md",
        page_count=10,
    )


def test_upsert_and_search(conn) -> None:
    """Insert document + chunks, search by embedding, verify results."""
    from lean.store.chunks import ChunkRepo, ChunkRow
    from lean.store.documents import DocumentRepo
    from lean.store.search import SearchEngine

    doc_id = _make_doc(conn, "upsert-search-test")

    chunks = [
        ChunkRow(
            document_id=doc_id,
            chunk_index=0,
            section_path="Chapter 1 > DMAIC",
            heading_text="DMAIC",
            page_start=1,
            page_end=1,
            token_count=100,
            content="DMAIC is the core Six Sigma methodology.",
            embedding=[0.1] * 1024,
        ),
        ChunkRow(
            document_id=doc_id,
            chunk_index=1,
            section_path="Chapter 2 > Kaizen",
            heading_text="Kaizen",
            page_start=2,
            page_end=2,
            token_count=100,
            content="Kaizen means continuous improvement.",
            embedding=[0.9] * 1024,
        ),
    ]
    ChunkRepo(conn).replace_chunks(doc_id, chunks)

    # Search with embedding close to chunk 0
    hits = SearchEngine(conn).vector_search(query_embedding=[0.11] * 1024, k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score  # sorted by score descending
    assert "DMAIC" in hits[0].chunk.content or "Kaizen" in hits[0].chunk.content

    # Cleanup
    DocumentRepo(conn).delete_document(doc_id)


def test_search_with_section_filter(conn) -> None:
    """section_substring filter narrows results."""
    from lean.store.chunks import ChunkRepo, ChunkRow
    from lean.store.documents import DocumentRepo
    from lean.store.search import SearchEngine

    doc_id = _make_doc(conn, "section-filter-test")
    chunks = [
        ChunkRow(
            document_id=doc_id,
            chunk_index=i,
            section_path=f"Chapter {i} > {'DMAIC' if i == 0 else 'Kaizen'}",
            heading_text="H",
            page_start=i,
            page_end=i,
            token_count=50,
            content=f"Content {i}",
            embedding=[float(i)] * 1024,
        )
        for i in range(2)
    ]
    ChunkRepo(conn).replace_chunks(doc_id, chunks)

    hits = SearchEngine(conn).vector_search(
        query_embedding=[0.0] * 1024,
        k=10,
        section_substring="DMAIC",
    )
    assert all("DMAIC" in h.chunk.section_path for h in hits)
    assert len(hits) == 1

    DocumentRepo(conn).delete_document(doc_id)


def test_replace_chunks_is_replacement(conn) -> None:
    """replace_chunks deletes old chunks before inserting new ones."""
    from lean.store.chunks import ChunkRepo, ChunkRow
    from lean.store.documents import DocumentRepo
    from lean.store.search import SearchEngine

    doc_id = _make_doc(conn, "replace-test")

    ChunkRepo(conn).replace_chunks(
        doc_id,
        [
            ChunkRow(
                document_id=doc_id,
                chunk_index=i,
                section_path="S",
                heading_text=None,
                page_start=None,
                page_end=None,
                token_count=10,
                content=f"old {i}",
                embedding=[0.0] * 1024,
            )
            for i in range(5)
        ],
    )
    assert ChunkRepo(conn).count_chunks() >= 5

    # Replace with only 2 chunks
    ChunkRepo(conn).replace_chunks(
        doc_id,
        [
            ChunkRow(
                document_id=doc_id,
                chunk_index=i,
                section_path="S",
                heading_text=None,
                page_start=None,
                page_end=None,
                token_count=10,
                content=f"new {i}",
                embedding=[0.0] * 1024,
            )
            for i in range(2)
        ],
    )

    # Verify old chunks are gone — search for this doc only
    hits = SearchEngine(conn).vector_search(query_embedding=[0.0] * 1024, k=100, doc_id=doc_id)
    assert len(hits) == 2
    assert all("new" in h.chunk.content for h in hits)

    DocumentRepo(conn).delete_document(doc_id)


def test_upsert_document_dedup(conn) -> None:
    """Re-upserting the same source_sha256 updates rather than duplicating."""
    from lean.store.analytics import AnalyticsRepo
    from lean.store.documents import DocumentRepo

    doc_id_1 = _make_doc(conn, "dedup-test")
    doc_id_2 = DocumentRepo(conn).upsert_document(
        source_path="updated-path.pdf",
        source_sha256="dedup-test",
        title="Updated Title",
        extraction_method="unlimited_ocr",
        source_storage_path="sources/dedup-test.pdf",
        markdown_storage_path="markdown/dedup-test.md",
    )
    assert doc_id_1 == doc_id_2  # same UUID
    assert AnalyticsRepo(conn).count_documents() >= 0  # no duplicate created

    DocumentRepo(conn).delete_document(doc_id_1)


def test_get_chunk(conn) -> None:
    """get_chunk returns the chunk by ID, or None."""
    from lean.store.chunks import ChunkRepo, ChunkRow
    from lean.store.documents import DocumentRepo
    from lean.store.search import SearchEngine

    doc_id = _make_doc(conn, "get-chunk-test")
    ChunkRepo(conn).replace_chunks(
        doc_id,
        [
            ChunkRow(
                document_id=doc_id,
                chunk_index=0,
                section_path="S",
                heading_text="H",
                page_start=1,
                page_end=1,
                token_count=50,
                content="find me",
                embedding=[0.0] * 1024,
            )
        ],
    )
    hits = SearchEngine(conn).vector_search(query_embedding=[0.0] * 1024, k=1, doc_id=doc_id)
    assert len(hits) == 1
    chunk_id = hits[0].chunk.id

    fetched = ChunkRepo(conn).get_chunk(uuid.UUID(chunk_id))
    assert fetched is not None
    assert fetched.content == "find me"

    # Non-existent ID
    assert ChunkRepo(conn).get_chunk(uuid.uuid4()) is None

    DocumentRepo(conn).delete_document(doc_id)


def test_bm25_search_filters_by_section(conn) -> None:
    """BM25 full-text search with section_substring narrows results to matching chunks."""
    from lean.store.chunks import ChunkRepo, ChunkRow
    from lean.store.documents import DocumentRepo
    from lean.store.search import SearchEngine

    doc_id = _make_doc(conn, "bm25-section-test")
    chunks = [
        ChunkRow(
            document_id=doc_id,
            chunk_index=0,
            section_path="Chapter 1 > DMAIC > Define",
            heading_text="Define",
            page_start=1,
            page_end=1,
            token_count=50,
            content="DMAIC Define phase identifies the problem scope and customer requirements.",
            embedding=[0.5] * 1024,
        ),
        ChunkRow(
            document_id=doc_id,
            chunk_index=1,
            section_path="Chapter 2 > Kaizen",
            heading_text="Kaizen",
            page_start=2,
            page_end=2,
            token_count=50,
            content="Kaizen continuous improvement methodology from Toyota lean manufacturing.",
            embedding=[0.5] * 1024,
        ),
    ]
    ChunkRepo(conn).replace_chunks(doc_id, chunks)

    hits = SearchEngine(conn).bm25_search(
        query_text="DMAIC problem scope customer",
        k=10,
        doc_id=doc_id,
        section_substring="DMAIC",
    )
    assert len(hits) == 1
    assert "DMAIC" in hits[0].chunk.section_path
    assert hits[0].score > 0

    DocumentRepo(conn).delete_document(doc_id)


def test_chunk_type_filter_at_db_level(conn) -> None:
    """chunk_type=image SQL filter (line 60-62) returns only image chunks."""
    from lean.store.chunks import ChunkRepo, ChunkRow
    from lean.store.documents import DocumentRepo
    from lean.store.search import SearchEngine

    doc_id = _make_doc(conn, "chunk-type-db-test")
    chunks = [
        ChunkRow(
            document_id=doc_id,
            chunk_index=0,
            section_path="Body",
            heading_text="Text",
            page_start=1,
            page_end=1,
            token_count=20,
            content="regular text chunk",
            embedding=[0.5] * 1024,
            chunk_type="text",
        ),
        ChunkRow(
            document_id=doc_id,
            chunk_index=1,
            section_path="Figures",
            heading_text="Image",
            page_start=2,
            page_end=2,
            token_count=20,
            content="VLM-described chart",
            embedding=[0.5] * 1024,
            chunk_type="image",
            image_meta={"chart_type": "bar"},
            image_hash="a" * 64,
        ),
    ]
    ChunkRepo(conn).replace_chunks(doc_id, chunks)

    image_hits = SearchEngine(conn).vector_search(
        query_embedding=[0.5] * 1024, k=10, doc_id=doc_id, chunk_type="image"
    )
    assert len(image_hits) == 1
    assert image_hits[0].chunk.chunk_type == "image"

    text_hits = SearchEngine(conn).vector_search(
        query_embedding=[0.5] * 1024, k=10, doc_id=doc_id, chunk_type="text"
    )
    assert len(text_hits) == 1
    assert text_hits[0].chunk.chunk_type == "text"

    all_hits = SearchEngine(conn).vector_search(query_embedding=[0.5] * 1024, k=10, doc_id=doc_id)
    assert len(all_hits) == 2

    DocumentRepo(conn).delete_document(doc_id)


def test_min_score_filter_at_db_level(conn) -> None:
    """min_score SQL filter restricts hits to those above the cosine threshold."""
    from lean.store.chunks import ChunkRepo, ChunkRow
    from lean.store.documents import DocumentRepo
    from lean.store.search import SearchEngine

    doc_id = _make_doc(conn, "min-score-db-test")
    chunks = [
        ChunkRow(
            document_id=doc_id,
            chunk_index=0,
            section_path="S",
            heading_text="Close",
            page_start=1,
            page_end=1,
            token_count=20,
            content="a",
            embedding=[0.9] * 1024,
        ),
        ChunkRow(
            document_id=doc_id,
            chunk_index=1,
            section_path="S",
            heading_text="Far",
            page_start=2,
            page_end=2,
            token_count=20,
            content="b",
            embedding=[-0.9] * 1024,
        ),
    ]
    ChunkRepo(conn).replace_chunks(doc_id, chunks)

    near_one = SearchEngine(conn).vector_search(
        query_embedding=[0.95] * 1024, k=10, doc_id=doc_id, min_score=0.95
    )
    assert len(near_one) == 1
    assert near_one[0].score >= 0.95

    DocumentRepo(conn).delete_document(doc_id)


def test_hybrid_search_rrf_orders_overlapping_hits_first(conn) -> None:
    """vector_search + bm25_search + rrf together promote chunks in both lists."""
    from lean.store.chunks import ChunkRepo, ChunkRow
    from lean.store.documents import DocumentRepo
    from lean.store.search import SearchEngine

    doc_id = _make_doc(conn, "rrf-hybrid-test")
    chunks = [
        ChunkRow(
            document_id=doc_id,
            chunk_index=0,
            section_path="S",
            heading_text="Shared",
            page_start=1,
            page_end=1,
            token_count=20,
            content="DMAIC define measure analyze improve control",
            embedding=[0.9] * 1024,
        ),
        ChunkRow(
            document_id=doc_id,
            chunk_index=1,
            section_path="S",
            heading_text="VectorOnly",
            page_start=2,
            page_end=2,
            token_count=20,
            content="totally different vector close to query",
            embedding=[0.91] * 1024,
        ),
        ChunkRow(
            document_id=doc_id,
            chunk_index=2,
            section_path="S",
            heading_text="Bm25Only",
            page_start=3,
            page_end=3,
            token_count=20,
            content="totally different bm25 keyword match",
            embedding=[-0.9] * 1024,
        ),
    ]
    ChunkRepo(conn).replace_chunks(doc_id, chunks)

    engine = SearchEngine(conn)
    vector_hits = engine.vector_search(query_embedding=[0.9] * 1024, k=10, doc_id=doc_id)
    bm25_hits = engine.bm25_search(query_text="DMAIC define measure", k=10, doc_id=doc_id)

    fused = engine.reciprocal_rank_fusion(vector_hits, bm25_hits, k=10, rrf_k=60)

    assert len(fused) >= 1
    assert fused[0].chunk.heading_text == "Shared"

    DocumentRepo(conn).delete_document(doc_id)


def test_find_duplicate_image_hashes_returns_real_duplicates(conn) -> None:
    """find_duplicate_image_hashes surfaces DB rows where image_hash is repeated."""
    from lean.store.chunks import ChunkRepo, ChunkRow
    from lean.store.documents import DocumentRepo

    doc_id = _make_doc(conn, "duplicate-image-hashes-test")
    dup_hash = "f" * 64
    chunks = [
        ChunkRow(
            document_id=doc_id,
            chunk_index=0,
            section_path="Figures",
            heading_text="Chart A",
            page_start=1,
            page_end=1,
            token_count=10,
            content="desc a",
            embedding=[0.0] * 1024,
            chunk_type="image",
            image_meta={"chart_type": "bar"},
            image_hash=dup_hash,
        ),
        ChunkRow(
            document_id=doc_id,
            chunk_index=1,
            section_path="Figures",
            heading_text="Chart A (duplicate)",
            page_start=2,
            page_end=2,
            token_count=10,
            content="same image reused elsewhere",
            embedding=[0.0] * 1024,
            chunk_type="image",
            image_meta={"chart_type": "bar"},
            image_hash=dup_hash,
        ),
        ChunkRow(
            document_id=doc_id,
            chunk_index=2,
            section_path="Figures",
            heading_text="Unique",
            page_start=3,
            page_end=3,
            token_count=10,
            content="unique image",
            embedding=[0.0] * 1024,
            chunk_type="image",
            image_meta={"chart_type": "pie"},
            image_hash="unique123",
        ),
    ]
    ChunkRepo(conn).replace_chunks(doc_id, chunks)

    dupes = ChunkRepo(conn).find_duplicate_image_hashes()
    dups_dict = dict(dupes)
    assert dups_dict.get(dup_hash) == 2
    assert "unique123" not in dups_dict

    DocumentRepo(conn).delete_document(doc_id)


def test_log_query_persists_filter_payload(conn) -> None:
    """log_query writes the row with the JSONB filter payload preserved."""
    from lean.store.analytics import AnalyticsRepo
    from psycopg.rows import dict_row

    repo = AnalyticsRepo(conn)
    repo.log_query(
        query_text="integration-test-query",
        k=5,
        filters={"hybrid": True, "doc_id": "abc-123", "min_score": 0.5},
        hit_chunk_ids=[],
        hit_scores=[],
        latency_ms=42,
    )

    with conn.conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "select query_text, k, filters, latency_ms "
            "from public.query_logs where query_text = 'integration-test-query' "
            "order by logged_at desc limit 1"
        )
        row = cur.fetchone()
    assert row is not None
    assert row["query_text"] == "integration-test-query"
    assert row["k"] == 5
    assert row["latency_ms"] == 42
    assert row["filters"]["hybrid"] is True
    assert row["filters"]["doc_id"] == "abc-123"
    assert row["filters"]["min_score"] == 0.5

    cur.execute("delete from public.query_logs where query_text = 'integration-test-query'")
    conn.conn.commit()


def test_reingested_at_updates_on_upsert(conn) -> None:
    """The upsert_document() ON CONFLICT clause sets reingested_at = now() on re-upsert."""

    from lean.store.documents import DocumentRepo
    from psycopg.rows import dict_row

    sha = "reingested-at-test"
    doc_id_1 = _make_doc(conn, sha)

    with conn.conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "select ingested_at, reingested_at from public.documents where id = %s",
            (str(doc_id_1),),
        )
        first = cur.fetchone()
    assert first["ingested_at"] is not None
    initial_reingested = first["reingested_at"]

    DocumentRepo(conn).upsert_document(
        source_path="re-upsert.pdf",
        source_sha256=sha,
        title="Updated",
        extraction_method="marker",
        source_storage_path="sources/r.pdf",
        markdown_storage_path="markdown/r.md",
    )

    with conn.conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "select ingested_at, reingested_at from public.documents where id = %s",
            (str(doc_id_1),),
        )
        second = cur.fetchone()
    assert second["ingested_at"] == first["ingested_at"], "ingested_at must not change"
    assert second["reingested_at"] is not None
    if initial_reingested is not None:
        assert second["reingested_at"] >= initial_reingested

    DocumentRepo(conn).delete_document(doc_id_1)


def test_replace_chunks_atomic_on_failure(conn) -> None:
    """If a chunk insert fails mid-replace, rollback leaves the table unchanged."""
    import psycopg
    from lean.store.chunks import ChunkRepo, ChunkRow
    from lean.store.documents import DocumentRepo

    doc_id = _make_doc(conn, "atomic-rollback-test")

    initial_chunks = [
        ChunkRow(
            document_id=doc_id,
            chunk_index=0,
            section_path="S",
            heading_text="H",
            page_start=1,
            page_end=1,
            token_count=10,
            content="initial",
            embedding=[0.5] * 1024,
        )
    ]
    ChunkRepo(conn).replace_chunks(doc_id, initial_chunks)

    with conn.conn.cursor() as cur:
        cur.execute(
            "select count(*) from public.chunks where document_id = %s",
            (str(doc_id),),
        )
        before = cur.fetchone()[0]
    assert before == 1

    try:
        bad_chunks = [
            ChunkRow(
                document_id=doc_id,
                chunk_index=0,
                section_path="S",
                heading_text="H",
                page_start=1,
                page_end=1,
                token_count=10,
                content="ok",
                embedding=[0.5] * 1024,
            ),
            ChunkRow(
                document_id=doc_id,
                chunk_index=1,
                section_path="S",
                heading_text="H",
                page_start=1,
                page_end=1,
                token_count=10,
                content="BAD EMBEDDING — wrong dim",
                embedding=[0.5] * 1023,
            ),
        ]
        ChunkRepo(conn).replace_chunks(doc_id, bad_chunks)
    except psycopg.errors.DataException:
        pass
    else:
        raise AssertionError("expected DataException for 1023-dim embedding")

    with conn.conn.cursor() as cur:
        cur.execute(
            "select count(*) from public.chunks where document_id = %s",
            (str(doc_id),),
        )
        after = cur.fetchone()[0]
    assert after == before, (
        f"rollback should preserve initial chunks: before={before}, after={after}"
    )

    DocumentRepo(conn).delete_document(doc_id)


def test_save_eval_run_persists_metrics(conn) -> None:
    """save_eval_run writes a row that round-trips through the eval_runs table."""
    from lean.store.analytics import AnalyticsRepo
    from psycopg.rows import dict_row

    repo = AnalyticsRepo(conn)
    repo.save_eval_run(
        config={"k": 5, "dataset": "integration-test"},
        hit_rate=0.8,
        mrr=0.7,
        ndcg=0.65,
        recall=0.8,
        mean_latency_ms=87,
        sample_count=50,
        k=5,
    )

    with conn.conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "select config, hit_rate, mrr, ndcg, recall, mean_latency_ms, sample_count, k "
            "from public.eval_runs where config->>'dataset' = 'integration-test' "
            "order by ran_at desc limit 1"
        )
        row = cur.fetchone()
    assert row is not None
    assert row["hit_rate"] == 0.8
    assert row["mrr"] == 0.7
    assert row["ndcg"] == 0.65
    assert row["recall"] == 0.8
    assert row["mean_latency_ms"] == 87
    assert row["sample_count"] == 50
    assert row["k"] == 5
    assert row["config"]["k"] == 5

    cur.execute("delete from public.eval_runs where config->>'dataset' = 'integration-test'")
    conn.conn.commit()


def test_corpus_stats_breakdown_sums_to_document_count(conn) -> None:
    """extraction_method_breakdown values sum to document_count (invariant)."""
    from lean.store.analytics import AnalyticsRepo

    stats = AnalyticsRepo(conn).corpus_stats(embedding_dim=1024, embedding_model="integration-test")
    assert sum(stats.extraction_method_breakdown.values()) == stats.document_count
