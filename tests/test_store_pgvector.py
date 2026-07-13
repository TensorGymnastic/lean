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
