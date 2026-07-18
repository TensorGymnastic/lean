"""Unit tests for store/chunks.py and store/documents.py — mock-cursor tests."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

_VALID_KEY = "x" * 32


def _mock_cursor(fetchone_return=None, fetchall_return=None):
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchone.return_value = fetchone_return
    cursor.fetchall.return_value = fetchall_return or []
    return cursor


def _mock_conn(cursor=None):
    conn = MagicMock()
    conn.conn.cursor.return_value = cursor or _mock_cursor()
    return conn


# --- ChunkRepo tests ---


def test_replace_chunks_empty(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.chunks import ChunkRepo

    cursor = _mock_cursor()
    conn = _mock_conn(cursor)
    repo = ChunkRepo(conn)
    repo.replace_chunks(uuid.uuid4(), [])
    cursor.execute.assert_called_once()  # only the DELETE


def test_replace_chunks_with_data(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.chunks import ChunkRepo, ChunkRow

    cursor = _mock_cursor()
    conn = _mock_conn(cursor)
    repo = ChunkRepo(conn)
    doc_id = uuid.uuid4()
    chunks = [
        ChunkRow(
            document_id=doc_id,
            chunk_index=0,
            section_path="Ch 1",
            heading_text="H",
            page_start=None,
            page_end=None,
            token_count=10,
            content="content",
            embedding=[0.1] * 1024,
        )
    ]
    repo.replace_chunks(doc_id, chunks)
    cursor.execute.assert_called_once()
    cursor.executemany.assert_called_once()
    conn.conn.commit.assert_called_once()


def test_get_chunk_found(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.chunks import ChunkRepo

    fake_row = {
        "id": "00000000-0000-0000-0000-000000000001",
        "document_id": "00000000-0000-0000-0000-000000000002",
        "chunk_index": 0,
        "section_path": "Ch 1",
        "heading_text": "H",
        "page_start": None,
        "page_end": None,
        "token_count": 10,
        "content": "content",
    }
    cursor = _mock_cursor(fetchone_return=fake_row)
    conn = _mock_conn(cursor)
    repo = ChunkRepo(conn)
    chunk = repo.get_chunk(uuid.uuid4())
    assert chunk is not None
    assert chunk.content == "content"


def test_get_chunk_not_found(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.chunks import ChunkRepo

    cursor = _mock_cursor(fetchone_return=None)
    conn = _mock_conn(cursor)
    repo = ChunkRepo(conn)
    assert repo.get_chunk(uuid.uuid4()) is None


def test_count_chunks(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.chunks import ChunkRepo

    cursor = _mock_cursor(fetchone_return=(42,))
    conn = _mock_conn(cursor)
    repo = ChunkRepo(conn)
    assert repo.count_chunks() == 42


def test_get_chunks_by_document(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.chunks import ChunkRepo

    fake_rows = [
        {
            "id": f"00000000-0000-0000-0000-00000000000{i}",
            "document_id": "00000000-0000-0000-0000-000000000099",
            "chunk_index": i,
            "section_path": "Ch 1",
            "heading_text": "H",
            "page_start": None,
            "page_end": None,
            "token_count": 10,
            "content": f"chunk {i}",
        }
        for i in range(3)
    ]
    cursor = _mock_cursor(fetchall_return=fake_rows)
    conn = _mock_conn(cursor)
    repo = ChunkRepo(conn)
    chunks = repo.get_chunks_by_document(uuid.uuid4())
    assert len(chunks) == 3
    assert chunks[0].content == "chunk 0"


# --- DocumentRepo tests ---


def test_upsert_document(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.documents import DocumentRepo

    doc_uuid = uuid.uuid4()
    cursor = _mock_cursor(fetchone_return=(doc_uuid,))
    conn = _mock_conn(cursor)
    repo = DocumentRepo(conn)
    result = repo.upsert_document(
        source_path="data/test.pdf",
        source_sha256="abc123",
        title="Test",
        extraction_method="markitdown",
        authors=["Author"],
        publisher=None,
        year=2024,
        page_count=100,
        metadata={"keywords": []},
    )
    assert result == doc_uuid
    cursor.execute.assert_called_once()
    conn.conn.commit.assert_called_once()


def test_delete_document(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.documents import DocumentRepo

    cursor = _mock_cursor()
    conn = _mock_conn(cursor)
    repo = DocumentRepo(conn)
    repo.delete_document(uuid.uuid4())
    cursor.execute.assert_called_once()
    conn.conn.commit.assert_called_once()


def test_get_source_path_found(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.documents import DocumentRepo

    cursor = _mock_cursor(fetchone_return=("data/test.pdf",))
    conn = _mock_conn(cursor)
    repo = DocumentRepo(conn)
    assert repo.get_source_path(uuid.uuid4()) == "data/test.pdf"


def test_get_source_path_not_found(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.store.documents import DocumentRepo

    cursor = _mock_cursor(fetchone_return=None)
    conn = _mock_conn(cursor)
    repo = DocumentRepo(conn)
    assert repo.get_source_path(uuid.uuid4()) is None


def test_list_documents(monkeypatch):
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from datetime import datetime

    from lean.store.documents import DocumentRepo

    fake_rows = [
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "source_path": "data/test.pdf",
            "title": "Book",
            "authors": ["Author"],
            "page_count": 100,
            "extraction_method": "markitdown",
            "ingested_at": datetime(2026, 7, 12),
            "chunk_count": 10,
        }
    ]
    cursor = _mock_cursor(fetchall_return=fake_rows)
    conn = _mock_conn(cursor)
    repo = DocumentRepo(conn)
    docs = repo.list_documents()
    assert len(docs) == 1
    assert docs[0].title == "Book"
