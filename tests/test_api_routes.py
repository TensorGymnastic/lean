"""Unit tests for the FastAPI REST API routes."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

_VALID_KEY = "x" * 32


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    from lean.api.routes import app

    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {_VALID_KEY}"}


def test_health_no_auth_required(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_without_token_returns_401(client):
    response = client.get("/search", params={"query": "DMAIC"})
    assert response.status_code == 401


def test_search_with_wrong_token_returns_401(client):
    response = client.get(
        "/search",
        params={"query": "DMAIC"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401


def test_search_with_valid_token_returns_results(client, auth_headers):
    from lean.models.schemas import Chunk

    fake_chunks = [
        Chunk(
            id="00000000-0000-0000-0000-000000000001",
            document_id="00000000-0000-0000-0000-000000000002",
            chunk_index=0,
            section_path="Ch 1",
            heading_text="DMAIC",
            token_count=100,
            content="DMAIC is a methodology.",
            score=0.95,
        )
    ]
    with patch("lean.api.routes._search", return_value=fake_chunks):
        response = client.get(
            "/search",
            params={"query": "DMAIC", "k": 5},
            headers=auth_headers,
        )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert "DMAIC" in data[0]["content"]


def test_documents_with_valid_token(client, auth_headers):
    from datetime import datetime

    from lean.models.schemas import DocumentSummary, ExtractionMethod

    fake_docs = [
        DocumentSummary(
            id="00000000-0000-0000-0000-000000000001",
            source_path="data/test.pdf",
            title="Test Doc",
            authors=["Author"],
            page_count=100,
            extraction_method=ExtractionMethod.MARKITDOWN,
            chunk_count=10,
            ingested_at=datetime(2026, 7, 12),
        )
    ]
    with patch("lean.api.routes._list", return_value=fake_docs):
        response = client.get("/documents", headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_stats_with_valid_token(client, auth_headers):
    from lean.models.schemas import CorpusStats

    fake_stats = CorpusStats(
        document_count=10,
        chunk_count=2535,
        total_tokens=859776,
        extraction_method_breakdown={"unlimited_ocr": 4, "markitdown": 6},
        embedding_dim=1024,
        embedding_model="LiquidAI/LFM2.5-Embedding-350M",
    )
    with patch("lean.api.routes._corpus_stats", return_value=fake_stats):
        response = client.get("/stats", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["document_count"] == 10


def test_ingest_with_valid_token(client, auth_headers):
    from lean.models.schemas import ExtractionMethod, IngestResult

    fake_result = IngestResult(
        document_id="00000000-0000-0000-0000-000000000001",
        source_sha256="abc123",
        page_count=100,
        extraction_method=ExtractionMethod.MARKITDOWN,
        chunk_count=10,
        elapsed_seconds=3.5,
    )
    with patch("lean.api.routes._ingest", return_value=fake_result):
        response = client.post(
            "/ingest",
            params={"path": "data/test.pdf"},
            headers=auth_headers,
        )
    assert response.status_code == 200
    assert response.json()["document_id"] == "00000000-0000-0000-0000-000000000001"


def test_search_empty_query_returns_400(client, auth_headers):
    """Domain ValueError (e.g. empty query) maps to HTTP 400, not 500."""
    with patch("lean.api.routes._search", side_effect=ValueError("query must not be empty")):
        response = client.get("/search", params={"query": "x"}, headers=auth_headers)
    assert response.status_code == 400
    assert "query must not be empty" in response.json()["detail"]


def test_ingest_path_outside_corpus_returns_403(client, auth_headers):
    """Path-traversal PermissionError maps to HTTP 403, not 500."""
    with patch(
        "lean.api.routes._ingest", side_effect=PermissionError("path outside corpus root: x")
    ):
        response = client.post("/ingest", params={"path": "x"}, headers=auth_headers)
    assert response.status_code == 403
    assert "outside corpus root" in response.json()["detail"]


def test_get_chunk_found(client, auth_headers):
    from lean.models.schemas import Chunk

    fake_chunk = Chunk(
        id="00000000-0000-0000-0000-000000000001",
        document_id="00000000-0000-0000-0000-000000000002",
        chunk_index=0,
        section_path="Ch 1",
        heading_text="DMAIC",
        token_count=100,
        content="DMAIC is a methodology.",
        score=0.95,
    )
    with patch("lean.api.routes._get_chunk", return_value=fake_chunk):
        response = client.get("/chunks/00000000-0000-0000-0000-000000000001", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["content"] == "DMAIC is a methodology."


def test_get_chunk_not_found(client, auth_headers):
    with patch("lean.api.routes._get_chunk", return_value=None):
        response = client.get("/chunks/00000000-0000-0000-0000-000000000099", headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "chunk not found"


def test_get_document_markdown_found(client, auth_headers):
    with patch("lean.api.routes._get_markdown", return_value="# Test Document\n\nContent"):
        response = client.get(
            "/documents/00000000-0000-0000-0000-000000000001/markdown", headers=auth_headers
        )
    assert response.status_code == 200
    body = response.json()
    assert body["markdown"] == "# Test Document\n\nContent"
    assert body["document_id"] == "00000000-0000-0000-0000-000000000001"


def test_get_document_markdown_not_found(client, auth_headers):
    with patch("lean.api.routes._get_markdown", side_effect=KeyError("not found")):
        response = client.get(
            "/documents/00000000-0000-0000-0000-000000000099/markdown", headers=auth_headers
        )
    assert response.status_code == 404
    assert response.json()["detail"] == "document not found"


def test_delete_document(client, auth_headers):
    doc_id = "00000000-0000-0000-0000-000000000001"
    with patch("lean.api.routes._delete", return_value={"deleted": doc_id}):
        response = client.delete(f"/documents/{doc_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"deleted": doc_id}
