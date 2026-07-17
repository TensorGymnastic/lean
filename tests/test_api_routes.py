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
