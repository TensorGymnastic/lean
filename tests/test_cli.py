"""Unit tests for the Typer CLI."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

_VALID_KEY = "x" * 32


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/postgres")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)
    return CliRunner()


def test_cli_no_args_shows_help(runner):
    from lean.cli import app

    result = runner.invoke(app, [])
    # Typer exits 0 or 2 when showing help via no_args_is_help
    assert result.exit_code in (0, 2)
    assert "Usage" in result.output or "Commands" in result.output


def test_cli_search_no_results(runner):
    from lean.cli import app

    with patch("lean.services.search.search", return_value=[]):
        result = runner.invoke(app, ["search", "nonexistent"])
    assert result.exit_code == 0
    assert "No results" in result.output


def test_cli_search_with_results(runner):
    from lean.cli import app
    from lean.models.schemas import Chunk

    fake_chunks = [
        Chunk(
            id="00000000-0000-0000-0000-000000000001",
            document_id="00000000-0000-0000-0000-000000000002",
            chunk_index=0,
            section_path="Ch 1",
            heading_text="DMAIC",
            token_count=100,
            content="DMAIC is Define Measure Analyze Improve Control.",
            score=0.95,
        )
    ]
    with patch("lean.services.search.search", return_value=fake_chunks):
        result = runner.invoke(app, ["search", "DMAIC"])
    assert result.exit_code == 0
    assert "DMAIC" in result.output


def test_cli_corpus_stats(runner):
    from lean.cli import app
    from lean.models.schemas import CorpusStats

    fake_stats = CorpusStats(
        document_count=10,
        chunk_count=2535,
        total_tokens=859776,
        extraction_method_breakdown={"unlimited_ocr": 4, "markitdown": 6},
        embedding_dim=1024,
        embedding_model="LiquidAI/LFM2.5-Embedding-350M",
    )
    with patch("lean.services.corpus.corpus_stats", return_value=fake_stats):
        result = runner.invoke(app, ["corpus-stats", "--json"])
    assert result.exit_code == 0
    import json

    data = json.loads(result.output)
    assert data["document_count"] == 10


def test_cli_list_documents(runner):
    from datetime import datetime

    from lean.cli import app
    from lean.models.schemas import DocumentSummary, ExtractionMethod

    fake_docs = [
        DocumentSummary(
            id="00000000-0000-0000-0000-000000000001",
            source_path="data/test.pdf",
            title="Test Book",
            authors=["Author"],
            page_count=100,
            extraction_method=ExtractionMethod.MARKITDOWN,
            chunk_count=10,
            ingested_at=datetime(2026, 7, 12),
        )
    ]
    with patch("lean.services.corpus.list_documents", return_value=fake_docs):
        result = runner.invoke(app, ["list-documents"])
    assert result.exit_code == 0
    assert "Test Book" in result.output


def test_cli_get_chunk_not_found(runner):
    from lean.cli import app

    with patch("lean.services.corpus.get_chunk", return_value=None):
        result = runner.invoke(app, ["get-chunk", "00000000-0000-0000-0000-000000000000"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_cli_get_chunk_found(runner):
    from lean.cli import app
    from lean.models.schemas import Chunk

    fake = Chunk(
        id="00000000-0000-0000-0000-000000000001",
        document_id="00000000-0000-0000-0000-000000000002",
        chunk_index=0,
        section_path="Ch 1",
        heading_text="DMAIC",
        token_count=50,
        content="DMAIC content",
    )
    with patch("lean.services.corpus.get_chunk", return_value=fake):
        result = runner.invoke(app, ["get-chunk", "00000000-0000-0000-0000-000000000001", "--json"])
    assert result.exit_code == 0


def test_cli_get_markdown(runner):
    from lean.cli import app

    with patch("lean.services.corpus.get_document_markdown", return_value="# Title\n\nContent"):
        result = runner.invoke(app, ["get-markdown", "doc-id"])
    assert result.exit_code == 0
    assert "Title" in result.output


def test_cli_get_markdown_json(runner):
    from lean.cli import app

    with patch("lean.services.corpus.get_document_markdown", return_value="# MD"):
        result = runner.invoke(app, ["get-markdown", "doc-id", "--json"])
    assert result.exit_code == 0
    import json

    data = json.loads(result.output)
    assert "markdown" in data


def test_cli_delete(runner):
    from lean.cli import app

    with patch("lean.services.corpus.delete_document", return_value={"deleted": "doc-id"}):
        result = runner.invoke(app, ["delete", "doc-id"])
    assert result.exit_code == 0
    assert "Deleted" in result.output


def test_cli_delete_json(runner):
    from lean.cli import app

    with patch("lean.services.corpus.delete_document", return_value={"deleted": "doc-id"}):
        result = runner.invoke(app, ["delete", "doc-id", "--json"])
    assert result.exit_code == 0
    import json

    data = json.loads(result.output)
    assert data["deleted"] == "doc-id"


def test_cli_reingest_all(runner):
    from datetime import datetime

    from lean.cli import app
    from lean.models.schemas import DocumentSummary, ExtractionMethod

    fake_docs = [
        DocumentSummary(
            id="00000000-0000-0000-0000-000000000001",
            source_path="data/test.pdf",
            title="Book",
            authors=[],
            page_count=10,
            extraction_method=ExtractionMethod.MARKITDOWN,
            chunk_count=5,
            ingested_at=datetime(2026, 7, 12),
        )
    ]
    with (
        patch("lean.services.corpus.list_documents", return_value=fake_docs),
    ):
        result = runner.invoke(app, ["reingest-all", "--json"])
    assert result.exit_code == 0


def test_cli_search_json(runner):
    from lean.cli import app
    from lean.models.schemas import Chunk

    fake_chunks = [
        Chunk(
            id="00000000-0000-0000-0000-000000000001",
            document_id="00000000-0000-0000-0000-000000000002",
            chunk_index=0,
            section_path="Ch 1",
            token_count=50,
            content="content",
            score=0.9,
        )
    ]
    with patch("lean.services.search.search", return_value=fake_chunks):
        result = runner.invoke(app, ["search", "query", "--json"])
    assert result.exit_code == 0
