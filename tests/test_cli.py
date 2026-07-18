"""Unit tests for the Typer CLI."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

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
    import json
    from datetime import datetime

    from lean.cli import app
    from lean.models.schemas import DocumentSummary, ExtractionMethod, IngestResult

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
    fake_result = IngestResult(
        document_id="00000000-0000-0000-0000-000000000001",
        source_sha256="abc",
        page_count=10,
        extraction_method=ExtractionMethod.MARKITDOWN,
        chunk_count=8,
        elapsed_seconds=3.0,
    )
    with (
        patch("lean.services.corpus.list_documents", return_value=fake_docs),
        patch("lean.services.ingestion.reingest", new_callable=AsyncMock, return_value=fake_result),
    ):
        result = runner.invoke(app, ["reingest-all", "--json"])
    assert result.exit_code == 0
    lines = result.stdout.strip().split("\n")
    json_start = next(i for i, line in enumerate(lines) if line.strip().startswith("{"))
    output = json.loads("\n".join(lines[json_start:]))
    assert output["success"] == 1
    assert output["failed"] == 0
    assert output["skipped"] == 0


def test_cli_reingest_all_skips_ocr_without_force(runner):
    import json
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
            extraction_method=ExtractionMethod.UNLIMITED_OCR,
            chunk_count=5,
            ingested_at=datetime(2026, 7, 12),
        )
    ]
    with (
        patch("lean.services.corpus.list_documents", return_value=fake_docs),
        patch("lean.services.ingestion.reingest", new_callable=AsyncMock) as mock_reingest,
    ):
        result = runner.invoke(app, ["reingest-all", "--json"])
    assert result.exit_code == 0
    lines = result.stdout.strip().split("\n")
    json_start = next(i for i, line in enumerate(lines) if line.strip().startswith("{"))
    output = json.loads("\n".join(lines[json_start:]))
    assert output["skipped"] == 1
    assert output["success"] == 0
    mock_reingest.assert_not_called()


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


def test_cli_eval_with_dataset_flag_uses_curated_loader(runner, tmp_path, monkeypatch):
    """--dataset <path> loads samples via load_curated_dataset, skipping build_eval_dataset."""
    import json
    from pathlib import Path

    from lean.cli import app
    from lean.eval.runner import EvalSample

    dataset_path: Path = tmp_path / "curated.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "query": "What is DMAIC?",
                    "expected_chunk_id": "11111111-1111-1111-1111-111111111111",
                },
                {
                    "query": "What is Pareto?",
                    "expected_chunk_id": "22222222-2222-2222-2222-222222222222",
                },
            ]
        )
    )

    fake_samples = [
        EvalSample(
            query="What is DMAIC?", expected_chunk_id="11111111-1111-1111-1111-111111111111"
        ),
        EvalSample(
            query="What is Pareto?", expected_chunk_id="22222222-2222-2222-2222-222222222222"
        ),
    ]

    with (
        patch("lean.eval.runner.load_curated_dataset", return_value=fake_samples) as mock_load,
        patch("lean.eval.runner.build_eval_dataset") as mock_build,  # must NOT be called
        patch(
            "lean.eval.runner.evaluate",
            return_value=type(
                "FakeResult",
                (),
                {
                    "hit_rate": 0.5,
                    "mrr": 0.5,
                    "ndcg": 0.5,
                    "recall": 0.5,
                    "mean_latency_ms": 10.0,
                    "sample_count": 2,
                    "k": 5,
                },
            )(),
        ),
        patch("lean.store.base.StoreConnection") as mock_store_cls,
        patch("lean.store.analytics.AnalyticsRepo.save_eval_run") as mock_save,
    ):
        mock_store_cls.return_value.close.return_value = None
        result = runner.invoke(app, ["eval", "--dataset", str(dataset_path), "--k", "5"])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Loading curated dataset" in result.output
    assert "Building eval dataset" not in result.output
    mock_load.assert_called_once_with(dataset_path)
    mock_build.assert_not_called()
    mock_save.assert_called_once()
    saved_config = mock_save.call_args.kwargs.get("config", {})
    assert saved_config.get("dataset") == str(dataset_path)


def test_cli_eval_without_dataset_flag_uses_builder(runner, monkeypatch):
    """Without --dataset, build_eval_dataset is used (existing behavior preserved)."""
    from lean.cli import app
    from lean.eval.runner import EvalSample

    with (
        patch(
            "lean.eval.runner.build_eval_dataset", return_value=[EvalSample("q", "id")]
        ) as mock_build,
        patch("lean.eval.runner.load_curated_dataset") as mock_load,  # must NOT be called
        patch(
            "lean.eval.runner.evaluate",
            return_value=type(
                "FakeResult",
                (),
                {
                    "hit_rate": 0.0,
                    "mrr": 0.0,
                    "ndcg": 0.0,
                    "recall": 0.0,
                    "mean_latency_ms": 0.0,
                    "sample_count": 1,
                    "k": 5,
                },
            )(),
        ),
        patch("lean.store.base.StoreConnection") as mock_store_cls,
        patch("lean.store.analytics.AnalyticsRepo.save_eval_run"),
    ):
        mock_store_cls.return_value.close.return_value = None
        result = runner.invoke(app, ["eval", "--sample-size", "1", "--k", "5"])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Building eval dataset" in result.output
    mock_build.assert_called_once()
    mock_load.assert_not_called()
