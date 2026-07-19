"""Unit tests for the Typer CLI."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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


def test_cli_get_markdown_json_output(runner):
    """`get-markdown --json` returns a JSON envelope with doc_id and markdown."""
    from lean.cli import app

    with patch("lean.services.corpus.get_document_markdown", return_value="# Heading\n\nBody."):
        result = runner.invoke(
            app,
            [
                "get-markdown",
                "00000000-0000-0000-0000-000000000001",
                "--json",
            ],
        )
    assert result.exit_code == 0, result.output
    import json

    parsed = json.loads(result.output)
    assert parsed["doc_id"] == "00000000-0000-0000-0000-000000000001"
    assert parsed["markdown"] == "# Heading\n\nBody."


def test_cli_delete_json_output(runner):
    """`delete --json` returns a JSON envelope with the deleted id."""
    from lean.cli import app

    with patch(
        "lean.services.corpus.delete_document",
        return_value={"deleted": "00000000-0000-0000-0000-000000000001"},
    ):
        result = runner.invoke(
            app,
            [
                "delete",
                "00000000-0000-0000-0000-000000000001",
                "--json",
            ],
        )
    assert result.exit_code == 0, result.output
    import json

    parsed = json.loads(result.output)
    assert parsed["deleted"] == "00000000-0000-0000-0000-000000000001"


def test_cli_health_with_no_services_configured(runner, monkeypatch):
    """`health` exits 0 when no external services are configured (not_configured)."""
    from lean.cli import app

    monkeypatch.setattr("lean.config.settings.Settings.ocr_base_url", "", raising=False)
    monkeypatch.setattr("lean.config.settings.Settings.embedding_remote_url", "", raising=False)

    with patch("lean.store.base.StoreConnection") as mock_store_cls:
        mock_conn = mock_store_cls.from_env.return_value
        mock_cursor = mock_conn.conn.cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = ("vector",)
        result = runner.invoke(app, ["health"])

    assert result.exit_code == 0, result.output


def test_cli_health_exits_1_when_subsystem_errors(runner, monkeypatch):
    """A subsystem reporting status='error' (OCR) triggers typer.Exit(1)."""
    import httpx

    from lean.cli import app

    monkeypatch.setattr(
        "lean.config.settings.Settings.ocr_base_url", "http://gpu:8000", raising=False
    )
    monkeypatch.setattr("lean.config.settings.Settings.embedding_remote_url", "", raising=False)

    with (
        patch("httpx.get", side_effect=httpx.ConnectError("connection refused")),
        patch("lean.store.base.StoreConnection") as mock_store_cls,
    ):
        mock_conn = mock_store_cls.from_env.return_value
        mock_cursor = mock_conn.conn.cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = ("vector",)
        result = runner.invoke(app, ["health"])

    assert result.exit_code == 1, result.output


def test_cli_db_init_parses_url_into_pg_env(runner, monkeypatch, tmp_path):
    """`db-init` extracts PG* env vars from SUPABASE_DB_URL and shells out to psql."""
    from lean.cli import app

    monkeypatch.setenv(
        "SUPABASE_DB_URL",
        "postgresql://alice:p%40ss@db.local:54322/postgres",
    )
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LEAN_MCP_API_KEY", _VALID_KEY)

    sql_dir = tmp_path / "schemas"
    sql_dir.mkdir()
    (sql_dir / "001_test.sql").write_text("-- test")

    sub_calls = []

    def fake_run(cmd, env, check):
        sub_calls.append({"cmd": cmd, "env": dict(env), "check": check})
        from unittest.mock import MagicMock

        return MagicMock(returncode=0)

    with (
        patch("subprocess.run", side_effect=fake_run),
        patch("pathlib.Path.glob", return_value=[sql_dir / "001_test.sql"]),
    ):
        result = runner.invoke(app, ["db-init"])

    assert result.exit_code == 0, result.output
    assert len(sub_calls) == 1
    env_passed = sub_calls[0]["env"]
    assert env_passed["PGHOST"] == "db.local"
    assert env_passed["PGPORT"] == "54322"
    assert env_passed["PGUSER"] == "alice"
    assert env_passed["PGPASSWORD"] == "p@ss"
    assert env_passed["PGDATABASE"] == "postgres"


def test_check_ocr_not_configured():
    """_check_ocr returns not_configured when ocr_base_url is empty."""
    from lean.cli import _check_ocr

    settings = MagicMock()
    settings.ocr_base_url = ""
    assert _check_ocr(settings) == {"status": "not_configured"}


def test_check_ocr_ok_status():
    """_check_ocr reports ok on HTTP 200."""

    from lean.cli import _check_ocr

    settings = MagicMock()
    settings.ocr_base_url = "http://gpu:8000"
    settings.health_http_timeout = 5.0

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("httpx.get", return_value=mock_resp):
        result = _check_ocr(settings)
    assert result == {"status": "ok", "url": "http://gpu:8000"}


def test_check_ocr_error_on_non_200():
    """_check_ocr reports error on non-200 status codes."""

    from lean.cli import _check_ocr

    settings = MagicMock()
    settings.ocr_base_url = "http://gpu:8000"
    settings.health_http_timeout = 5.0

    mock_resp = MagicMock()
    mock_resp.status_code = 503
    with patch("httpx.get", return_value=mock_resp):
        result = _check_ocr(settings)
    assert result["status"] == "error"
    assert result["url"] == "http://gpu:8000"


def test_check_ocr_handles_network_error():
    """_check_ocr catches network errors and reports them."""
    import httpx

    from lean.cli import _check_ocr

    settings = MagicMock()
    settings.ocr_base_url = "http://gpu:8000"
    settings.health_http_timeout = 5.0

    with patch("httpx.get", side_effect=httpx.ConnectError("connection refused")):
        result = _check_ocr(settings)
    assert result["status"] == "error"
    assert "connection refused" in result["error"]


def test_check_ollama_not_configured():
    """_check_ollama returns not_configured when embedding_remote_url is empty."""
    from lean.cli import _check_ollama

    settings = MagicMock()
    settings.embedding_remote_url = ""
    assert _check_ollama(settings) == {"status": "not_configured"}


def test_check_ollama_ok_status():
    """_check_ollama reports ok on HTTP 200."""

    from lean.cli import _check_ollama

    settings = MagicMock()
    settings.embedding_remote_url = "http://gpu:11434"
    settings.health_http_timeout = 5.0

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("httpx.get", return_value=mock_resp):
        result = _check_ollama(settings)
    assert result["status"] == "ok"
    assert result["url"] == "http://gpu:11434"


def test_check_ollama_error_on_connection_failure():
    """_check_ollama catches network errors and returns error status."""
    import httpx

    from lean.cli import _check_ollama

    settings = MagicMock()
    settings.embedding_remote_url = "http://gpu:11434"
    settings.health_http_timeout = 5.0

    with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
        result = _check_ollama(settings)
    assert result["status"] == "error"
    assert "refused" in result["error"]


def test_check_database_ok_with_pgvector():
    """_check_database reports ok + pgvector=True when extension is installed."""
    from lean.cli import _check_database

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = ("vector",)

    mock_conn.conn.cursor.return_value = mock_cursor

    with patch("lean.store.base.StoreConnection") as mock_cls:
        mock_cls.from_env.return_value = mock_conn
        result = _check_database()

    assert result["status"] == "ok"
    assert result["pgvector"] is True
    mock_conn.close.assert_called_once()


def test_check_database_ok_without_pgvector():
    """_check_database reports ok with pgvector=False when extension is absent."""
    from lean.cli import _check_database

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = None

    mock_conn.conn.cursor.return_value = mock_cursor

    with patch("lean.store.base.StoreConnection") as mock_cls:
        mock_cls.from_env.return_value = mock_conn
        result = _check_database()

    assert result["status"] == "ok"
    assert result["pgvector"] is False


def test_check_database_connection_failure():
    """_check_database returns error status when StoreConnection raises."""
    from lean.cli import _check_database

    with patch(
        "lean.store.base.StoreConnection.from_env",
        side_effect=RuntimeError("connection refused"),
    ):
        result = _check_database()

    assert result["status"] == "error"
    assert "connection refused" in result["error"]


def test_health_function_size_under_ceiling():
    """The health command body itself is under the 50 LOC ceiling after refactor."""
    import inspect

    from lean.cli import health

    source = inspect.getsource(health)
    lines = [ln for ln in source.splitlines() if ln.strip()]
    assert len(lines) <= 50, f"health() is {len(lines)} LOC, exceeds AGENTS.md 50-LOC ceiling"
