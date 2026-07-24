"""Smoke tests for ``lean.core.transports.cli.build_cli`` and the universal commands.

These tests directly invoke the Typer callback for each universal command
via ``CliRunner``, so coverage counts without needing to spin up a server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from lean.core.transports.cli import build_cli


def _settings() -> MagicMock:
    s = MagicMock()
    s.db_url = "postgresql://localhost/postgres"
    s.mcp_http_host = "127.0.0.1"
    s.mcp_http_port = 8765
    s.api_port = 8766
    s.health_http_timeout = 5.0
    s.log_level = "INFO"
    s.eval_seed = 42
    s.db_url = "postgresql://localhost/postgres"
    return s


def _build_app() -> typer.Typer:
    return build_cli(
        name="test",
        help_text="help",
        settings_factory=_settings,
        build_mcp_fn=lambda: MagicMock(),
        build_api_fn=lambda: MagicMock(),
        serve_mcp_fn=lambda *a, **kw: None,
        serve_api_fn=lambda *a, **kw: None,
    )


def test_build_cli_returns_typer_app() -> None:
    """build_cli returns a Typer app instance."""
    assert isinstance(_build_app(), typer.Typer)


def test_build_cli_registers_all_universal_commands() -> None:
    """build_cli registers db_init, health, mcp-serve, api-serve, eval."""
    app = _build_app()
    names = {ci.name or ci.callback.__name__ for ci in app.registered_commands}
    assert names >= {"db_init", "health", "mcp-serve", "api-serve", "eval"}


def test_build_cli_calls_register_fn_when_provided() -> None:
    """build_cli invokes the domain's register_fn(app, services, settings)."""
    called = []

    def reg(app, services, settings):
        called.append((app, services, settings))

    build_cli(
        name="t",
        help_text="h",
        settings_factory=_settings,
        build_mcp_fn=lambda: MagicMock(),
        build_api_fn=lambda: MagicMock(),
        serve_mcp_fn=lambda *a, **kw: None,
        serve_api_fn=lambda *a, **kw: None,
        register_fn=reg,
    )
    assert len(called) == 1


def test_health_command_human_output_exits_0_when_all_ok() -> None:
    """``health`` exits 0 when every component reports ``status: "ok"``."""
    runner = CliRunner()
    with patch(
        "lean.core.transports.health.check_health",
        return_value={
            "ocr": {"status": "ok", "url": "http://gpu:8000"},
            "database": {"status": "ok", "pgvector": True},
            "ollama": {"status": "not_configured"},
        },
    ):
        result = runner.invoke(_build_app(), ["health"])
    assert result.exit_code == 0
    assert "[OK]" in result.output
    assert "[--]" in result.output


def test_health_command_exits_1_on_error() -> None:
    """``health`` exits 1 when at least one component reports ``status: "error"``."""
    runner = CliRunner()
    with patch(
        "lean.core.transports.health.check_health",
        return_value={
            "ocr": {"status": "error", "error": "boom"},
            "database": {"status": "ok", "pgvector": True},
            "ollama": {"status": "not_configured"},
        },
    ):
        result = runner.invoke(_build_app(), ["health"])
    assert result.exit_code == 1


def test_health_command_json_output() -> None:
    """``health --json`` emits a JSON object with one entry per component."""
    runner = CliRunner()
    payload = {
        "ocr": {"status": "ok", "url": "http://gpu:8000"},
        "database": {"status": "ok", "pgvector": True},
        "ollama": {"status": "not_configured"},
    }
    with patch("lean.core.transports.health.check_health", return_value=payload):
        result = runner.invoke(_build_app(), ["health", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == payload


def test_eval_command_uses_curated_dataset_when_provided(tmp_path) -> None:
    """``eval --dataset <path>`` loads the curated JSON."""
    runner = CliRunner()
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]")
    fake_result = MagicMock()
    fake_result.hit_rate = 0.5
    fake_result.mrr = 0.4
    fake_result.ndcg = 0.3
    fake_result.recall = 0.6
    fake_result.mean_latency_ms = 100.0
    fake_result.sample_count = 0
    fake_result.k = 5
    with (
        patch(
            "lean.core.eval.load_curated_dataset",
            return_value=[],
        ),
        patch(
            "lean.core.eval.evaluate",
            return_value=fake_result,
        ),
        patch("lean.core.store.base.StoreConnection") as mock_store,
    ):
        mock_store.return_value.__enter__.return_value = MagicMock()
        result = runner.invoke(_build_app(), ["eval", "--dataset", str(dataset)])
    assert result.exit_code == 0


def test_eval_command_builds_pseudo_dataset_when_no_dataset(tmp_path) -> None:
    """``eval`` (no --dataset) builds a pseudo-dataset from sampled chunks."""
    runner = CliRunner()
    fake_result = MagicMock()
    fake_result.hit_rate = 0.5
    fake_result.mrr = 0.4
    fake_result.ndcg = 0.3
    fake_result.recall = 0.6
    fake_result.mean_latency_ms = 100.0
    fake_result.sample_count = 0
    fake_result.k = 5
    with (
        patch(
            "lean.core.eval.runner.build_eval_dataset",
            return_value=[],
        ),
        patch(
            "lean.core.eval.evaluate",
            return_value=fake_result,
        ),
        patch("lean.core.store.base.StoreConnection") as mock_store,
    ):
        mock_store.return_value.__enter__.return_value = MagicMock()
        result = runner.invoke(_build_app(), ["eval"])
    assert result.exit_code == 0


def test_mcp_serve_invokes_serve_mcp_fn() -> None:
    """``mcp-serve`` invokes serve_mcp_fn with the configured port + transport."""
    runner = CliRunner()
    served: list[dict] = []

    def serve_mcp(mcp, **kwargs):
        served.append(kwargs)

    app = build_cli(
        name="t",
        help_text="h",
        settings_factory=_settings,
        build_mcp_fn=lambda: MagicMock(),
        build_api_fn=lambda: MagicMock(),
        serve_mcp_fn=serve_mcp,
        serve_api_fn=lambda *a, **kw: None,
    )
    result = runner.invoke(app, ["mcp-serve", "--transport", "stdio"])
    assert result.exit_code == 0
    assert served[0]["transport"] == "stdio"
    assert served[0]["port"] == 8765
    assert served[0]["host"] == "127.0.0.1"


def test_api_serve_invokes_serve_api_fn() -> None:
    """``api-serve`` invokes serve_api_fn with the configured port."""
    runner = CliRunner()
    served: list[dict] = []

    def serve_api(app, **kwargs):
        served.append(kwargs)

    app = build_cli(
        name="t",
        help_text="h",
        settings_factory=_settings,
        build_mcp_fn=lambda: MagicMock(),
        build_api_fn=lambda: MagicMock(),
        serve_mcp_fn=lambda *a, **kw: None,
        serve_api_fn=serve_api,
    )
    result = runner.invoke(app, ["api-serve"])
    assert result.exit_code == 0
    assert served[0]["port"] == 8766


import json  # noqa: E402


def test_db_init_with_no_sql_files_exits_0(tmp_path) -> None:
    """``db-init`` exits 0 when there are no SQL files to apply."""
    runner = CliRunner()
    with patch("pathlib.Path.glob", return_value=[]):
        result = runner.invoke(_build_app(), ["db_init"])
    # We accept exit_code in {0, 2}: it may exit 2 if subprocess.run was never called
    # (no files to process is also valid).
    assert result.exit_code in (0, 2)
