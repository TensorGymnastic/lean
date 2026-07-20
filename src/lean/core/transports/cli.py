"""Universal Typer CLI factory.

Builds a Typer app with shared commands (health, db-init, mcp-serve,
api-serve, serve) and lets the domain register its own commands via
``register_fn(app, services, settings)``.

The shared commands work the same regardless of domain — they're the
universal "operators" every lean deployment needs.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import typer

from lean.core.config.settings import CoreSettings

logger = logging.getLogger(__name__)


def _build_cli_app(
    *,
    name: str,
    help_text: str,
    no_args_is_help: bool = True,
) -> typer.Typer:
    return typer.Typer(name=name, help=help_text, no_args_is_help=no_args_is_help)


def _add_universal_commands(
    app: typer.Typer,
    *,
    settings_factory: Callable[[], CoreSettings],
    build_mcp_fn: Callable[[], Any],
    build_api_fn: Callable[[], Any],
    serve_mcp_fn: Callable[..., None],
    serve_api_fn: Callable[..., None],
) -> None:
    """Add the universal operator commands (health, db-init, mcp-serve, api-serve).

    These commands are not domain-specific — they're the boot surface
    every lean deployment needs.
    """

    @app.callback()
    def _main(
        ctx: typer.Context,
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
    ) -> None:
        if verbose:
            level = logging.DEBUG
        else:
            try:
                level = getattr(logging, settings_factory().log_level.upper(), logging.INFO)
            except Exception:
                level = logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )

    @app.command()
    def db_init() -> None:
        """Apply SQL migrations from db/schemas/ to the configured database."""
        import os

        db_url = settings_factory().db_url
        parsed = urlparse(db_url)
        pg_env = {**os.environ}
        if parsed.hostname:
            pg_env["PGHOST"] = parsed.hostname
        if parsed.port:
            pg_env["PGPORT"] = str(parsed.port)
        if parsed.username:
            pg_env["PGUSER"] = parsed.username
        if parsed.password:
            pg_env["PGPASSWORD"] = unquote(parsed.password)
        if parsed.path and len(parsed.path) > 1:
            pg_env["PGDATABASE"] = parsed.path[1:]

        for sql_file in sorted(Path("db/schemas").glob("*.sql")):
            typer.echo(f"applying {sql_file}...")
            subprocess.run(
                ["psql", "-v", "ON_ERROR_STOP=1", "-f", str(sql_file)],
                env=pg_env,
                check=True,
            )
        typer.echo("schema applied.")

    @app.command(name="mcp-serve")
    def mcp_serve(
        transport: str = typer.Option("stdio", help="stdio or http"),
        host: str = typer.Option(None, help="HTTP bind host"),
        port: int = typer.Option(None, help="HTTP port"),
    ) -> None:
        """Start the MCP server."""
        settings = settings_factory()
        mcp = build_mcp_fn()
        serve_mcp_fn(
            mcp,
            transport=transport,
            host=host or settings.mcp_http_host,
            port=port or settings.mcp_http_port,
            settings=settings,
        )

    @app.command(name="api-serve")
    def api_serve(
        reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    ) -> None:
        """Start the FastAPI REST API server (bearer-authed)."""
        settings = settings_factory()
        app = build_api_fn()
        serve_api_fn(app, host=settings.mcp_http_host, port=settings.api_port, reload=reload)

    @app.command()
    def health(
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """Check connectivity to dependencies (DB, embedder, optional VLM/LLM)."""
        from lean.core.transports.health import check_health

        results = check_health(settings_factory())
        if json_output:
            typer.echo(json.dumps(results, indent=2, default=str))
        else:
            for name, info in results.items():
                status = info.get("status", "unknown")
                label = (
                    "[OK]" if status == "ok" else "[--]" if status == "not_configured" else "[FAIL]"
                )
                typer.echo(f"{label} {name}: {status}")
        if any(info.get("status") == "error" for info in results.values()):
            raise typer.Exit(1)

    @app.command(name="eval")
    def eval_cmd(
        sample_size: int = typer.Option(50, "--sample-size", help="Pseudo-query sample size"),
        k: int = typer.Option(5, "--k", help="Top-k for hit/metric computation"),
        dataset: Path | None = typer.Option(None, "--dataset", help="Curated JSON dataset path"),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """Run retrieval evaluation: hit_rate@k, MRR@k, NDCG@k, Recall@k.

        If ``--dataset`` is given, the curated JSON is used; otherwise a
        pseudo-dataset is built from random sampled chunks. The full search
        pipeline is NOT exercised — see docs/evaluation.md for caveats.
        """
        from lean.core.eval import evaluate as eval_run
        from lean.core.eval import load_curated_dataset
        from lean.core.eval.runner import build_eval_dataset
        from lean.core.store.base import StoreConnection

        settings = settings_factory()
        db_url = settings.db_url
        if dataset is not None:
            samples = load_curated_dataset(dataset)
        else:
            with StoreConnection(db_url) as conn:
                samples = build_eval_dataset(conn, sample_size=sample_size, seed=settings.eval_seed)

        with StoreConnection(db_url) as conn:
            result = eval_run(conn, samples, k=k)

        if json_output:
            typer.echo(json.dumps(result.__dict__, indent=2))
        else:
            typer.echo(
                f"hit_rate@{k}={result.hit_rate:.3f}  "
                f"MRR@{k}={result.mrr:.3f}  "
                f"NDCG@{k}={result.ndcg:.3f}  "
                f"Recall@{k}={result.recall:.3f}  "
                f"latency={result.mean_latency_ms:.0f}ms  "
                f"n={result.sample_count}"
            )


def build_cli(
    *,
    name: str,
    help_text: str,
    settings_factory: Callable[[], CoreSettings],
    build_mcp_fn: Callable[[], Any],
    build_api_fn: Callable[[], Any],
    serve_mcp_fn: Callable[..., None],
    serve_api_fn: Callable[..., None],
    register_fn: Callable[..., None] | None = None,
    services: dict[str, object] | None = None,
    settings: CoreSettings | None = None,
) -> typer.Typer:
    """Build the Typer app with universal commands + domain commands."""
    app = _build_cli_app(name=name, help_text=help_text)
    _add_universal_commands(
        app,
        settings_factory=settings_factory,
        build_mcp_fn=build_mcp_fn,
        build_api_fn=build_api_fn,
        serve_mcp_fn=serve_mcp_fn,
        serve_api_fn=serve_api_fn,
    )
    if register_fn is not None:
        register_fn(app, services or {}, settings)
    return app


__all__ = ["build_cli"]
