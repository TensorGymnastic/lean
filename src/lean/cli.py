"""Typer CLI for lean: full parity with MCP tools via direct service calls."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from lean.config.settings import Settings
    from lean.eval.runner import EvalResult
    from lean.store.base import StoreConnection

app = typer.Typer(
    name="lean",
    help="Lean Six Sigma corpus MCP server and CLI.",
    no_args_is_help=True,
)


@app.callback()
def _main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    if verbose:
        level = logging.DEBUG
    else:
        try:
            from lean.config.settings import get_settings

            level = getattr(logging, get_settings().log_level.upper(), logging.INFO)
        except Exception:
            level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _output(data: object, json_mode: bool) -> None:
    if hasattr(data, "model_dump_json"):
        typer.echo(data.model_dump_json(indent=2))
        return
    if isinstance(data, list):
        if json_mode:
            typer.echo(
                json.dumps(
                    [d.model_dump(mode="json") if hasattr(d, "model_dump") else d for d in data],
                    indent=2,
                    default=str,
                )
            )
        else:
            for item in data:
                if hasattr(item, "model_dump_json"):
                    typer.echo(item.model_dump_json(indent=2))
                else:
                    typer.echo(item)
    else:
        typer.echo(json.dumps(data, indent=2, default=str))


@app.command()
def ingest(
    path: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Ingest a PDF into the corpus."""
    from lean.services.ingestion import ingest_pdf

    result = asyncio.run(ingest_pdf(path))
    _output(result, json_output)


@app.command()
def search(
    query: str,
    k: int = typer.Option(5, help="Number of results"),
    doc_id: str = typer.Option(None, help="Filter by document UUID"),
    section: str = typer.Option(None, help="Filter by section substring"),
    author: str = typer.Option(None, help="Filter by author substring"),
    year_min: int = typer.Option(None, help="Filter by min publication year"),
    year_max: int = typer.Option(None, help="Filter by max publication year"),
    min_score: float = typer.Option(None, help="Minimum cosine similarity"),
    chunk_type: str = typer.Option(
        None,
        help="Filter by chunk type: 'text' or 'image'. "
        "Omit for both (default). Typo returns ValueError.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Semantic + hybrid search over the corpus."""
    from lean.services.search import search as _search

    chunks = _search(
        query,
        k=k,
        doc_id=doc_id,
        section=section,
        author=author,
        year_min=year_min,
        year_max=year_max,
        min_score=min_score,
        chunk_type=chunk_type,
    )
    if json_output:
        _output(chunks, True)
        return
    if not chunks:
        typer.echo("No results found.")
        return
    for c in chunks:
        score_str = f"[{c.score:.4f}] " if c.score else ""
        typer.echo(f"{score_str}{c.section_path} (chunk {c.chunk_index})\n  {c.content[:200]}...\n")


@app.command(name="list-documents")
def list_documents(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List all documents in the corpus."""
    from lean.services.corpus import list_documents as _list

    docs = _list()
    if json_output:
        _output(docs, True)
        return
    for d in docs:
        auth = f"  {', '.join(d.authors)}" if d.authors else ""
        typer.echo(f"{d.id}  {d.title or '(untitled)'}  [{d.chunk_count} chunks]{auth}")


@app.command(name="corpus-stats")
def corpus_stats(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show corpus statistics."""
    from lean.services.corpus import corpus_stats as _stats

    stats = _stats()
    _output(stats, json_output)


@app.command(name="get-chunk")
def get_chunk(
    chunk_id: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Retrieve a single chunk by ID."""
    from lean.services.corpus import get_chunk as _get

    chunk = _get(chunk_id)
    if chunk is None:
        typer.echo("Chunk not found.", err=True)
        raise typer.Exit(1)
    _output(chunk, json_output)


@app.command(name="get-markdown")
def get_markdown(
    doc_id: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Get the extracted markdown for a document."""
    from lean.services.corpus import get_document_markdown as _md

    md = _md(doc_id)
    if json_output:
        typer.echo(json.dumps({"doc_id": doc_id, "markdown": md}, indent=2))
    else:
        typer.echo(md)


@app.command()
def delete(
    doc_id: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Delete a document and all its chunks."""
    from lean.services.corpus import delete_document as _del

    result = _del(doc_id)
    if json_output:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"Deleted: {result['deleted']}")


@app.command()
def reingest(
    doc_id: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Re-ingest a document (re-extract with current settings)."""
    from lean.services.ingestion import reingest as _re

    result = asyncio.run(_re(doc_id))
    _output(result, json_output)


@app.command(name="reingest-all")
def reingest_all(
    force: bool = typer.Option(False, "--force", help="Reingest even if already OCR'd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Reingest all documents (smallest first, skips OCR'd unless --force)."""
    from lean.models.schemas import ExtractionMethod
    from lean.services.corpus import list_documents
    from lean.services.ingestion import reingest

    docs = sorted(list_documents(), key=lambda d: d.chunk_count)
    results: list[dict[str, object]] = []
    success = failed = skipped = 0

    for doc in docs:
        if not force and doc.extraction_method == ExtractionMethod.UNLIMITED_OCR:
            skipped += 1
            results.append({"id": doc.id, "status": "skipped"})
            typer.echo(f"SKIP  {doc.id}  (already OCR'd, {doc.chunk_count} chunks)")
            continue
        typer.echo(f"START {doc.id}  ({doc.page_count or '?'} pages, was {doc.extraction_method})")
        try:
            res = asyncio.run(reingest(doc.id))
            success += 1
            results.append(
                {
                    "id": doc.id,
                    "status": "ok",
                    "chunks": res.chunk_count,
                    "seconds": res.elapsed_seconds,
                }
            )
            typer.echo(f"DONE  {doc.id}  ({res.chunk_count} chunks, {res.elapsed_seconds:.1f}s)")
        except Exception as exc:
            failed += 1
            results.append({"id": doc.id, "status": "failed", "error": str(exc)})
            typer.echo(f"FAIL  {doc.id}  ({exc})")

    if json_output:
        typer.echo(
            json.dumps(
                {"success": success, "failed": failed, "skipped": skipped, "details": results},
                indent=2,
                default=str,
            )
        )
    else:
        typer.echo(f"\nSuccess: {success}  Failed: {failed}  Skipped: {skipped}")


@app.command()
def health(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Check health of OCR server, database, and Ollama embedding server."""
    from lean.config.settings import get_settings

    settings = get_settings()
    checks: dict[str, dict[str, object]] = {
        "ocr": _check_ocr(settings),
        "database": _check_database(),
        "ollama": _check_ollama(settings),
    }

    if json_output:
        typer.echo(json.dumps(checks, indent=2))
    else:
        for name, result in checks.items():
            status = result["status"]
            label = "[OK]" if status == "ok" else "[--]" if status == "not_configured" else "[FAIL]"
            typer.echo(f"{label} {name}: {status}")

    if any(r["status"] == "error" for r in checks.values()):
        raise typer.Exit(1)


def _check_ocr(settings: Settings) -> dict[str, object]:
    """Probe the OCR server's /health endpoint. Returns ``{status, url?, error?}``."""
    if not settings.ocr_base_url:
        return {"status": "not_configured"}
    import httpx

    try:
        resp = httpx.get(
            f"{settings.ocr_base_url.rstrip('/')}/health",
            timeout=settings.health_http_timeout,
        )
        return {
            "status": "ok" if resp.status_code == 200 else "error",
            "url": settings.ocr_base_url,
        }
    except Exception as exc:
        return {
            "status": "error",
            "url": settings.ocr_base_url,
            "error": str(exc),
        }


def _check_database() -> dict[str, object]:
    """Verify Postgres connectivity + pgvector extension presence."""
    try:
        from lean.store.base import StoreConnection

        conn = StoreConnection.from_env()
        try:
            with conn.conn.cursor() as cur:
                cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                has_pgvector = cur.fetchone() is not None
        finally:
            conn.close()
        return {"status": "ok", "pgvector": has_pgvector}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _check_ollama(settings: Settings) -> dict[str, object]:
    """Probe the Ollama /api/tags endpoint. Returns ``{status, url?, error?}``."""
    if not settings.embedding_remote_url:
        return {"status": "not_configured"}
    import httpx

    try:
        resp = httpx.get(
            f"{settings.embedding_remote_url.rstrip('/')}/api/tags",
            timeout=settings.health_http_timeout,
        )
        return {
            "status": "ok" if resp.status_code == 200 else "error",
            "url": settings.embedding_remote_url,
        }
    except Exception as exc:
        return {
            "status": "error",
            "url": settings.embedding_remote_url,
            "error": str(exc),
        }


@app.command(name="mcp-serve")
def mcp_serve(
    transport: str = typer.Option("stdio", help="stdio or http"),
    port: int = typer.Option(None, help="HTTP port (default: from settings)"),
) -> None:
    """Run the MCP server."""
    from lean.config.settings import get_settings

    settings = get_settings()
    if port is None:
        port = settings.mcp_http_port
    from lean.mcp_server.__main__ import main

    main(["--transport", transport, "--port", str(port)])


@app.command(name="api-serve")
def api_serve(
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
) -> None:
    """Start the FastAPI REST API server (bearer-authed)."""
    import uvicorn

    from lean.config.settings import get_settings

    settings = get_settings()
    uvicorn.run(
        "lean.api.routes:app",
        host=settings.mcp_http_host,
        port=settings.api_port,
        reload=reload,
    )


@app.command(name="db-init")
def db_init() -> None:
    """Apply db/schemas/*.sql to local Supabase."""
    import os
    from urllib.parse import unquote, urlparse

    from lean.config.settings import get_settings

    db_url = get_settings().supabase_db_url
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


def _print_eval_result(result: EvalResult) -> None:
    """Print EvalResult metrics in the human-readable eval format."""
    typer.echo(f"\nResults (n={result.sample_count}, k={result.k}):")
    typer.echo(f"  hit_rate:     {result.hit_rate:.4f}")
    typer.echo(f"  MRR:          {result.mrr:.4f}")
    typer.echo(f"  NDCG:         {result.ndcg:.4f}")
    typer.echo(f"  Recall:       {result.recall:.4f}")
    typer.echo(f"  mean_latency: {result.mean_latency_ms:.0f}ms")


def _save_eval_run(conn: StoreConnection, result: EvalResult, config: dict[str, object]) -> None:
    """Persist an EvalResult to the ``eval_runs`` table and confirm to stdout."""
    from lean.store.analytics import AnalyticsRepo

    AnalyticsRepo(conn).save_eval_run(
        config=config,
        hit_rate=result.hit_rate,
        mrr=result.mrr,
        ndcg=result.ndcg,
        recall=result.recall,
        mean_latency_ms=int(result.mean_latency_ms),
        sample_count=result.sample_count,
        k=result.k,
    )
    typer.echo("\nEval run saved to eval_runs table.")


@app.command()
def eval(
    sample_size: int | None = typer.Option(None, help="Number of chunks to sample for eval"),
    k: int | None = typer.Option(None, help="Top-k for hit_rate/MRR"),
    dataset: Path | None = typer.Option(
        None,
        "--dataset",
        help="Path to a curated eval dataset JSON; overrides --sample-size.",
    ),
) -> None:
    """Run retrieval evaluation (hit_rate@k, MRR@k, NDCG@k, Recall@k).

    By default, samples chunks from the corpus and uses their heading +
    content preview as pseudo-queries (measures self-similarity, not real
    queries). With ``--dataset <path>``, loads a curated JSON list of
    ``{"query": str, "expected_chunk_id": str}`` entries instead. See
    ``data/curated_eval_dataset.json`` and ``scripts/build_eval_dataset.py``.
    """
    from lean.config.settings import get_settings
    from lean.eval.runner import build_eval_dataset, evaluate, load_curated_dataset
    from lean.store.base import StoreConnection

    settings = get_settings()
    if k is None:
        k = settings.eval_k

    conn = StoreConnection(settings.supabase_db_url)
    try:
        if dataset is not None:
            typer.echo(f"Loading curated dataset from {dataset}...")
            samples = load_curated_dataset(dataset)
            typer.echo(f"Loaded {len(samples)} curated samples.")
            config: dict[str, object] = {"k": k, "dataset": str(dataset)}
            sample_size = len(samples)
        else:
            if sample_size is None:
                sample_size = settings.eval_sample_size
            typer.echo(f"Building eval dataset ({sample_size} samples)...")
            samples = build_eval_dataset(conn, sample_size=sample_size, seed=settings.eval_seed)
            config = {"sample_size": sample_size, "k": k}

        typer.echo(f"Running evaluation (k={k})...")
        result = evaluate(conn, samples, k=k)

        _print_eval_result(result)
        _save_eval_run(conn, result, config)
    finally:
        conn.close()


if __name__ == "__main__":
    app()
