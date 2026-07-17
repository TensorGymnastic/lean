"""Typer CLI for lean: full parity with MCP tools via direct service calls."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import typer

app = typer.Typer(
    name="lean",
    help="Lean Six Sigma corpus MCP server and CLI.",
    no_args_is_help=True,
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
    import httpx

    from lean.config.settings import get_settings

    settings = get_settings()
    checks: dict[str, dict[str, object]] = {}

    if settings.ocr_base_url:
        try:
            resp = httpx.get(f"{settings.ocr_base_url.rstrip('/')}/health", timeout=10)
            checks["ocr"] = {
                "status": "ok" if resp.status_code == 200 else "error",
                "url": settings.ocr_base_url,
            }
        except Exception as exc:
            checks["ocr"] = {"status": "error", "url": settings.ocr_base_url, "error": str(exc)}
    else:
        checks["ocr"] = {"status": "not_configured"}

    try:
        from lean.store.base import StoreConnection

        conn = StoreConnection.from_env()
        with conn.conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            has_pgvector = cur.fetchone() is not None
        conn.close()
        checks["database"] = {"status": "ok", "pgvector": has_pgvector}
    except Exception as exc:
        checks["database"] = {"status": "error", "error": str(exc)}

    if settings.embedding_remote_url:
        try:
            resp = httpx.get(f"{settings.embedding_remote_url.rstrip('/')}/api/tags", timeout=10)
            checks["ollama"] = {
                "status": "ok" if resp.status_code == 200 else "error",
                "url": settings.embedding_remote_url,
            }
        except Exception as exc:
            checks["ollama"] = {
                "status": "error",
                "url": settings.embedding_remote_url,
                "error": str(exc),
            }
    else:
        checks["ollama"] = {"status": "not_configured"}

    if json_output:
        typer.echo(json.dumps(checks, indent=2))
    else:
        for name, result in checks.items():
            status = result["status"]
            label = "[OK]" if status == "ok" else "[--]" if status == "not_configured" else "[FAIL]"
            typer.echo(f"{label} {name}: {status}")


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


@app.command(name="db-init")
def db_init() -> None:
    """Apply db/schemas/*.sql to local Supabase."""
    from lean.config.settings import get_settings

    db_url = get_settings().supabase_db_url
    for sql_file in sorted(Path("db/schemas").glob("*.sql")):
        typer.echo(f"applying {sql_file}...")
        subprocess.run(["psql", db_url, "-f", str(sql_file)], check=True)
    typer.echo("schema applied.")


@app.command()
def eval(
    sample_size: int = typer.Option(50, help="Number of chunks to sample for eval"),
    k: int = typer.Option(5, help="Top-k for hit_rate/MRR"),
) -> None:
    """Run retrieval evaluation (hit_rate@k, MRR@k, NDCG@k, Recall@k)."""
    from lean.config.settings import get_settings
    from lean.eval.runner import build_eval_dataset, evaluate
    from lean.store.analytics import AnalyticsRepo
    from lean.store.base import StoreConnection

    conn = StoreConnection(get_settings().supabase_db_url)
    try:
        typer.echo(f"Building eval dataset ({sample_size} samples)...")
        samples = build_eval_dataset(conn, sample_size=sample_size)
        typer.echo(f"Running evaluation (k={k})...")
        result = evaluate(conn, samples, k=k)

        typer.echo(f"\nResults (n={result.sample_count}, k={result.k}):")
        typer.echo(f"  hit_rate:     {result.hit_rate:.4f}")
        typer.echo(f"  MRR:          {result.mrr:.4f}")
        typer.echo(f"  NDCG:         {result.ndcg:.4f}")
        typer.echo(f"  Recall:       {result.recall:.4f}")
        typer.echo(f"  mean_latency: {result.mean_latency_ms:.0f}ms")

        AnalyticsRepo(conn).save_eval_run(
            config={"sample_size": sample_size, "k": k},
            hit_rate=result.hit_rate,
            mrr=result.mrr,
            ndcg=result.ndcg,
            recall=result.recall,
            mean_latency_ms=int(result.mean_latency_ms),
            sample_count=result.sample_count,
            k=result.k,
        )
        typer.echo("\nEval run saved to eval_runs table.")
    finally:
        conn.close()


if __name__ == "__main__":
    app()
