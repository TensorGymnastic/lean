"""MCP tools for the lean corpus.

Eight tools exposed via fastmcp:
  ingest_pdf, search, get_chunk, list_documents, get_document_markdown,
  delete_document, reingest, corpus_stats.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from uuid import UUID

import anyio
from fastmcp import FastMCP

from lean.chunker.markdown_ast import build_sections
from lean.chunker.recursive import chunk_sections
from lean.embeddings.liquid_lmf import LiquidLMFEmbedder
from lean.extraction.pipeline import extract_pdf_markdown
from lean.models.schemas import Chunk, CorpusStats, DocumentSummary, IngestResult
from lean.settings import Settings
from lean.store.pgvector import ChunkRow, PgVectorStore

logger = logging.getLogger(__name__)

mcp = FastMCP("lean")

_embedder: LiquidLMFEmbedder | None = None


def _get_embedder() -> LiquidLMFEmbedder:
    global _embedder
    if _embedder is None:
        settings = Settings()
        _embedder = LiquidLMFEmbedder(
            model=settings.embedding_model,
            hf_token=settings.hf_token,
        )
    return _embedder


@mcp.tool
async def ingest_pdf(path: str) -> IngestResult:
    """Ingest a PDF into the lean corpus.

    Runs extraction (Unlimited-OCR via vLLM, or markitdown fallback) →
    markdown → section-aware chunks → LFM2.5 embeddings → pgvector upsert.
    Re-ingesting the same PDF (by SHA-256) updates the existing document.
    """
    start = time.monotonic()
    settings = Settings()
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    source_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    markdown, page_count, method = await anyio.to_thread.run_sync(
        lambda: extract_pdf_markdown(
            pdf_path,
            vllm_base_url=settings.vllm_base_url,
            hf_token=settings.hf_token,
        )
    )

    warnings: list[str] = []
    if method.value == "markitdown":
        warnings.append("vLLM unavailable, fell back to markitdown")

    sections = await anyio.to_thread.run_sync(lambda: build_sections(markdown))
    chunk_results = await anyio.to_thread.run_sync(
        lambda: chunk_sections(
            sections,
            target_min=settings.chunk_target_min,
            target_max=settings.chunk_target_max,
            hard_cap=settings.chunk_hard_cap,
        )
    )

    embedder = _get_embedder()
    chunk_texts = [c.content for c in chunk_results]
    embeddings = await anyio.to_thread.run_sync(lambda: embedder.embed_documents(chunk_texts))

    source_storage_path = f"sources/{source_sha256}.pdf"
    markdown_storage_path = f"markdown/{source_sha256}.md"

    store = PgVectorStore.from_env()
    try:
        doc_id = store.upsert_document(
            source_path=str(pdf_path),
            source_sha256=source_sha256,
            title=pdf_path.stem,
            extraction_method=method.value,
            source_storage_path=source_storage_path,
            markdown_storage_path=markdown_storage_path,
            page_count=page_count,
        )
        chunk_rows = [
            ChunkRow(
                document_id=doc_id,
                chunk_index=global_idx,
                section_path=c.section_path,
                heading_text=c.heading_text,
                page_start=None,
                page_end=None,
                token_count=c.token_count,
                content=c.content,
                embedding=emb,
            )
            for global_idx, (c, emb) in enumerate(zip(chunk_results, embeddings, strict=True))
        ]
        store.replace_chunks(doc_id, chunk_rows)
    finally:
        store.close()

    elapsed = time.monotonic() - start
    return IngestResult(
        document_id=str(doc_id),
        source_sha256=source_sha256,
        page_count=page_count,
        extraction_method=method,
        chunk_count=len(chunk_rows),
        elapsed_seconds=elapsed,
        warnings=warnings,
    )


@mcp.tool
async def search(
    query: str,
    k: int = 5,
    doc_id: str | None = None,
    section: str | None = None,
) -> list[Chunk]:
    """Semantic search over the Lean Six Sigma corpus.

    Returns top-k chunks by cosine similarity. Filter by doc_id or
    section substring (case-insensitive) to narrow results.
    """
    from lean.retrieval.search import search as _search_impl

    return await anyio.to_thread.run_sync(
        lambda: _search_impl(query, k=k, doc_id=doc_id, section=section)
    )


@mcp.tool
async def get_chunk(chunk_id: str) -> Chunk:
    """Fetch a specific chunk by its UUID."""
    store = PgVectorStore.from_env()
    try:
        chunk = store.get_chunk(UUID(chunk_id))
    finally:
        store.close()
    if chunk is None:
        raise KeyError(f"chunk {chunk_id} not found")
    return chunk


@mcp.tool
async def list_documents() -> list[DocumentSummary]:
    """List all documents in the corpus, newest first."""
    from psycopg.rows import dict_row

    store = PgVectorStore.from_env()
    try:
        with store._conn.cursor(row_factory=dict_row) as cur:  # noqa: SLF001
            cur.execute(
                """
                select d.id, d.source_path, d.title, d.authors, d.page_count,
                       d.extraction_method, d.ingested_at,
                       count(c.id) as chunk_count
                from public.documents d
                left join public.chunks c on c.document_id = d.id
                group by d.id
                order by d.ingested_at desc
                """
            )
            rows = cur.fetchall()
    finally:
        store.close()

    return [
        DocumentSummary(
            id=str(r["id"]),
            source_path=r["source_path"],
            title=r["title"],
            authors=r["authors"] or [],
            page_count=r["page_count"],
            extraction_method=r["extraction_method"],
            chunk_count=r["chunk_count"],
            ingested_at=r["ingested_at"],
        )
        for r in rows
    ]


@mcp.tool
async def get_document_markdown(doc_id: str) -> str:
    """Fetch the full extracted markdown for a document.

    Note: v1 returns the storage path. Full Supabase Storage fetch
    will be wired in a follow-up task.
    """
    from psycopg.rows import dict_row

    store = PgVectorStore.from_env()
    try:
        with store._conn.cursor(row_factory=dict_row) as cur:  # noqa: SLF001
            cur.execute(
                "select markdown_storage_path from public.documents where id = %s",
                (UUID(doc_id),),
            )
            r = cur.fetchone()
    finally:
        store.close()
    if r is None:
        raise KeyError(f"document {doc_id} not found")
    return f"[markdown at storage path {r['markdown_storage_path']}]"


@mcp.tool
async def delete_document(doc_id: str) -> bool:
    """Delete a document and all its chunks."""
    store = PgVectorStore.from_env()
    try:
        store.delete_document(UUID(doc_id))
    finally:
        store.close()
    return True


@mcp.tool
async def reingest(doc_id: str) -> IngestResult:
    """Re-run the ingest pipeline for an existing document."""
    from psycopg.rows import dict_row

    store = PgVectorStore.from_env()
    try:
        with store._conn.cursor(row_factory=dict_row) as cur:  # noqa: SLF001
            cur.execute(
                "select source_path from public.documents where id = %s",
                (UUID(doc_id),),
            )
            r = cur.fetchone()
    finally:
        store.close()
    if r is None:
        raise KeyError(f"document {doc_id} not found")
    source_path = r["source_path"]
    return await ingest_pdf(source_path)


@mcp.tool
async def corpus_stats() -> CorpusStats:
    """Return corpus statistics: document/chunk counts, extraction breakdown."""
    from psycopg.rows import dict_row

    store = PgVectorStore.from_env()
    try:
        with store._conn.cursor(row_factory=dict_row) as cur:  # noqa: SLF001
            cur.execute("select count(*) from public.documents")
            row = cur.fetchone()
            assert row is not None
            doc_count: int = row["count"]
            cur.execute("select count(*) from public.chunks")
            row = cur.fetchone()
            assert row is not None
            chunk_count: int = row["count"]
            cur.execute("select coalesce(sum(token_count), 0) as total from public.chunks")
            row = cur.fetchone()
            assert row is not None
            total_tokens: int = row["total"]
            cur.execute(
                "select extraction_method, count(*) as cnt "
                "from public.documents group by extraction_method"
            )
            breakdown = {row["extraction_method"]: row["cnt"] for row in cur.fetchall()}
            cur.execute("select max(ingested_at) as last from public.documents")
            last_row = cur.fetchone()
            last_ingested = last_row["last"] if last_row else None
    finally:
        store.close()

    settings = Settings()
    return CorpusStats(
        document_count=doc_count,
        chunk_count=chunk_count,
        total_tokens=total_tokens,
        extraction_method_breakdown=breakdown,
        embedding_dim=settings.embedding_dim,
        embedding_model=settings.embedding_model,
        last_ingested_at=last_ingested,
    )
