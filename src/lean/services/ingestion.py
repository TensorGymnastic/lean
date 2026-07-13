"""Ingestion service: PDF → markdown → chunks → embeddings → pgvector.

Full ingestion pipeline plus re-ingest by document id. Uses the focused
store modules (``DocumentRepo``, ``ChunkRepo``) directly.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from uuid import UUID

import anyio

from lean.chunker.markdown_ast import build_sections
from lean.chunker.recursive import chunk_sections
from lean.config.settings import Settings
from lean.extraction.metadata import extract_metadata
from lean.extraction.pipeline import extract_pdf_markdown
from lean.infrastructure.embedder import get_embedder
from lean.models.schemas import ExtractionMethod, IngestResult
from lean.store.base import StoreConnection
from lean.store.chunks import ChunkRepo, ChunkRow
from lean.store.documents import DocumentRepo

logger = logging.getLogger(__name__)


async def ingest_pdf(path: str) -> IngestResult:
    """Full pipeline: PDF → extract → metadata → chunk → embed → store.

    Steps:
        1. SHA-256 the source PDF (dedup key).
        2. Extract markdown via Unlimited-OCR (vLLM) or markitdown fallback.
        3. Build section AST and split into token-bounded chunks.
        4. Embed all chunks with the singleton LiquidLMF embedder.
        5. Extract PDF metadata (title, authors, publisher, year).
        6. Upsert document and replace its chunks in a single connection.

    Re-ingesting the same PDF (matched by SHA-256) updates the existing
    document row and replaces its chunks.
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
            ocr_model=settings.ocr_model,
            ocr_dpi=settings.ocr_dpi,
            ocr_timeout_s=settings.ocr_timeout_s,
            ocr_max_tokens=settings.ocr_max_tokens,
        )
    )

    warnings: list[str] = []
    if method == ExtractionMethod.MARKITDOWN:
        warnings.append("vLLM unavailable, fell back to markitdown")

    sections = await anyio.to_thread.run_sync(
        lambda: build_sections(markdown, max_heading_level=settings.max_section_heading_level)
    )
    chunk_results = await anyio.to_thread.run_sync(
        lambda: chunk_sections(
            sections,
            target_min=settings.chunk_target_min,
            target_max=settings.chunk_target_max,
            hard_cap=settings.chunk_hard_cap,
            encoding=settings.token_counter_encoding,
        )
    )

    embedder = get_embedder()
    chunk_texts = [c.content for c in chunk_results]
    embeddings = await anyio.to_thread.run_sync(lambda: embedder.embed_documents(chunk_texts))

    source_storage_path = f"sources/{source_sha256}.pdf"
    markdown_storage_path = f"markdown/{source_sha256}.md"

    pdf_meta = await anyio.to_thread.run_sync(lambda: extract_metadata(pdf_path))

    conn = StoreConnection.from_env()
    try:
        documents = DocumentRepo(conn)
        chunks_repo = ChunkRepo(conn)
        doc_id = documents.upsert_document(
            source_path=str(pdf_path),
            source_sha256=source_sha256,
            title=pdf_meta.title or pdf_path.stem,
            extraction_method=method.value,
            source_storage_path=source_storage_path,
            markdown_storage_path=markdown_storage_path,
            authors=pdf_meta.authors,
            publisher=pdf_meta.publisher,
            year=pdf_meta.year,
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
        chunks_repo.replace_chunks(doc_id, chunk_rows)
    finally:
        conn.close()

    elapsed = time.monotonic() - start
    logger.info(
        "ingested path=%s sha256=%s chunks=%d pages=%d method=%s elapsed=%.2fs",
        path,
        source_sha256[:12],
        len(chunk_rows),
        page_count,
        method.value,
        elapsed,
    )
    return IngestResult(
        document_id=str(doc_id),
        source_sha256=source_sha256,
        page_count=page_count,
        extraction_method=method,
        chunk_count=len(chunk_rows),
        elapsed_seconds=elapsed,
        warnings=warnings,
    )


async def reingest(document_id: str) -> IngestResult:
    """Re-ingest by looking up source_path from DB and calling ingest_pdf.

    Raises ``KeyError`` if the document id is not found.
    """
    doc_uuid = UUID(document_id)
    conn = StoreConnection.from_env()
    try:
        with conn.conn.cursor() as cur:
            cur.execute(
                "select source_path from public.documents where id = %s",
                (doc_uuid,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise KeyError(f"document {document_id} not found")
    source_path = str(row[0])
    return await ingest_pdf(source_path)
