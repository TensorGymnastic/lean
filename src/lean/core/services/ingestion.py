"""Ingestion service: PDF → markdown → chunks → embeddings → pgvector.

Universal orchestration. Domain code provides the extraction pipeline
via ``lean.core.extraction.set_pipeline()`` at startup.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import anyio

from lean.core.chunker.markdown_ast import Section, build_sections
from lean.core.chunker.recursive import ChunkResult, chunk_sections
from lean.core.config.settings import CoreSettings, get_settings
from lean.core.extraction import (
    BlockMeta,
    ExtractionMethod,
    ExtractionResult,
    PdfMetadata,
    extract_metadata,
    get_pipeline,
)
from lean.core.extraction.pipeline_helpers import (
    build_chunk_rows,
    describe_images,
    enrich_chunks_with_block_meta,
)
from lean.core.infrastructure.embedder import get_embedder
from lean.core.models import IngestResult
from lean.core.store.base import StoreConnection
from lean.core.store.chunks import ChunkRepo
from lean.core.store.documents import DocumentRepo

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_SHA256_CHUNK_SIZE = 1 << 20


def _sha256_streaming(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_SHA256_CHUNK_SIZE), b""):
            h.update(block)
    return h.hexdigest()


async def _validate_pdf_path(path: str, settings: CoreSettings) -> tuple[Path, str]:
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    corpus_root = Path(settings.corpus_root).resolve()
    resolved = pdf_path.resolve()
    if not resolved.is_relative_to(corpus_root):
        raise PermissionError(f"path outside corpus root: {path}")

    max_bytes = settings.max_pdf_mb * 1024 * 1024
    file_size = pdf_path.stat().st_size
    if file_size > max_bytes:
        raise ValueError(
            f"PDF too large: {file_size / (1024 * 1024):.1f} MB "
            f"(max {settings.max_pdf_mb} MB): {path}"
        )

    return resolved, _sha256_streaming(pdf_path)


async def _run_extraction(pdf_path: Path) -> ExtractionResult:
    """Run the registered extraction pipeline."""
    return await anyio.to_thread.run_sync(lambda: get_pipeline().extract(pdf_path))


async def _chunk_markdown(
    markdown: str,
    method: ExtractionMethod,
    block_metas: list[BlockMeta],
    settings: CoreSettings,
) -> tuple[list[Section], list[ChunkResult], list[str]]:
    """Build section AST, chunk, attach provenance metadata."""
    warnings: list[str] = []
    if method == ExtractionMethod.MARKITDOWN:
        warnings.append("OCR server unavailable, fell back to markitdown")

    sections = await anyio.to_thread.run_sync(
        lambda: build_sections(markdown, max_heading_level=settings.max_section_heading_level)
    )
    chunk_results = await anyio.to_thread.run_sync(
        lambda: chunk_sections(
            sections,
            target_max=settings.chunk_target_max,
            hard_cap=settings.chunk_hard_cap,
            overlap=settings.chunk_overlap,
            encoding=settings.token_counter_encoding,
        )
    )
    enrich_chunks_with_block_meta(
        chunk_results, block_metas, min_overlap=settings.block_match_min_overlap
    )
    return sections, chunk_results, warnings


async def _maybe_contextualize(
    chunks: list[ChunkResult],
    sections: list[Section],
    settings: CoreSettings,
    warnings: list[str],
) -> tuple[list[ChunkResult], list[str]]:
    """Apply Contextual Retrieval only when enabled and an LLM sidecar is configured."""
    if not settings.llm_contextual_retrieval:
        return chunks, warnings

    from lean.core.extraction.contextual import add_context_to_chunks
    from lean.core.llm.base import get_llm

    llm = get_llm()
    if llm is None:
        warnings.append("contextual_retrieval enabled but no LLM configured")
        return chunks, warnings

    new_chunks = await anyio.to_thread.run_sync(
        lambda: add_context_to_chunks(
            chunks,
            sections,
            llm,
            max_tokens=settings.llm_generate_max_tokens,
            temperature=settings.llm_generate_temperature,
        )
    )
    return new_chunks, warnings


async def _embed_chunks(
    chunks: list[ChunkResult],
    image_descriptions: list[tuple[str, dict[str, object], str]],
) -> list[list[float]]:
    embedder = get_embedder()
    texts = [c.content for c in chunks]
    texts.extend(desc for desc, _, _ in image_descriptions)
    return await anyio.to_thread.run_sync(lambda: embedder.embed_documents(texts))


async def _persist_ingest(
    pdf_path: Path,
    source_sha256: str,
    pdf_meta: PdfMetadata,
    page_count: int,
    method: ExtractionMethod,
    chunk_results: list[ChunkResult],
    embeddings: list[list[float]],
    image_descriptions: list[tuple[str, dict[str, object], str]],
    settings: CoreSettings,
) -> tuple[UUID, int]:
    conn = StoreConnection.from_env()
    try:
        documents = DocumentRepo(conn)
        chunks_repo = ChunkRepo(conn)
        doc_id = documents.upsert_document(
            source_path=str(pdf_path),
            source_sha256=source_sha256,
            title=pdf_meta.title or pdf_path.stem,
            extraction_method=method.value,
            authors=pdf_meta.authors,
            year=pdf_meta.year,
            page_count=page_count,
            metadata={
                "keywords": pdf_meta.keywords,
                "subject": pdf_meta.subject,
                "toc": pdf_meta.toc,
            },
            commit=False,
        )
        chunk_rows = build_chunk_rows(
            chunk_results, embeddings, image_descriptions, settings, doc_id
        )
        try:
            chunks_repo.replace_chunks(doc_id, chunk_rows, commit=False)
        except Exception:
            conn.conn.rollback()
            raise
        conn.conn.commit()
        return doc_id, len(chunk_rows)
    finally:
        conn.close()


async def ingest_pdf(path: str) -> IngestResult:
    """PDF → extract → chunk → optional context → optional VLM → embed → persist.

    Pipeline:
    1. Validate path + corpus-root + size + SHA-256.
    2. Run the registered extraction pipeline.
    3. Build sections + chunks, attach block-metas for provenance.
    4. Optional contextual retrieval (LLM-side).
    5. Optional image enrichment (VLM-side; configurable by domain).
    6. Embed text + image descriptions.
    7. Persist document + chunks in a single transaction.
    """
    start = time.monotonic()
    settings = get_settings()
    pdf_path, source_sha256 = await _validate_pdf_path(path, settings)
    result = await _run_extraction(pdf_path)
    sections, chunk_results, warnings = await _chunk_markdown(
        result.markdown, result.method, result.block_metas, settings
    )
    chunk_results, warnings = await _maybe_contextualize(
        chunk_results, sections, settings, warnings
    )
    image_descriptions, image_warnings = await describe_images(result.images, settings)
    warnings.extend(image_warnings)
    embeddings = await _embed_chunks(chunk_results, image_descriptions)
    pdf_meta = await anyio.to_thread.run_sync(lambda: extract_metadata(pdf_path))
    doc_id, chunk_count = await _persist_ingest(
        pdf_path,
        source_sha256,
        pdf_meta,
        result.page_count,
        result.method,
        chunk_results,
        embeddings,
        image_descriptions,
        settings,
    )
    elapsed = time.monotonic() - start
    logger.info(
        "ingested path=%s sha256=%s chunks=%d pages=%d method=%s elapsed=%.2fs",
        path,
        source_sha256[:12],
        chunk_count,
        result.page_count,
        result.method.value,
        elapsed,
    )
    return IngestResult(
        document_id=str(doc_id),
        source_sha256=source_sha256,
        page_count=result.page_count,
        extraction_method=result.method,
        chunk_count=chunk_count,
        elapsed_seconds=elapsed,
        warnings=warnings,
    )


async def reingest(document_id: str) -> IngestResult:
    """Re-ingest by looking up source_path from DB and calling ingest_pdf."""
    doc_uuid = UUID(document_id)
    conn = StoreConnection.from_env()
    try:
        source_path = DocumentRepo(conn).get_source_path(doc_uuid)
    finally:
        conn.close()
    if source_path is None:
        raise KeyError(f"document {document_id} not found")
    return await ingest_pdf(source_path)


__all__ = ["ingest_pdf", "reingest"]
