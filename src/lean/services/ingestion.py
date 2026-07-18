"""Ingestion service: PDF → markdown → chunks → embeddings → pgvector.

Full ingestion pipeline plus re-ingest by document id. Uses the focused
store modules (``DocumentRepo``, ``ChunkRepo``) directly.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import anyio

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

from lean.chunker.markdown_ast import build_sections
from lean.chunker.recursive import chunk_sections
from lean.config.settings import get_settings
from lean.extraction.metadata import extract_metadata
from lean.extraction.pipeline import extract_pdf_markdown
from lean.infrastructure.embedder import get_embedder
from lean.models.schemas import ExtractionMethod, IngestResult
from lean.store.base import StoreConnection
from lean.store.chunks import ChunkRepo, ChunkRow
from lean.store.documents import DocumentRepo

logger = logging.getLogger(__name__)

_SHA256_CHUNK_SIZE = 1 << 20  # 1 MiB


def _sha256_streaming(path: Path) -> str:
    """Stream SHA-256 in 1 MiB blocks — avoids loading whole PDF into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_SHA256_CHUNK_SIZE), b""):
            h.update(block)
    return h.hexdigest()


async def ingest_pdf(path: str) -> IngestResult:
    """Full pipeline: PDF → extract → metadata → chunk → embed → store.

    Steps:
        1. Validate path is within the configured corpus root (security).
        2. SHA-256 the source PDF (dedup key).
        3. Extract markdown via Unlimited-OCR (remote server) or markitdown fallback.
        4. Build section AST and split into token-bounded chunks.
        5. Embed all chunks with the singleton LiquidLMF embedder.
        6. Extract PDF metadata (title, authors, publisher, year).
        7. Upsert document and replace its chunks in a single connection.

    Re-ingesting the same PDF (matched by SHA-256) updates the existing
    document row and replaces its chunks.

    Raises ``PermissionError`` if the path resolves outside the corpus root.
    """
    start = time.monotonic()
    settings = get_settings()
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    # Security: confine to corpus root to prevent path traversal
    corpus_root = Path(settings.corpus_root).resolve()
    resolved = pdf_path.resolve()
    if not resolved.is_relative_to(corpus_root):
        raise PermissionError(f"path outside corpus root: {path}")

    # Resource exhaustion guard: reject oversized PDFs before reading into memory
    max_bytes = settings.max_pdf_mb * 1024 * 1024
    file_size = pdf_path.stat().st_size
    if file_size > max_bytes:
        raise ValueError(
            f"PDF too large: {file_size / (1024 * 1024):.1f} MB "
            f"(max {settings.max_pdf_mb} MB): {path}"
        )

    source_sha256 = _sha256_streaming(pdf_path)

    markdown, page_count, method, images = await anyio.to_thread.run_sync(
        lambda: extract_pdf_markdown(
            pdf_path,
            ocr_base_url=settings.ocr_base_url,
            hf_token=settings.hf_token,
            ocr_model=settings.ocr_model,
            ocr_dpi=settings.ocr_dpi,
            ocr_timeout_s=settings.ocr_timeout_s,
            ocr_max_tokens=settings.ocr_max_tokens,
            ocr_batch_size=settings.ocr_batch_size,
            marker_force_ocr=settings.marker_force_ocr,
            marker_remote_url=settings.marker_remote_url,
        )
    )

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

    if settings.llm_contextual_retrieval:
        from lean.extraction.contextual import add_context_to_chunks
        from lean.llm.base import get_llm

        llm = get_llm()
        if llm:
            chunk_results = await anyio.to_thread.run_sync(
                lambda: add_context_to_chunks(
                    chunk_results,
                    sections,
                    llm,
                    max_tokens=settings.llm_generate_max_tokens,
                    temperature=settings.llm_generate_temperature,
                )
            )
        else:
            warnings.append("contextual_retrieval enabled but no LLM configured")

    embedder = get_embedder()

    image_descriptions: list[tuple[str, dict[str, object], str]] = []
    if settings.vlm_enabled and images:
        from lean.vlm.client import VLMClient, VLMError
        from lean.vlm.prompts import CHART_EXTRACTION_PROMPT, parse_description

        vlm = VLMClient(
            base_url=settings.vlm_base_url,
            model=settings.vlm_model,
            api_key=settings.vlm_api_key,
            timeout=settings.vlm_timeout_s,
            detail=settings.vlm_detail,
            disable_thinking=settings.vlm_disable_thinking,
        )
        try:
            results: dict[str, tuple[str, dict[str, object], str] | Exception] = {}
            sem = anyio.Semaphore(settings.vlm_max_concurrency)

            async def _describe_one(img_name: str, pil_img: PILImage) -> None:
                async with sem:
                    try:
                        raw = await anyio.to_thread.run_sync(
                            partial(
                                vlm.describe_image,
                                pil_img,
                                prompt=CHART_EXTRACTION_PROMPT,
                                max_tokens=settings.vlm_max_tokens,
                            )
                        )
                        parsed = parse_description(raw)
                        embed_text = str(parsed.get("description", raw.strip()))
                        if parsed.get("title"):
                            embed_text = f"{parsed['title']}\n\n{embed_text}"
                        if parsed.get("key_data_points"):
                            points = parsed["key_data_points"]
                            if isinstance(points, list):
                                embed_text += "\n\nKey data: " + "; ".join(str(p) for p in points)
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        img_hash = hashlib.sha256(buf.getvalue()).hexdigest()
                        results[img_name] = (embed_text, parsed, img_hash)
                    except VLMError as e:
                        results[img_name] = e

            async with anyio.create_task_group() as tg:
                for img_name, pil_img in images.items():
                    tg.start_soon(_describe_one, img_name, pil_img)

            for img_name in images:
                r = results[img_name]
                if isinstance(r, tuple):
                    image_descriptions.append(r)
                else:
                    warnings.append(f"VLM failed for image {img_name}: {r}")
        finally:
            vlm.close()
    elif images and not settings.vlm_enabled:
        logger.info("VLM disabled, skipping description of %d images", len(images))

    chunk_texts = [c.content for c in chunk_results]
    chunk_texts.extend(desc for desc, _, _ in image_descriptions)
    embeddings = await anyio.to_thread.run_sync(lambda: embedder.embed_documents(chunk_texts))

    pdf_meta = await anyio.to_thread.run_sync(lambda: extract_metadata(pdf_path))

    conn = StoreConnection.from_env()
    try:
        documents = DocumentRepo(conn)
        chunks_repo = ChunkRepo(conn)
        # Single-transaction boundary: chunk-replace failure must roll back the doc upsert.
        doc_id = documents.upsert_document(
            source_path=str(pdf_path),
            source_sha256=source_sha256,
            title=pdf_meta.title or pdf_path.stem,
            extraction_method=method.value,
            authors=pdf_meta.authors,
            publisher=pdf_meta.publisher,
            year=pdf_meta.year,
            page_count=page_count,
            metadata={
                "keywords": pdf_meta.keywords,
                "subject": pdf_meta.subject,
                "toc": pdf_meta.toc,
            },
            commit=False,
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
                embedding_model=settings.embedding_model,
                embedding_dim=settings.embedding_dim,
            )
            for global_idx, (c, emb) in enumerate(
                zip(chunk_results, embeddings[: len(chunk_results)], strict=True)
            )
        ]
        image_offset = len(chunk_results)
        for i, (desc, meta, img_hash) in enumerate(image_descriptions):
            chunk_rows.append(
                ChunkRow(
                    document_id=doc_id,
                    chunk_index=image_offset + i,
                    section_path="Images",
                    heading_text=str(meta.get("title") or f"Image {i + 1}"),
                    page_start=None,
                    page_end=None,
                    token_count=max(len(desc.split()), 1),
                    content=desc,
                    embedding=embeddings[image_offset + i],
                    chunk_type="image",
                    image_meta=meta,
                    image_hash=img_hash,
                    provenance_model=settings.vlm_model,
                    embedding_model=settings.embedding_model,
                    embedding_dim=settings.embedding_dim,
                )
            )
        try:
            chunks_repo.replace_chunks(doc_id, chunk_rows, commit=False)
        except Exception:
            conn.conn.rollback()
            raise
        conn.conn.commit()
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
        source_path = DocumentRepo(conn).get_source_path(doc_uuid)
    finally:
        conn.close()
    if source_path is None:
        raise KeyError(f"document {document_id} not found")
    return await ingest_pdf(source_path)
