"""Helpers used by the ingestion service — kept out of ingestion.py
so the orchestrator stays at one concern.

``describe_images`` calls the VLM (if enabled); the prompt + parser
are domain concerns, so this function takes them as callables. The
domain provides them at registration time.

``build_chunk_rows`` + ``enrich_chunks_with_block_meta`` +
``find_best_block_match`` are universal data-shaping helpers.
"""

from __future__ import annotations

import hashlib
import io
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID

import anyio

from lean.core.chunker.recursive import ChunkResult
from lean.core.config.settings import CoreSettings
from lean.core.extraction.marker import BlockMeta
from lean.core.models import CHUNK_TYPE_IMAGE
from lean.core.store.chunks import ChunkRow

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

ImageDescription = tuple[str, dict[str, object], str]
"""(embed_text, parsed_meta, image_hash) tuple per image."""


async def describe_images(
    images: dict[str, PILImage],
    settings: CoreSettings,
    *,
    describe_one: Callable[..., Any] | None = None,
    prompt: str | None = None,
) -> tuple[list[ImageDescription], list[str]]:
    """Describe each image via the domain's VLM callback.

    Returns ``(descriptions, warnings)``:
    - descriptions: tuples of ``(embed_text, parsed_meta, image_hash)``
      for successful calls.
    - warnings: human-readable strings for failed calls (VLMError caught per-image).

    The ``describe_one`` and ``prompt`` callables let the domain inject
    its VLM prompt + parser. When neither is provided, no descriptions
    are produced (the function is a no-op pass-through).
    """
    if not images or describe_one is None:
        if images and describe_one is None:
            logger.info("no VLM describe_one callable; skipping %d images", len(images))
        return [], []

    sem = anyio.Semaphore(settings.vlm_max_concurrency)
    results: dict[str, ImageDescription | Exception] = {}

    async def _run(name: str, img: PILImage) -> None:
        async with sem:
            try:
                results[name] = await describe_one(name, img, settings, prompt)
            except Exception as exc:
                results[name] = exc

    async with anyio.create_task_group() as tg:
        for name, img in images.items():
            tg.start_soon(_run, name, img)

    descriptions: list[ImageDescription] = []
    warnings: list[str] = []
    for name in images:
        r = results[name]
        if isinstance(r, tuple):
            descriptions.append(r)
        else:
            warnings.append(f"VLM failed for image {name}: {r}")
    return descriptions, warnings


def find_best_block_match(
    content: str,
    blocks: list[BlockMeta],
    *,
    min_overlap: float = 0.15,
) -> tuple[int | None, dict[str, float] | None]:
    """Find the block whose text overlaps most with the chunk content.

    Uses Jaccard-like word overlap. Returns (page, bbox_dict) or
    (None, None) if no block meets the ``min_overlap`` threshold.
    """
    content_words = set(content.lower().split())
    if not content_words:
        return None, None

    best_score = 0.0
    best_page: int | None = None
    best_bbox: list[float] | None = None

    for block in blocks:
        block_words = set(block.text.lower().split())
        if not block_words:
            continue
        overlap = len(content_words & block_words)
        score = overlap / min(len(content_words), len(block_words))
        if score > best_score:
            best_score = score
            best_page = block.page
            best_bbox = block.bbox

    if best_score < min_overlap:
        return None, None
    if best_bbox and len(best_bbox) == 4:
        return best_page, {
            "x0": best_bbox[0],
            "y0": best_bbox[1],
            "x1": best_bbox[2],
            "y1": best_bbox[3],
        }
    return best_page, None


def enrich_chunks_with_block_meta(
    chunks: list[ChunkResult],
    block_metas: list[BlockMeta],
    *,
    min_overlap: float,
) -> None:
    """Populate bbox + page_start/page_end on chunks by matching content to blocks."""
    if not block_metas:
        return
    for chunk in chunks:
        page, bbox = find_best_block_match(chunk.content, block_metas, min_overlap=min_overlap)
        chunk.page_start = page
        chunk.page_end = page
        chunk.bbox = bbox


def build_chunk_rows(
    chunk_results: list[ChunkResult],
    embeddings: list[list[float]],
    image_descriptions: list[ImageDescription],
    settings: CoreSettings,
    doc_id: UUID,
    *,
    chunk_type_for: Callable[[ChunkResult], str] | None = None,
) -> list[ChunkRow]:
    """Zip chunk results + image descriptions with their embeddings into ChunkRow instances.

    Image rows follow text rows at offset ``len(chunk_results)``; their
    embeddings come from the same ``embeddings`` list (text embeddings
    first, image embeddings after).
    """
    classify = chunk_type_for or _default_chunk_type_for
    chunk_rows = [
        ChunkRow(
            document_id=doc_id,
            chunk_index=global_idx,
            section_path=c.section_path,
            heading_text=c.heading_text,
            page_start=c.page_start,
            page_end=c.page_end,
            token_count=c.token_count,
            content=c.content,
            embedding=emb,
            chunk_type=classify(c),
        )
        for global_idx, (c, emb) in enumerate(
            zip(chunk_results, embeddings[: len(chunk_results)], strict=True)
        )
    ]
    image_offset = len(chunk_results)
    for i, (desc, _meta, _img_hash) in enumerate(image_descriptions):
        chunk_rows.append(
            ChunkRow(
                document_id=doc_id,
                chunk_index=image_offset + i,
                section_path="Images",
                heading_text=f"Image {i + 1}",
                page_start=None,
                page_end=None,
                token_count=max(len(desc.split()), 1),
                content=desc,
                embedding=embeddings[image_offset + i],
                chunk_type=CHUNK_TYPE_IMAGE,
            )
        )
    return chunk_rows


def _default_chunk_type_for(_chunk: ChunkResult) -> str:
    from lean.core.models import CHUNK_TYPE_TEXT

    return CHUNK_TYPE_TEXT


def hash_image(pil_img: PILImage) -> str:
    """SHA-256 of a PIL image's PNG bytes — used for image dedup."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


__all__ = [
    "ImageDescription",
    "describe_images",
    "find_best_block_match",
    "enrich_chunks_with_block_meta",
    "build_chunk_rows",
    "hash_image",
]
