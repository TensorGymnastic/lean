"""Marker PDF extraction via datalab-to/marker (surya OCR + texify).

Supports two modes:
1. Remote GPU server (``marker.remote_url`` set) — sends PDF over HTTP,
   gets markdown + images back. Fast (~0.2s/page on GPU).
2. Local (no remote_url) — runs marker in-process. Slow on CPU (~20s/page).

Install: ``uv sync --extra marker``  (local mode only)
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MAX_IMAGES_PER_DOC = 200
MAX_IMAGE_B64_BYTES = 20 * 1024 * 1024  # 20 MiB per decoded image
_MAX_PIL_PIXELS = 50_000_000  # ~50 MP — decompression bomb guard

_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class BlockMeta:
    """Per-block metadata from marker extraction — page number + bbox + text for matching."""

    page: int
    bbox: list[float] | None
    text: str


def _extract_block_metas_from_chunks(chunk_output: Any) -> list[BlockMeta]:
    """Extract BlockMeta list from a ChunkRenderer ChunkOutput.

    Strips HTML tags from each block's html field to produce clean text
    suitable for matching against chunked content.
    """
    metas: list[BlockMeta] = []
    for block in chunk_output.blocks:
        clean_text = _HTML_TAG_RE.sub("", block.html).strip()
        if not clean_text:
            continue
        metas.append(
            BlockMeta(
                page=block.page,
                bbox=list(block.bbox) if block.bbox else None,
                text=clean_text[:500],
            )
        )
    return metas


class MarkerNotInstalled(RuntimeError):
    """Raised when marker-pdf is not installed but marker extraction is requested."""


class MarkerRemoteError(RuntimeError):
    """Raised when the remote marker server returns an error or malformed response.

    Distinct from ``MarkerNotInstalled`` (which means marker is absent and the
    pipeline should fall through to OCR/markitdown). A remote error means the
    server is present but the request failed — logged, then pipeline falls through.
    """


def _pil() -> Any:
    """Lazy-import PIL.Image and set the decompression-bomb guard on first call."""
    from PIL import Image

    if Image.MAX_IMAGE_PIXELS != _MAX_PIL_PIXELS:
        Image.MAX_IMAGE_PIXELS = _MAX_PIL_PIXELS
    return Image


def extract_markdown(
    pdf_path: Path,
    *,
    force_ocr: bool = False,
    remote_url: str = "",
) -> tuple[str, int, dict[str, Any], list[BlockMeta]]:
    """Extract markdown from a PDF via marker.

    Returns ``(markdown, page_count, images, block_metas)`` where ``images`` is a dict of
    ``{image_name: PIL.Image}`` extracted from the PDF (charts, figures,
    diagrams) and ``block_metas`` is per-block page/bbox metadata (local path only;
    empty for remote). Raises ``MarkerNotInstalled`` if marker-pdf is not available
    and no remote_url is set. Raises ``MarkerRemoteError`` on remote failures.
    """
    if remote_url:
        return _extract_remote(pdf_path, remote_url, force_ocr=force_ocr)
    return _extract_local(pdf_path, force_ocr=force_ocr)


def _extract_remote(
    pdf_path: Path, remote_url: str, *, force_ocr: bool = False
) -> tuple[str, int, dict[str, Any], list[BlockMeta]]:
    """Send PDF to remote GPU marker server, get results back."""
    Image = _pil()
    url = remote_url.rstrip("/") + "/extract"
    pdf_bytes = pdf_path.read_bytes()

    logger.info("marker-remote: sending %s (%d bytes) to %s", pdf_path.name, len(pdf_bytes), url)
    transport = httpx.HTTPTransport(retries=2)
    with httpx.Client(timeout=600, transport=transport) as client:
        resp = client.post(url, content=pdf_bytes, headers={"Content-Type": "application/pdf"})

    if resp.status_code >= 500:
        raise MarkerRemoteError(f"remote marker server error: HTTP {resp.status_code}")
    if resp.status_code >= 400:
        raise MarkerRemoteError(
            f"remote marker client error: HTTP {resp.status_code} {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except ValueError as e:
        raise MarkerRemoteError(f"remote marker returned non-JSON response: {e}") from e

    if "error" in data:
        raise MarkerRemoteError(f"remote marker error: {data['error']}")

    try:
        text: str = data["markdown"]
        page_count: int = data["page_count"]
    except KeyError as e:
        raise MarkerRemoteError(f"remote marker response missing key {e}") from e

    raw_images = data.get("images", {})
    if len(raw_images) > MAX_IMAGES_PER_DOC:
        raise MarkerRemoteError(
            f"remote marker returned {len(raw_images)} images (max {MAX_IMAGES_PER_DOC})"
        )

    images: dict[str, Any] = {}
    for name, b64 in raw_images.items():
        if len(b64) > MAX_IMAGE_B64_BYTES:
            raise MarkerRemoteError(
                f"remote marker image '{name}' too large: {len(b64)} bytes "
                f"(max {MAX_IMAGE_B64_BYTES})"
            )
        img_bytes = base64.b64decode(b64, validate=True)
        images[name] = Image.open(io.BytesIO(img_bytes))

    logger.info("marker-remote: %d pages → %d chars, %d images", page_count, len(text), len(images))
    return text, page_count, images, []


def _extract_local(
    pdf_path: Path, *, force_ocr: bool = False
) -> tuple[str, int, dict[str, Any], list[BlockMeta]]:
    """Run marker locally (in-process). Requires ``uv sync --extra marker``.

    Builds the Document once, then renders both markdown (for the chunker) and
    chunk-structured output (for bbox/page metadata). Falls back gracefully if
    ChunkRenderer fails — block_metas will be empty, same as the remote path.
    """
    converter = _get_converter(force_ocr=force_ocr)

    logger.info("marker: converting %s (force_ocr=%s)", pdf_path, force_ocr)
    document = converter.build_document(str(pdf_path))
    page_count = len(document.pages)

    md_renderer = converter.resolve_dependencies(converter.renderer)
    rendered = md_renderer(document)

    from marker.output import text_from_rendered

    text, _, images = text_from_rendered(rendered)

    block_metas: list[BlockMeta] = []
    try:
        from marker.renderers.chunk import ChunkRenderer

        chunk_renderer = converter.resolve_dependencies(ChunkRenderer)
        chunk_output = chunk_renderer(document)
        block_metas = _extract_block_metas_from_chunks(chunk_output)
    except Exception as exc:
        logger.warning("marker: ChunkRenderer metadata extraction failed: %s", exc)

    logger.info(
        "marker: %d pages → %d chars, %d images, %d block metas",
        page_count,
        len(text),
        len(images),
        len(block_metas),
    )
    return text, page_count, images, block_metas


@lru_cache(maxsize=4)
def _get_converter(force_ocr: bool = False) -> Any:
    """Cached singleton PdfConverter (loads surya + texify models on first call).

    Separate cache entries for force_ocr=True vs False.
    """
    try:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError as e:
        raise MarkerNotInstalled(
            "marker-pdf not installed. Install with: uv sync --extra marker"
        ) from e

    logger.info("marker: loading models (first call, ~30s)…")
    config = {"output_format": "markdown", "force_ocr": force_ocr}
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )
    logger.info("marker: models loaded")
    return converter
