"""Marker-pdf extraction (datalab-to/marker, surya OCR + texify).

Two modes:
1. Remote GPU server (``remote_url`` set) — sends PDF over HTTP, gets
   markdown + images back. Fast (~0.2s/page on GPU).
2. Local (no remote_url) — runs marker in-process. Slow on CPU
   (~20s/page). Requires ``uv sync --extra marker``.

Implements the universal ``Extractor`` Protocol — usable as one link in
a domain's ``Pipeline([...])``.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from lean.core.extraction.base import (
    BackendUnavailable,
    ExtractionResult,
    ExtractorUnavailable,
)
from lean.core.models import ExtractionMethod

logger = logging.getLogger(__name__)

MAX_IMAGES_PER_DOC = 200
MAX_IMAGE_B64_BYTES = 20 * 1024 * 1024
_MAX_PIL_PIXELS = 50_000_000

_HTML_TAG_RE = __import__("re").compile(r"<[^>]+>")


@dataclass
class BlockMeta:
    """Per-block metadata from marker extraction — page number + bbox + text."""

    page: int
    bbox: list[float] | None
    text: str


def _extract_block_metas_from_chunks(chunk_output: Any) -> list[BlockMeta]:
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


def _pil() -> Any:
    from PIL import Image

    if Image.MAX_IMAGE_PIXELS != _MAX_PIL_PIXELS:
        Image.MAX_IMAGE_PIXELS = _MAX_PIL_PIXELS
    return Image


def _extract_remote(
    pdf_path: Path, remote_url: str, *, force_ocr: bool = False
) -> ExtractionResult:
    Image = _pil()
    url = remote_url.rstrip("/") + "/extract"
    pdf_bytes = pdf_path.read_bytes()

    logger.info("marker-remote: sending %s (%d bytes) to %s", pdf_path.name, len(pdf_bytes), url)
    transport = httpx.HTTPTransport(retries=2)
    with httpx.Client(timeout=600, transport=transport) as client:
        resp = client.post(url, content=pdf_bytes, headers={"Content-Type": "application/pdf"})

    if resp.status_code >= 500:
        raise ExtractorUnavailable(f"remote marker server error: HTTP {resp.status_code}")
    if resp.status_code >= 400:
        raise ExtractorUnavailable(
            f"remote marker client error: HTTP {resp.status_code} {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except ValueError as e:
        raise ExtractorUnavailable(f"remote marker returned non-JSON response: {e}") from e

    if "error" in data:
        raise ExtractorUnavailable(f"remote marker error: {data['error']}")

    try:
        text: str = data["markdown"]
        page_count: int = data["page_count"]
    except KeyError as e:
        raise ExtractorUnavailable(f"remote marker response missing key {e}") from e

    raw_images = data.get("images", {})
    if len(raw_images) > MAX_IMAGES_PER_DOC:
        raise ExtractorUnavailable(
            f"remote marker returned {len(raw_images)} images (max {MAX_IMAGES_PER_DOC})"
        )

    images: dict[str, Any] = {}
    for name, b64 in raw_images.items():
        if len(b64) > MAX_IMAGE_B64_BYTES:
            raise ExtractorUnavailable(
                f"remote marker image '{name}' too large: {len(b64)} bytes "
                f"(max {MAX_IMAGE_B64_BYTES})"
            )
        img_bytes = base64.b64decode(b64, validate=True)
        images[name] = Image.open(io.BytesIO(img_bytes))

    return ExtractionResult(
        markdown=text,
        page_count=page_count,
        method=ExtractionMethod.MARKER,
        images=images,
        block_metas=[],
    )


def _extract_local(pdf_path: Path, *, force_ocr: bool = False) -> ExtractionResult:
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

    return ExtractionResult(
        markdown=text,
        page_count=page_count,
        method=ExtractionMethod.MARKER,
        images=images,
        block_metas=block_metas,
    )


@lru_cache(maxsize=4)
def _get_converter(force_ocr: bool = False) -> Any:
    try:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError as e:
        raise BackendUnavailable(
            "marker-pdf not installed. Install with: uv sync --extra marker"
        ) from e

    logger.info("marker: loading models (first call, ~30s)…")
    config = {"output_format": "markdown", "force_ocr": force_ocr}
    config_parser = ConfigParser(config)
    return PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )


class MarkerExtractor:
    """``Extractor`` Protocol implementation: datalab-to/marker.

    Configure with ``force_ocr`` and ``remote_url``. If ``remote_url`` is
    set, sends the PDF to that HTTP endpoint; otherwise runs marker
    in-process (requires the ``marker`` extra).
    """

    method = ExtractionMethod.MARKER

    def __init__(self, *, force_ocr: bool = False, remote_url: str = "") -> None:
        self._force_ocr = force_ocr
        self._remote_url = remote_url

    def is_configured(self) -> bool:
        if self._remote_url:
            return True
        try:
            import marker  # noqa: F401

            return True
        except ImportError:
            return False

    def extract(self, pdf_path: Path) -> ExtractionResult:
        if self._remote_url:
            return _extract_remote(pdf_path, self._remote_url, force_ocr=self._force_ocr)
        return _extract_local(pdf_path, force_ocr=self._force_ocr)


__all__ = [
    "BlockMeta",
    "MarkerExtractor",
]
