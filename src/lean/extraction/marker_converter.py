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
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MAX_IMAGES_PER_DOC = 200
MAX_IMAGE_B64_BYTES = 20 * 1024 * 1024  # 20 MiB per decoded image
_MAX_PIL_PIXELS = 50_000_000  # ~50 MP — decompression bomb guard


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
) -> tuple[str, int, dict[str, Any]]:
    """Extract markdown from a PDF via marker.

    Returns ``(markdown, page_count, images)`` where ``images`` is a dict of
    ``{image_name: PIL.Image}`` extracted from the PDF (charts, figures,
    diagrams). Raises ``MarkerNotInstalled`` if marker-pdf is not available
    and no remote_url is set. Raises ``MarkerRemoteError`` on remote failures.
    """
    if remote_url:
        return _extract_remote(pdf_path, remote_url, force_ocr=force_ocr)
    return _extract_local(pdf_path, force_ocr=force_ocr)


def _extract_remote(
    pdf_path: Path, remote_url: str, *, force_ocr: bool = False
) -> tuple[str, int, dict[str, Any]]:
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
    return text, page_count, images


def _extract_local(pdf_path: Path, *, force_ocr: bool = False) -> tuple[str, int, dict[str, Any]]:
    """Run marker locally (in-process). Requires ``uv sync --extra marker``."""
    converter = _get_converter(force_ocr=force_ocr)

    logger.info("marker: converting %s (force_ocr=%s)", pdf_path, force_ocr)
    rendered = converter(str(pdf_path))

    from marker.output import text_from_rendered

    text, _, images = text_from_rendered(rendered)

    page_count = 1
    meta = rendered.metadata if hasattr(rendered, "metadata") else {}
    if isinstance(meta, dict) and "page_stats" in meta:
        page_count = len(meta["page_stats"])

    logger.info("marker: %d pages → %d chars, %d images", page_count, len(text), len(images))
    return text, page_count, images


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
