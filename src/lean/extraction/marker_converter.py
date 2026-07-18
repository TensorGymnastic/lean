"""Marker PDF extraction via datalab-to/marker (surya OCR + texify).

Lazy-imports marker/PyTorch inside functions so the rest of lean stays
fast at import time (~30s model load only happens when marker is actually
called). The PdfConverter is cached as a singleton via @lru_cache.

Install: ``uv sync --extra marker``
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MarkerNotInstalled(RuntimeError):
    """Raised when marker-pdf is not installed but marker extraction is requested."""


def extract_markdown(
    pdf_path: Path,
    *,
    force_ocr: bool = False,
) -> tuple[str, int, dict[str, Any]]:
    """Extract markdown from a PDF via marker.

    Returns ``(markdown, page_count, images)`` where ``images`` is a dict of
    ``{image_name: PIL.Image}`` extracted from the PDF (charts, figures,
    diagrams). Raises ``MarkerNotInstalled`` if marker-pdf is not available.
    """
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
