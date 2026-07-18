"""Unlimited-OCR client via an OpenAI-compatible HTTP API.

Calls a remote server serving ``baidu/Unlimited-OCR`` with PDF page
images. Raises ``OCRBackendUnavailable`` when the server is unreachable so
the pipeline orchestrator can fall back to markitdown.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import fitz  # PyMuPDF
import httpx

logger = logging.getLogger(__name__)

MAX_PAGE_PIXELS = 50_000_000  # 50 MP — blocks page-bomb PDFs (100"×100" @ 300 DPI = 900 MP)


class OCRBackendUnavailable(RuntimeError):
    """Raised when the remote OCR backend is unreachable or returns an error."""


def extract_markdown(
    pdf_path: Path,
    *,
    ocr_base_url: str,
    hf_token: str | None = None,
    model: str = "baidu/Unlimited-OCR",
    dpi: int = 300,
    timeout: float = 1800.0,
    max_tokens: int = 32768,
    batch_size: int = 20,
) -> tuple[str, int]:
    """Extract markdown from a PDF via Unlimited-OCR, batching pages to avoid oversized requests.

    Renders each PDF page to a PNG at ``dpi``, sends them in batches of
    ``batch_size`` to the OCR server, and concatenates the markdown responses.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    page_images = _pdf_to_base64_images(pdf_path, dpi=dpi)
    page_count = len(page_images)
    logger.info("OCR: %d pages, batch_size=%d, server=%s", page_count, batch_size, ocr_base_url)

    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    markdown_parts: list[str] = []
    total_batches = (page_count + batch_size - 1) // batch_size

    try:
        with httpx.Client(timeout=timeout) as client:
            for batch_idx in range(total_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, page_count)
                batch = page_images[start:end]

                logger.info(
                    "OCR batch %d/%d: pages %d-%d",
                    batch_idx + 1,
                    total_batches,
                    start + 1,
                    end,
                )

                content_parts: list[dict[str, object]] = [
                    {"type": "text", "text": "Multi page parsing."}
                ]
                for img_b64 in batch:
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        }
                    )

                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": content_parts}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "stream": False,
                }

                resp = client.post(
                    f"{ocr_base_url.rstrip('/')}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()

                data = resp.json()
                batch_markdown: str = data["choices"][0]["message"]["content"]
                markdown_parts.append(batch_markdown)
                logger.info(
                    "OCR batch %d/%d done: %d chars",
                    batch_idx + 1,
                    total_batches,
                    len(batch_markdown),
                )
    except (httpx.ConnectError, httpx.HTTPStatusError, httpx.ReadTimeout) as exc:
        raise OCRBackendUnavailable(f"OCR server call failed: {exc}") from exc

    markdown = "\n\n".join(markdown_parts)
    logger.info("OCR complete: %d pages → %d chars", page_count, len(markdown))
    return markdown, page_count


def _pdf_to_base64_images(pdf_path: Path, *, dpi: int) -> list[str]:
    """Render each PDF page to a PNG and return base64-encoded strings."""
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    images: list[str] = []
    try:
        for page in doc:
            rect = page.rect
            rasterized_pixels = rect.width * (dpi / 72) * rect.height * (dpi / 72)
            if rasterized_pixels > MAX_PAGE_PIXELS:
                max_mp = MAX_PAGE_PIXELS / 1e6
                actual_mp = rasterized_pixels / 1e6
                raise ValueError(
                    f"PDF page too large: {rect.width:.0f}x{rect.height:.0f} pt "
                    f"at {dpi} DPI = {actual_mp:.0f} MP (max {max_mp:.0f} MP)"
                )
            pixmap = page.get_pixmap(matrix=mat)
            png_bytes = pixmap.tobytes("png")
            images.append(base64.b64encode(png_bytes).decode("ascii"))
    finally:
        doc.close()
    return images
