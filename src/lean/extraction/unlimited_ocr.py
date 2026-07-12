"""Unlimited-OCR client via vLLM OpenAI-compatible HTTP API.

Calls a remote vLLM server serving ``baidu/Unlimited-OCR`` with PDF page
images. Raises ``OCRBackendUnavailable`` when vLLM is unreachable so the
pipeline orchestrator can fall back to markitdown.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import httpx

logger = logging.getLogger(__name__)


class OCRBackendUnavailable(RuntimeError):
    """Raised when the remote vLLM backend is unreachable or returns an error."""


def extract_markdown(
    pdf_path: Path,
    *,
    vllm_base_url: str,
    hf_token: str | None = None,
    model: str = "baidu/Unlimited-OCR",
    dpi: int = 300,
    timeout: float = 600.0,
    max_tokens: int = 32768,
) -> tuple[str, int]:
    """Extract markdown from a PDF via Unlimited-OCR served on vLLM.

    Renders each PDF page to a PNG at ``dpi``, sends them all to vLLM's
    ``/v1/chat/completions`` endpoint as base64-encoded images, and returns
    the model's markdown response.

    Args:
        pdf_path: Path to the PDF file.
        vllm_base_url: Base URL of the vLLM server (e.g. ``http://3080ti:8000``).
        hf_token: Optional HuggingFace token for the Authorization header.
        model: vLLM model name to request.
        dpi: DPI for page rendering (300 is Unlimited-OCR's recommended default).
        timeout: HTTP timeout in seconds (large PDFs take time).
        max_tokens: Max output tokens for the vLLM completion.

    Returns:
        Tuple of (markdown_text, page_count).

    Raises:
        OCRBackendUnavailable: On connection errors, HTTP errors, or timeouts.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    page_images = _pdf_to_base64_images(pdf_path, dpi=dpi)
    page_count = len(page_images)
    logger.info("sending %d pages to Unlimited-OCR at %s", page_count, vllm_base_url)

    content_parts: list[dict[str, object]] = [{"type": "text", "text": "Multi page parsing."}]
    for img_b64 in page_images:
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
    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{vllm_base_url.rstrip('/')}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.HTTPStatusError, httpx.ReadTimeout) as exc:
        raise OCRBackendUnavailable(f"vLLM call failed: {exc}") from exc

    data = resp.json()
    markdown: str = data["choices"][0]["message"]["content"]
    logger.info("Unlimited-OCR returned %d chars of markdown", len(markdown))
    return markdown, page_count


def _pdf_to_base64_images(pdf_path: Path, *, dpi: int) -> list[str]:
    """Render each PDF page to a PNG and return base64-encoded strings.

    Uses a temp directory for intermediate PNG files.
    """
    doc = fitz.open(str(pdf_path))
    tmp_dir = tempfile.mkdtemp(prefix="lean_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    images: list[str] = []
    try:
        for i, page in enumerate(doc):
            out = os.path.join(tmp_dir, f"page_{i:04d}.png")
            page.get_pixmap(matrix=mat).save(out)
            with open(out, "rb") as f:
                images.append(base64.b64encode(f.read()).decode("ascii"))
    finally:
        doc.close()
    return images
