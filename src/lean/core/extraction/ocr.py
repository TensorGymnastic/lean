"""Unlimited-OCR client via an OpenAI-compatible HTTP API.

Calls a remote server serving ``baidu/Unlimited-OCR`` with PDF page
images. Raises ``BackendUnavailable`` when the server is unreachable so
the pipeline orchestrator can fall through.

Implements the universal ``Extractor`` Protocol.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import httpx

from lean.core.extraction.base import BackendUnavailable, ExtractionResult
from lean.core.extraction.ocr_postprocess import clean_ocr_output
from lean.core.models import ExtractionMethod

logger = logging.getLogger(__name__)

MAX_PAGE_PIXELS = 50_000_000  # blocks page-bomb PDFs (100"×100" @ 300 DPI = 900 MP)


class UnlimitedOCRExtractor:
    """``Extractor`` Protocol implementation: baidu/Unlimited-OCR via remote transformers.

    Configure with ``base_url``, ``hf_token``, ``model``, ``dpi``,
    ``timeout``, ``max_tokens``, ``batch_size``.
    """

    method = ExtractionMethod.UNLIMITED_OCR

    def __init__(
        self,
        *,
        base_url: str,
        hf_token: str | None = None,
        model: str,
        dpi: int,
        timeout: float,
        max_tokens: int,
        batch_size: int,
    ) -> None:
        self._base_url = base_url
        self._hf_token = hf_token
        self._model = model
        self._dpi = dpi
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._batch_size = batch_size

    @classmethod
    def apply_settings_defaults(cls, config: dict[str, Any], settings: Any) -> dict[str, Any]:
        out = dict(config)
        if not out.get("base_url"):
            out["base_url"] = settings.ocr_base_url
        if not out.get("model"):
            out["model"] = settings.ocr_model
        if not out.get("dpi"):
            out["dpi"] = settings.ocr_dpi
        if not out.get("timeout"):
            out["timeout"] = settings.ocr_timeout_s
        if not out.get("max_tokens"):
            out["max_tokens"] = settings.ocr_max_tokens
        if not out.get("batch_size"):
            out["batch_size"] = settings.ocr_batch_size
        return out

    def is_configured(self) -> bool:
        return bool(self._base_url)

    def extract(self, pdf_path: Path) -> ExtractionResult:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        page_images = self._pdf_to_base64_images(pdf_path, dpi=self._dpi)
        page_count = len(page_images)
        logger.info(
            "OCR: %d pages, batch_size=%d, server=%s", page_count, self._batch_size, self._base_url
        )

        headers = {"Content-Type": "application/json"}
        if self._hf_token:
            headers["Authorization"] = f"Bearer {self._hf_token}"

        markdown_parts: list[str] = []
        total_batches = (page_count + self._batch_size - 1) // self._batch_size

        try:
            with httpx.Client(timeout=self._timeout) as client:
                for batch_idx in range(total_batches):
                    start = batch_idx * self._batch_size
                    end = min(start + self._batch_size, page_count)
                    batch = page_images[start:end]

                    logger.info(
                        "OCR batch %d/%d: pages %d-%d", batch_idx + 1, total_batches, start + 1, end
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
                        "model": self._model,
                        "messages": [{"role": "user", "content": content_parts}],
                        "temperature": 0,
                        "max_tokens": self._max_tokens,
                        "stream": False,
                    }

                    resp = client.post(
                        f"{self._base_url.rstrip('/')}/v1/chat/completions",
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
            raise BackendUnavailable(f"OCR server call failed: {exc}") from exc

        markdown = clean_ocr_output("\n\n".join(markdown_parts))
        logger.info("OCR complete: %d pages → %d chars", page_count, len(markdown))
        return ExtractionResult(
            markdown=markdown,
            page_count=page_count,
            method=ExtractionMethod.UNLIMITED_OCR,
            images={},
            block_metas=[],
        )

    @staticmethod
    def _pdf_to_base64_images(pdf_path: Path, *, dpi: int) -> list[str]:
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


__all__ = ["UnlimitedOCRExtractor"]
