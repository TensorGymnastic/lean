"""Vision-Language Model client for chart/image description.

Works with any API that follows the OpenAI chat completions format with
image_url content parts:
- Ollama (http://gpu-host:11434/v1)
- vLLM (http://gpu-host:8001/v1)
- MiniMax (https://api.minimax.io/v1)
- DashScope (https://dashscope-intl.aliyuncs.com/compatible-mode/v1)

Follows the same httpx-based pattern as ``lean.llm.openai_compatible``.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class VLMError(RuntimeError):
    """Raised when a VLM request fails."""


class VLMClient:
    """Calls /v1/chat/completions on an OpenAI-compatible endpoint with image input.

    Args:
        base_url: API root, no trailing ``/v1`` (e.g. ``http://gpu:11434``,
            ``https://api.minimax.io``). The client appends
            ``/v1/chat/completions`` to match the OpenAI-compatible
            convention used by the OCR and LLM clients.
        model: Model ID (e.g. gemma3:27b, qwen2.5-vl:7b, MiniMax-M3).
        api_key: Bearer token (empty for local servers that don't require auth).
        timeout: Request timeout in seconds.
        detail: Image resolution tier — "low", "default", or "high".
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 120.0,
        detail: str = "default",
        disable_thinking: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._detail = detail
        self._disable_thinking = disable_thinking
        self._client = httpx.Client(timeout=timeout)
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def describe_image(
        self,
        image: Any,
        *,
        prompt: str,
        max_tokens: int = 1000,
    ) -> str:
        """Send an image to the VLM and return the text response.

        Args:
            image: PIL.Image instance.
            prompt: Text instruction for the model.
            max_tokens: Maximum response tokens.

        Returns:
            The model's text response.

        Raises:
            VLMError: On network failure, non-200 status, or empty response.
        """
        data_uri = self._image_to_data_uri(image)

        request_body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri,
                                "detail": self._detail,
                            },
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        if self._disable_thinking:
            request_body["thinking"] = {"type": "disabled"}

        try:
            resp = self._client.post(
                f"{self._base_url}/v1/chat/completions",
                headers=self._headers,
                json=request_body,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise VLMError(f"VLM request failed: {e}") from e

        data = resp.json()
        try:
            content: str = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise VLMError(f"VLM unexpected response shape: {data}") from e

        if not content or not content.strip():
            raise VLMError("VLM returned empty response")

        return content.strip()

    @staticmethod
    def _image_to_data_uri(image: Any) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    def close(self) -> None:
        self._client.close()
