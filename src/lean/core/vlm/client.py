"""Vision-Language Model client — OpenAI-compatible transport.

Sends an image (PIL.Image) to a /v1/chat/completions endpoint and
returns the raw text response. Domain code is responsible for the
prompt + response parsing (e.g. JSON fence handling) — that is a
domain decision, not a transport one.

Works with any API that follows the OpenAI chat completions format
with image_url content parts: Ollama, vLLM, MiniMax, DashScope, etc.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


class VLMError(RuntimeError):
    """Raised when a VLM request fails."""


@runtime_checkable
class VLMClient(Protocol):
    """Universal VLM contract — domain code uses this type.

    Implementations: ``OpenAICompatibleVLM`` (default), or domain-
    supplied wrappers.
    """

    def describe_image(self, image: Any, *, prompt: str, max_tokens: int = 1000) -> str: ...


class OpenAICompatibleVLM:
    """Calls /v1/chat/completions with an image_url content part.

    Args:
        base_url: API root, no trailing ``/v1`` (e.g. ``http://gpu:11434``,
            ``https://api.minimax.io``). The client appends
            ``/v1/chat/completions``.
        model: Model ID (e.g. ``gemma3:27b``, ``qwen2.5-vl:7b``, ``MiniMax-M3``).
        api_key: Bearer token (empty for local servers that don't require auth).
        timeout: Request timeout in seconds.
        detail: Image resolution tier — "low", "default", or "high".
        disable_thinking: Some providers (e.g. MiniMax) accept ``thinking``
            flags; this passes through. Ignored if the provider doesn't support it.
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
        """Send an image to the VLM and return the raw text response.

        The prompt is whatever the domain decides to ask. This method
        does no parsing — domain code parses the response.
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


__all__ = ["VLMClient", "VLMError", "OpenAICompatibleVLM"]
