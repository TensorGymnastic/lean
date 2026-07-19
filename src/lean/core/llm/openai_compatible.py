"""OpenAI-compatible LLM client.

Works with any API that follows the OpenAI chat completions format:
- MiniMax (api.minimax.io)
- Ollama (localhost:11434 or remote)
- vLLM, LM Studio, LiteLLM, etc.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM:
    """Calls /v1/chat/completions on an OpenAI-compatible endpoint.

    Args:
        base_url: API root (e.g. https://api.minimax.io, http://gpu:11434).
        model: Model ID (e.g. MiniMax-Text-01, qwen3.5:9b).
        api_key: Bearer token (empty for local servers that don't require auth).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.Client(timeout=timeout)
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        """Generate text via chat completions endpoint."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.post(
            f"{self._base_url}/v1/chat/completions",
            headers=self._headers,
            json={
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content: str = data["choices"][0]["message"]["content"]
        return content.strip()

    def close(self) -> None:
        self._client.close()
