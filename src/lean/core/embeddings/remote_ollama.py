"""Remote Ollama embedding client for LFM2.5-Embedding-350M via GPU.

Calls Ollama's ``/api/embeddings`` endpoint instead of loading the model
locally. Uses configurable context window to match the model's capacity.
"""

from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)


class RemoteOllamaEmbedder:
    """Embeds text via a remote Ollama server running LFM2.5 on GPU.

    Args:
        base_url: Ollama server URL (e.g. http://gpu-host:11434).
        model: Ollama model name (e.g. lfm2.5-embed-32k).
        dim: Embedding dimension (1024 for LFM2.5).
        timeout: HTTP timeout in seconds.
        num_ctx: Context window size passed to Ollama. Must match the
            model's configured ``PARAMETER num_ctx`` (32768 for 32K model).
        max_retries: Number of retry attempts on non-200 responses.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dim: int = 1024,
        timeout: float = 30.0,
        num_ctx: int = 32768,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = dim
        self._num_ctx = num_ctx
        self._max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)
        logger.info(
            "RemoteOllamaEmbedder: %s model=%s dim=%d num_ctx=%d",
            base_url,
            model,
            dim,
            num_ctx,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple passages (sequential to avoid Ollama 500s)."""
        if not texts:
            return []
        return [self._embed_one(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self._embed_one(query)

    def _embed_one(self, text: str) -> list[float]:
        """Embed a single text with retry logic."""
        for attempt in range(self._max_retries):
            resp = self._client.post(
                f"{self._base_url}/api/embeddings",
                json={
                    "model": self._model,
                    "prompt": text,
                    "options": {"num_ctx": self._num_ctx},
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                embedding: list[float] = data["embedding"]
                return embedding
            if attempt < self._max_retries - 1:
                wait = 2**attempt
                logger.warning(
                    "Ollama embedding attempt %d/%d failed: %d — retrying in %ds",
                    attempt + 1,
                    self._max_retries,
                    resp.status_code,
                    wait,
                )
                time.sleep(wait)
        resp.raise_for_status()
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._client.close()
