"""Remote Ollama embedding client for LFM2.5-Embedding-350M via GPU.

Same interface as LiquidLMFEmbedder but calls Ollama's /api/embeddings
endpoint instead of loading the model locally. Eliminates the CPU
bottleneck and CUDA driver incompatibility.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class RemoteOllamaEmbedder:
    """Embeds text via a remote Ollama server running LFM2.5 on GPU."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dim: int = 1024,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = dim
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        logger.info("RemoteOllamaEmbedder: %s model=%s dim=%d", base_url, model, dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple passages using the Ollama API (sequential to avoid 500s)."""
        if not texts:
            return []
        return [self._embed_one(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self._embed_one(query)

    def _embed_one(self, text: str) -> list[float]:
        import time

        for attempt in range(3):
            resp = self._client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            if resp.status_code == 200:
                data = resp.json()
                embedding: list[float] = data["embedding"]
                return embedding
            logger.warning(
                "Ollama embedding attempt %d failed: %d %s",
                attempt + 1,
                resp.status_code,
                resp.text[:200],
            )
            time.sleep(1.0)
        resp.raise_for_status()
        return []  # unreachable

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self.close()
