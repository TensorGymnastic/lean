"""Remote Ollama embedding client for LFM2.5-Embedding-350M via GPU.

Same interface as LiquidLMFEmbedder but calls Ollama's /api/embeddings
endpoint instead of loading the model locally. Eliminates the CPU
bottleneck and CUDA driver incompatibility.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

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
        """Embed multiple passages using the Ollama API (parallel)."""
        if not texts:
            return []
        with ThreadPoolExecutor(max_workers=min(8, len(texts))) as pool:
            results = list(pool.map(self._embed_one, texts))
        return results

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self._embed_one(query)

    def _embed_one(self, text: str) -> list[float]:
        resp = self._client.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text},
        )
        resp.raise_for_status()
        data = resp.json()
        embedding: list[float] = data["embedding"]
        return embedding

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self.close()
