"""Singleton embedder factory — chooses remote Ollama or local based on config."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from lean.config.settings import get_settings


class Embedder(Protocol):
    """Interface shared by LiquidLMFEmbedder and RemoteOllamaEmbedder."""

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the singleton embedder, creating it on first call.

    If ``embedding.remote_url`` is set in config, uses the remote Ollama
    embedder (GPU-accelerated). Otherwise falls back to local CPU.
    """
    s = get_settings()
    if s.embedding_remote_url and s.embedding_remote_model:
        from lean.embeddings.remote_ollama import RemoteOllamaEmbedder

        return RemoteOllamaEmbedder(
            base_url=s.embedding_remote_url,
            model=s.embedding_remote_model,
            dim=s.embedding_dim,
            num_ctx=s.embedding_num_ctx,
            timeout=s.embedding_http_timeout,
            max_retries=s.embedding_max_retries,
        )
    from lean.embeddings.liquid_lmf import LiquidLMFEmbedder

    return LiquidLMFEmbedder(
        model=s.embedding_model,
        hf_token=s.hf_token,
        device=s.embedding_device,
        dim=s.embedding_dim,
        revision=s.embedding_model_revision,
    )
