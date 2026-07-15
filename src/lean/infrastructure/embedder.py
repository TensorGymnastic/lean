"""Singleton embedder factory — chooses remote Ollama or local based on config."""

from __future__ import annotations

from typing import Protocol

from lean.config.settings import Settings


class Embedder(Protocol):
    """Interface shared by LiquidLMFEmbedder and RemoteOllamaEmbedder."""

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the singleton embedder, creating it on first call.

    If ``embedding.remote_url`` is set in config, uses the remote Ollama
    embedder (GPU-accelerated). Otherwise falls back to local CPU.
    """
    global _embedder
    if _embedder is None:
        s = Settings()
        if s.embedding_remote_url and s.embedding_remote_model:
            from lean.embeddings.remote_ollama import RemoteOllamaEmbedder

            _embedder = RemoteOllamaEmbedder(
                base_url=s.embedding_remote_url,
                model=s.embedding_remote_model,
                dim=s.embedding_dim,
            )
        else:
            from lean.embeddings.liquid_lmf import LiquidLMFEmbedder

            _embedder = LiquidLMFEmbedder(
                model=s.embedding_model,
                hf_token=s.hf_token,
                device=s.embedding_device,
                dim=s.embedding_dim,
            )
    return _embedder
