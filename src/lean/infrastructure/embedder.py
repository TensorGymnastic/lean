"""Singleton embedder factory — single source of truth for the LiquidLMF embedder."""

from __future__ import annotations

from lean.config.settings import Settings
from lean.embeddings.liquid_lmf import LiquidLMFEmbedder

_embedder: LiquidLMFEmbedder | None = None


def get_embedder() -> LiquidLMFEmbedder:
    """Return the singleton embedder, creating it on first call."""
    global _embedder
    if _embedder is None:
        s = Settings()
        _embedder = LiquidLMFEmbedder(
            model=s.embedding_model,
            hf_token=s.hf_token,
            device=s.embedding_device,
            dim=s.embedding_dim,
        )
    return _embedder
