"""Tests for LiquidLMFEmbedder.

Marked @pytest.mark.slow — downloads ~700MB model on first run.
Skip with: pytest -m "not slow"
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def test_embed_documents_returns_1024_dim() -> None:
    """Document embeddings are 1024-dim and normalized."""
    from lean.embeddings.liquid_lmf import LiquidLMFEmbedder

    embedder = LiquidLMFEmbedder(model="LiquidAI/LFM2.5-Embedding-350M")
    vectors = embedder.embed_documents(["DMAIC is the Six Sigma methodology."])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1024
    norm = sum(v * v for v in vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-3


def test_embed_query_returns_1024_dim() -> None:
    """Query embedding uses the 'query' prompt and returns 1024-dim."""
    from lean.embeddings.liquid_lmf import LiquidLMFEmbedder

    embedder = LiquidLMFEmbedder(model="LiquidAI/LFM2.5-Embedding-350M")
    vec = embedder.embed_query("What is DMAIC?")
    assert len(vec) == 1024
