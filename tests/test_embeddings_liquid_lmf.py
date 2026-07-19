"""Tests for LiquidLMFEmbedder.

Functional tests are marked @pytest.mark.slow — they download ~700MB model
on first run. Smoke tests run by default (mocking SentenceTransformer).
Skip functional tests with: pytest -m "not slow"
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_st():
    """Patch SentenceTransformer so .__init__ does NOT download the model."""
    with patch("lean.embeddings.liquid_lmf.SentenceTransformer") as mock_cls:
        mock_cls.return_value = MagicMock()
        yield mock_cls


@pytest.mark.slow
def test_embed_documents_returns_1024_dim() -> None:
    """Document embeddings are 1024-dim and normalized."""
    from lean.embeddings.liquid_lmf import LiquidLMFEmbedder

    embedder = LiquidLMFEmbedder(model="LiquidAI/LFM2.5-Embedding-350M")
    vectors = embedder.embed_documents(["DMAIC is the Six Sigma methodology."])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1024
    norm = sum(v * v for v in vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-3


@pytest.mark.slow
def test_embed_query_returns_1024_dim() -> None:
    """Query embedding uses the 'query' prompt and returns 1024-dim."""
    from lean.embeddings.liquid_lmf import LiquidLMFEmbedder

    embedder = LiquidLMFEmbedder(model="LiquidAI/LFM2.5-Embedding-350M")
    vec = embedder.embed_query("What is DMAIC?")
    assert len(vec) == 1024


def test_init_passes_revision_and_device_to_sentence_transformer(fake_st) -> None:
    """Default revision SHA + device are forwarded to SentenceTransformer."""
    from lean.embeddings.liquid_lmf import LiquidLMFEmbedder

    LiquidLMFEmbedder()
    call = fake_st.call_args
    assert call.kwargs["revision"] == "f35ae2c91d687658dbf1f2b449382f0b019b9808"
    assert call.kwargs["device"] == "cpu"
    assert call.kwargs["trust_remote_code"] is True


def test_dim_property_returns_configured_dim(fake_st) -> None:
    """Custom dim= arg is exposed via the dim property."""
    from lean.embeddings.liquid_lmf import LiquidLMFEmbedder

    embedder = LiquidLMFEmbedder(dim=768)
    assert embedder.dim == 768


def test_embed_documents_uses_document_prompt(fake_st) -> None:
    """embed_documents forwards prompt_name='document' to the model."""
    from lean.embeddings.liquid_lmf import LiquidLMFEmbedder

    embedder = LiquidLMFEmbedder()
    embedder.embed_documents(["x", "y"])
    encode_calls = [
        c for c in fake_st.return_value.encode.call_args_list if "prompt_name" in c.kwargs
    ]
    assert len(encode_calls) >= 1
    for c in encode_calls:
        assert c.kwargs["prompt_name"] == "document"
        assert c.kwargs["normalize_embeddings"] is True


def test_embed_query_uses_query_prompt(fake_st) -> None:
    """embed_query forwards prompt_name='query' to the model."""
    from lean.embeddings.liquid_lmf import LiquidLMFEmbedder

    embedder = LiquidLMFEmbedder()
    embedder.embed_query("hi")
    encode_call = fake_st.return_value.encode.call_args
    assert encode_call.kwargs["prompt_name"] == "query"
