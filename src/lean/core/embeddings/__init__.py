"""Embedder implementations: local CPU (sentence-transformers) and remote Ollama."""

from lean.core.embeddings.liquid_lmf import LiquidLMFEmbedder
from lean.core.embeddings.remote_ollama import RemoteOllamaEmbedder

__all__ = ["LiquidLMFEmbedder", "RemoteOllamaEmbedder"]
