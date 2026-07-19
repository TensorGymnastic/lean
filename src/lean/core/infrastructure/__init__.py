"""Singleton factories shared across the framework."""

from lean.core.infrastructure.embedder import Embedder, get_embedder

__all__ = ["Embedder", "get_embedder"]
