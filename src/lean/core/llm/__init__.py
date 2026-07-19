"""LLM sidecar: OpenAI-compatible text generation. Optional — features
that depend on it degrade gracefully when no LLM is configured."""

from lean.core.llm.base import LLMClient, get_llm
from lean.core.llm.openai_compatible import OpenAICompatibleLLM

__all__ = ["LLMClient", "OpenAICompatibleLLM", "get_llm"]
