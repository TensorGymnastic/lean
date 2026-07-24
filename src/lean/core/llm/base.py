"""LLM client interface and factory.

Supports any OpenAI-compatible API (MiniMax, Ollama, OpenAI, vLLM, etc.).
The factory returns ``None`` when no LLM is configured, so all LLM-
dependent features degrade gracefully to the no-LLM pipeline.

Reads ``llm_api_key`` (from .env) + ``llm_base_url`` / ``llm_ollama_url``
(from ``CoreSettings``). To extend, subclass ``CoreSettings`` and pass
the subclass to ``get_llm()``.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from lean.core.config.settings import CoreSettings, get_settings

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Interface for text generation LLMs."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str: ...


def get_llm(cls: type[CoreSettings] = CoreSettings) -> LLMClient | None:
    """Build an LLM client for the given Settings subclass, or None.

    Reads the (already cached) ``CoreSettings`` for ``cls`` and returns
    a freshly-built ``OpenAICompatibleLLM`` on every call. ``httpx.Client``
    construction is cheap; if you need connection-pool reuse across calls,
    cache the returned instance yourself.

    Returns ``None`` when no LLM is configured (no ``llm_api_key`` and
    no ``llm_ollama_url``).
    """
    from lean.core.llm.openai_compatible import OpenAICompatibleLLM

    settings = get_settings(cls)
    if settings.llm_api_key:
        logger.info("LLM: %s at %s", settings.llm_model, settings.llm_base_url)
        return OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
        )
    if settings.llm_ollama_url:
        logger.info("LLM: Ollama %s at %s", settings.llm_ollama_model, settings.llm_ollama_url)
        return OpenAICompatibleLLM(
            base_url=settings.llm_ollama_url,
            model=settings.llm_ollama_model,
            api_key="",
            timeout=settings.llm_timeout_s,
        )
    logger.info("LLM: none configured (no contextual retrieval or query transforms)")
    return None


__all__ = ["LLMClient", "get_llm"]
