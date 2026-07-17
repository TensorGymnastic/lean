"""LLM client interface and factory.

Supports any OpenAI-compatible API (MiniMax, Ollama, OpenAI, vLLM, etc.).
The factory returns ``None`` when no LLM is configured, so all LLM-dependent
features degrade gracefully to the no-LLM pipeline.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

from lean.config.settings import get_settings

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Interface for text generation LLMs."""

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from a prompt. Returns the assistant's response text."""
        ...


@lru_cache(maxsize=1)
def get_llm() -> LLMClient | None:
    """Return the singleton LLM client, or None if no LLM is configured.

    Selects provider based on settings:
    - MiniMax if MINIMAX_API_KEY is set
    - Ollama if llm.ollama_url is set
    - None otherwise (no LLM features)

    Callers must handle ``None`` gracefully.
    """
    s = get_settings()

    if s.minimax_api_key:
        from lean.llm.openai_compatible import OpenAICompatibleLLM

        logger.info("LLM: MiniMax %s at %s", s.llm_minimax_model, s.llm_minimax_base_url)
        return OpenAICompatibleLLM(
            base_url=s.llm_minimax_base_url,
            model=s.llm_minimax_model,
            api_key=s.minimax_api_key,
            timeout=s.llm_timeout_s,
        )
    if s.llm_ollama_url:
        from lean.llm.openai_compatible import OpenAICompatibleLLM

        logger.info("LLM: Ollama %s at %s", s.llm_ollama_model, s.llm_ollama_url)
        return OpenAICompatibleLLM(
            base_url=s.llm_ollama_url,
            model=s.llm_ollama_model,
            api_key="",
            timeout=s.llm_timeout_s,
        )
    logger.info("LLM: none configured (no contextual retrieval or query transforms)")
    return None
