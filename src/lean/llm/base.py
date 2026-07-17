"""LLM client interface and factory.

Supports any OpenAI-compatible API (MiniMax, Ollama, OpenAI, vLLM, etc.).
The factory returns ``None`` when no LLM is configured, so all LLM-dependent
features degrade gracefully to the no-LLM pipeline.
"""

from __future__ import annotations

import logging
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


_llm: LLMClient | None = None
_llm_checked: bool = False


def get_llm() -> LLMClient | None:
    """Return the singleton LLM client, or None if no LLM is configured.

    Selects provider based on settings:
    - MiniMax if MINIMAX_API_KEY is set
    - Ollama if llm.ollama_url is set
    - None otherwise (no LLM features)

    Callers must handle ``None`` gracefully.
    """
    global _llm, _llm_checked
    if _llm_checked:
        return _llm
    _llm_checked = True

    s = get_settings()

    if s.minimax_api_key:
        from lean.llm.openai_compatible import OpenAICompatibleLLM

        _llm = OpenAICompatibleLLM(
            base_url=s.llm_minimax_base_url,
            model=s.llm_minimax_model,
            api_key=s.minimax_api_key,
            timeout=s.llm_timeout_s,
        )
        logger.info("LLM: MiniMax %s at %s", s.llm_minimax_model, s.llm_minimax_base_url)
    elif s.llm_ollama_url:
        from lean.llm.openai_compatible import OpenAICompatibleLLM

        _llm = OpenAICompatibleLLM(
            base_url=s.llm_ollama_url,
            model=s.llm_ollama_model,
            api_key="",
            timeout=s.llm_timeout_s,
        )
        logger.info("LLM: Ollama %s at %s", s.llm_ollama_model, s.llm_ollama_url)
    else:
        _llm = None
        logger.info("LLM: none configured (no contextual retrieval or query transforms)")

    return _llm
