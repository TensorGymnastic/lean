"""Integration tests for LLM features against real MiniMax API and local Ollama.

These tests call real LLM endpoints and are marked as 'integration'.
Run with: uv run pytest -m integration tests/test_llm_integration.py -v

Requirements:
  - MINIMAX_API_KEY in .env (or environment)
  - Remote Ollama at http://192.168.2.37:11434 with LFM2.5-8B-A1B loaded
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lean.chunker.markdown_ast import Section
from lean.chunker.recursive import ChunkResult

MINIMAX_KEY = os.environ.get("MINIMAX_API_KEY", "")
OLLAMA_URL = "http://192.168.2.37:11434"
OLLAMA_MODEL = "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q6_K"

pytestmark = pytest.mark.integration


def _load_env_minimax_key() -> str:
    """Load MINIMAX_API_KEY from .env file if not in environment."""
    if MINIMAX_KEY:
        return MINIMAX_KEY
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("MINIMAX_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


_MINIMAX_KEY = _load_env_minimax_key()

minimax_skip = pytest.mark.skipif(
    not _MINIMAX_KEY,
    reason="MINIMAX_API_KEY not set",
)

ollama_skip = pytest.mark.skipif(
    True,
    reason="Set OLLAMA_URL to run Ollama integration tests",
)


def _make_minimax():
    from lean.llm.openai_compatible import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        base_url="https://api.minimax.io",
        model="MiniMax-Text-01",
        api_key=_MINIMAX_KEY,
        timeout=30.0,
    )


def _make_ollama():
    from lean.llm.openai_compatible import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        api_key="",
        timeout=60.0,
    )


def _make_test_chunks() -> tuple[list[ChunkResult], list[Section]]:
    sections = [
        Section(
            path="Chapter 2 > DMAIC",
            level=2,
            heading="DMAIC Methodology",
            content=(
                "DMAIC is the core methodology of Six Sigma. It consists of five phases: "
                "Define, Measure, Analyze, Improve, and Control. The Define phase identifies "
                "the problem and project scope. Measure collects baseline data. Analyze "
                "identifies root causes. Improve implements solutions. Control sustains gains."
            ),
        )
    ]
    chunks = [
        ChunkResult(
            section_path="Chapter 2 > DMAIC",
            heading_text="DMAIC Methodology",
            chunk_index=0,
            content=(
                "DMAIC is the core methodology of Six Sigma. "
                "Define, Measure, Analyze, Improve, Control."
            ),
            token_count=20,
        )
    ]
    return chunks, sections


# ── MiniMax Tests ──────────────────────────────────────────────


@minimax_skip
def test_minimax_generate_returns_text() -> None:
    """MiniMax generate() returns a non-empty string."""
    llm = _make_minimax()
    result = llm.generate("What is kaizen? Answer in one sentence.", max_tokens=100)
    assert len(result) > 10
    assert any(kw in result.lower() for kw in ["improv", "continuous", "change", "japan"])


@minimax_skip
def test_minimax_hyde_returns_passage() -> None:
    """HyDE generates a hypothetical document about the query topic."""
    from lean.retrieval.query_transform import hyde_transform

    llm = _make_minimax()
    passage = hyde_transform("What is DMAIC in Lean Six Sigma?", llm)
    assert len(passage) > 50
    assert any(kw in passage.lower() for kw in ["dmaic", "define", "measure", "analyze", "control"])


@minimax_skip
def test_minimax_multi_query_returns_variants() -> None:
    """Multi-query generates paraphrased queries."""
    from lean.retrieval.query_transform import multi_query_transform

    llm = _make_minimax()
    queries = multi_query_transform("What is DMAIC?", llm, num_queries=4)
    assert len(queries) == 4
    assert queries[0] == "What is DMAIC?"
    assert all(len(q) > 5 for q in queries)


@minimax_skip
def test_minimax_contextual_adds_context() -> None:
    """Contextual retrieval prepends LLM-generated context to chunks."""
    from lean.extraction.contextual import add_context_to_chunks

    llm = _make_minimax()
    chunks, sections = _make_test_chunks()
    result = add_context_to_chunks(chunks, sections, llm)
    assert len(result) == 1
    assert len(result[0].content) > len(chunks[0].content)
    assert "DMAIC" in result[0].content or "Six Sigma" in result[0].content


# ── Ollama LFM2.5-8B Tests ─────────────────────────────────────


def test_ollama_lfm_generate_returns_text() -> None:
    """Ollama LFM2.5-8B generate() returns a non-empty string."""
    try:
        llm = _make_ollama()
        result = llm.generate("What is kaizen? Answer in one sentence.", max_tokens=800)
        assert len(result) > 10
        assert any(
            kw in result.lower() for kw in ["improv", "continuous", "change", "japan", "small"]
        )
    except Exception as e:
        if "connect" in str(e).lower() or "refused" in str(e).lower():
            pytest.skip(f"Ollama not reachable: {e}")
        raise


def test_ollama_lfm_hyde_returns_passage() -> None:
    """HyDE via Ollama generates a passage about the query topic."""
    from lean.retrieval.query_transform import hyde_transform

    try:
        llm = _make_ollama()
        passage = hyde_transform("What is DMAIC?", llm)
        assert len(passage) > 30
        assert any(
            kw in passage.lower() for kw in ["dmaic", "define", "measure", "methodology", "sigma"]
        )
    except Exception as e:
        if "connect" in str(e).lower() or "refused" in str(e).lower():
            pytest.skip(f"Ollama not reachable: {e}")
        raise


def test_ollama_lfm_multi_query_returns_variants() -> None:
    """Multi-query via Ollama generates paraphrased queries."""
    from lean.retrieval.query_transform import multi_query_transform

    try:
        llm = _make_ollama()
        queries = multi_query_transform("What is DMAIC?", llm, num_queries=3)
        assert len(queries) >= 2
        assert queries[0] == "What is DMAIC?"
    except Exception as e:
        if "connect" in str(e).lower() or "refused" in str(e).lower():
            pytest.skip(f"Ollama not reachable: {e}")
        raise


def test_ollama_lfm_contextual_adds_context() -> None:
    """Contextual retrieval via Ollama adds context to chunks."""
    from lean.extraction.contextual import add_context_to_chunks

    try:
        llm = _make_ollama()
        chunks, sections = _make_test_chunks()
        result = add_context_to_chunks(chunks, sections, llm)
        assert len(result) == 1
        assert len(result[0].content) >= len(chunks[0].content)
    except Exception as e:
        if "connect" in str(e).lower() or "refused" in str(e).lower():
            pytest.skip(f"Ollama not reachable: {e}")
        raise
