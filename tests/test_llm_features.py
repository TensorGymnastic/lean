"""Tests for Contextual Retrieval and query transforms."""

from __future__ import annotations

from lean.chunker.markdown_ast import Section
from lean.chunker.recursive import ChunkResult


class FakeLLM:
    """Deterministic LLM mock for testing."""

    def __init__(self, response: str = "Context: This chunk discusses DMAIC methodology.") -> None:
        self._response = response
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        self.call_count += 1
        return self._response


class FailingLLM:
    """LLM that always raises — tests graceful degradation."""

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        raise RuntimeError("LLM unavailable")


def test_contextual_retrieval_prepends_context() -> None:
    """Each chunk gets LLM-generated context prepended."""
    from lean.extraction.contextual import add_context_to_chunks

    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="Full document text here.")]
    chunks = [
        ChunkResult(
            section_path="Ch 1",
            heading_text="Ch 1",
            chunk_index=0,
            content="Raw chunk A",
            token_count=10,
        ),
        ChunkResult(
            section_path="Ch 1",
            heading_text="Ch 1",
            chunk_index=1,
            content="Raw chunk B",
            token_count=10,
        ),
    ]
    llm = FakeLLM(response="Context for this chunk.")

    result = add_context_to_chunks(chunks, sections, llm)

    assert len(result) == 2
    assert result[0].content.startswith("Context for this chunk.")
    assert "Raw chunk A" in result[0].content
    assert result[1].content.startswith("Context for this chunk.")
    assert "Raw chunk B" in result[1].content
    assert llm.call_count == 2


def test_contextual_retrieval_falls_back_on_error() -> None:
    """When LLM fails, chunk content is unchanged (graceful degradation)."""
    from lean.extraction.contextual import add_context_to_chunks

    sections = [Section(path="Ch 1", level=1, heading="Ch 1", content="Doc text.")]
    chunks = [
        ChunkResult(
            section_path="Ch 1",
            heading_text="Ch 1",
            chunk_index=0,
            content="Important content",
            token_count=5,
        ),
    ]

    result = add_context_to_chunks(chunks, sections, FailingLLM())

    assert result[0].content == "Important content"


def test_hyde_transform_returns_passage() -> None:
    """HyDE returns the LLM-generated hypothetical document."""
    from lean.retrieval.query_transform import hyde_transform

    llm = FakeLLM(response="DMAIC stands for Define, Measure, Analyze, Improve, Control.")
    result = hyde_transform("What is DMAIC?", llm)

    assert "DMAIC" in result
    assert result != "What is DMAIC?"


def test_hyde_transform_falls_back_on_error() -> None:
    """HyDE returns original query when LLM fails."""
    from lean.retrieval.query_transform import hyde_transform

    result = hyde_transform("What is DMAIC?", FailingLLM())
    assert result == "What is DMAIC?"


def test_multi_query_returns_original_plus_variants() -> None:
    """Multi-query returns the original query plus generated variants."""
    from lean.retrieval.query_transform import multi_query_transform

    llm = FakeLLM(response="What does DMAIC mean?\nExplain DMAIC methodology\nDMAIC process steps")
    result = multi_query_transform("What is DMAIC?", llm, num_queries=4)

    assert len(result) == 4
    assert result[0] == "What is DMAIC?"
    assert len(result[1:]) == 3


def test_multi_query_falls_back_on_error() -> None:
    """Multi-query returns [original] when LLM fails."""
    from lean.retrieval.query_transform import multi_query_transform

    result = multi_query_transform("What is DMAIC?", FailingLLM())
    assert result == ["What is DMAIC?"]


def test_multi_query_trims_excess() -> None:
    """When LLM returns more lines than requested, trim to num_queries-1."""
    from lean.retrieval.query_transform import multi_query_transform

    llm = FakeLLM(response="Q1\nQ2\nQ3\nQ4\nQ5\nQ6\nQ7\nQ8")
    result = multi_query_transform("original", llm, num_queries=4)

    assert len(result) == 4
    assert result[0] == "original"
