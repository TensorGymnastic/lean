"""Contextual Retrieval — prepend LLM-generated context to each chunk.

Anthropic's "Contextual Retrieval" technique: each chunk is prefixed with
a short LLM-generated context (50-100 tokens) situating it within the
parent document. Reduces retrieval failures by ~49% per Anthropic's
study. Requires an LLM client. No-op when no LLM is configured.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lean.core.chunker.markdown_ast import Section
from lean.core.chunker.recursive import ChunkResult

if TYPE_CHECKING:
    from lean.core.llm.openai_compatible import OpenAICompatibleLLM

logger = logging.getLogger(__name__)

_CONTEXTUAL_PROMPT = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""


def add_context_to_chunks(
    chunks: list[ChunkResult],
    sections: list[Section],
    llm: OpenAICompatibleLLM,
    *,
    max_tokens: int,
    temperature: float,
    max_doc_chars: int = 8000,
) -> list[ChunkResult]:
    """Prepend LLM-generated context to each chunk.

    Returns new ChunkResult objects with context prepended to content.
    Original chunk text is not preserved separately — the contextualized
    version replaces the content for both embedding and BM25 indexing.
    """
    doc_text = "\n\n".join(s.content for s in sections)[:max_doc_chars]
    updated: list[ChunkResult] = []

    for chunk in chunks:
        try:
            context = llm.generate(
                _CONTEXTUAL_PROMPT.format(document=doc_text, chunk=chunk.content[:1000]),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            contextualized = f"{context}\n\n{chunk.content}" if context else chunk.content
        except Exception:
            logger.warning(
                "contextual retrieval failed for chunk %d, using raw text",
                chunk.chunk_index,
                exc_info=True,
            )
            contextualized = chunk.content

        updated.append(
            ChunkResult(
                section_path=chunk.section_path,
                heading_text=chunk.heading_text,
                chunk_index=chunk.chunk_index,
                content=contextualized,
                token_count=chunk.token_count,
            )
        )

    logger.info("contextual retrieval: %d chunks processed", len(updated))
    return updated


__all__ = ["add_context_to_chunks"]
