"""Query transformation techniques that improve retrieval recall.

Both are optional and require an LLM client. When no LLM is configured,
they return the original query unchanged.

- HyDE: generates a hypothetical answer document, embeds that instead of
  the raw query (Gao et al., arxiv 2212.10496).
- Multi-query: generates N paraphrased queries, each used for retrieval,
  then results are fused via RRF (LangChain MultiQueryRetriever pattern).
"""

from __future__ import annotations

import logging

from lean.llm.base import LLMClient

logger = logging.getLogger(__name__)

_HYDE_PROMPT = """Please write a passage to answer the question.
Try to include as many key details as possible.

{query}

Passage:"""

_MULTI_QUERY_PROMPT = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:"
)


def hyde_transform(query: str, llm: LLMClient, *, max_tokens: int, temperature: float) -> str:
    """Generate a hypothetical answer document for the query.

    The hypothetical doc is embedded instead of the raw query for the
    dense (vector) search leg. The raw query is still used for BM25.

    Returns the generated passage text. Falls back to the original
    query on any error.
    """
    try:
        passage = llm.generate(
            _HYDE_PROMPT.format(query=query),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if passage.strip():
            return passage
    except Exception:
        logger.warning("HyDE failed, using raw query", exc_info=True)
    return query


def multi_query_transform(
    query: str,
    llm: LLMClient,
    *,
    num_queries: int,
    max_tokens: int,
    temperature: float,
) -> list[str]:
    """Generate paraphrased queries for multi-query retrieval.

    Returns a list of queries (original + generated paraphrases).
    Falls back to [query] on any error.
    """
    try:
        raw = llm.generate(
            _MULTI_QUERY_PROMPT.format(query=query, num_queries=num_queries - 1),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        generated = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        if len(generated) > num_queries - 1:
            generated = generated[: num_queries - 1]
        queries = [query] + generated
        logger.info("multi-query: generated %d variants for: %s", len(queries), query[:60])
        return queries
    except Exception:
        logger.warning("multi-query failed, using single query", exc_info=True)
        return [query]
