"""Liquid LMF2.5-Embedding-350M wrapper using sentence-transformers.

Uses asymmetric prompts: ``prompt_name="query"`` for queries,
``"document"`` for passages. Returns normalized 1024-dim vectors so
dot product equals cosine similarity.

``sentence_transformers`` is an optional dependency (``uv sync --extra local-models``);
the import is deferred to ``__init__`` so the package remains importable
without it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LiquidLMFEmbedder:
    """Wraps LiquidAI/LFM2.5-Embedding-350M via sentence-transformers.

    The model is 350M params (~700MB BF16) and runs fine on CPU for
    batch sizes typical of RAG ingestion. For high-throughput serving,
    move to GPU (FlashAttention 2 optional).
    """

    def __init__(
        self,
        *,
        model: str,
        hf_token: str | None = None,
        device: str = "cpu",
        dim: int = 1024,
        revision: str,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed; install with "
                "`uv sync --extra local-models` to use LiquidLMFEmbedder"
            ) from e
        logger.info("loading embedding model %s@%s on %s", model, revision[:8], device)
        self._model = SentenceTransformer(
            model,
            trust_remote_code=True,
            device=device,
            token=hf_token,
            revision=revision,
        )
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode passages using the 'document' prompt name."""
        vectors = self._model.encode(
            texts,
            prompt_name="document",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Encode a query using the 'query' prompt name."""
        vec = self._model.encode(
            query,
            prompt_name="query",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return list(vec.tolist())
