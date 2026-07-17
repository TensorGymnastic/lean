"""Liquid LMF2.5-Embedding-350M wrapper using sentence-transformers.

Uses asymmetric prompts: ``prompt_name="query"`` for queries,
``"document"`` for passages. Returns normalized 1024-dim vectors so
dot product equals cosine similarity.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

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
        model: str = "LiquidAI/LFM2.5-Embedding-350M",
        hf_token: str | None = None,
        device: str = "cpu",
        dim: int = 1024,
    ) -> None:
        logger.info("loading embedding model %s on %s", model, device)
        self._model = SentenceTransformer(
            model,
            trust_remote_code=True,
            device=device,
            token=hf_token,
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
