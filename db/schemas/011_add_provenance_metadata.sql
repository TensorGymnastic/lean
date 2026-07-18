-- 011_add_provenance_metadata.sql — production-grade provenance columns.
-- Aligns chunk schema with industry standards (Docling ProvenanceItem,
-- Unstructured CoordinatesMetadata, LlamaIndex ImageNode.hash).
--
-- bbox: bounding box {l, t, r, b, coord_origin} for "click to source" highlighting.
--       NULL until marker block-level polygon extraction is wired (future work).
-- image_hash: SHA-256 of image pixel bytes — dedup + stable IDs across re-ingest.
-- provenance_model: which VLM described this image (e.g. "MiniMax-M3", "Qwen3-VL-8B").
-- embedding_model: which embedder produced the vector — prevents silent re-embed breakage.
-- embedding_dim: vector dimension — must match embedding_model.

alter table public.chunks
    add column if not exists bbox jsonb,
    add column if not exists image_hash text,
    add column if not exists provenance_model text,
    add column if not exists embedding_model text,
    add column if not exists embedding_dim int;

comment on column public.chunks.bbox is
    'Bounding box {l, t, r, b, coord_origin} in PDF coordinate space. NULL until block-level extraction is wired.';
comment on column public.chunks.image_hash is
    'SHA-256 of image pixel bytes. NULL for text chunks. Used for dedup and stable IDs.';
comment on column public.chunks.provenance_model is
    'Model that produced the VLM description (e.g. "MiniMax-M3"). NULL for text chunks.';
comment on column public.chunks.embedding_model is
    'Embedding model name (e.g. "LiquidAI/LFM2.5-Embedding-350M"). NULL for legacy chunks.';
comment on column public.chunks.embedding_dim is
    'Embedding vector dimension. Should match embedding_model. NULL for legacy chunks.';
