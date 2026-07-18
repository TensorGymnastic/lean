-- 010_add_chunk_types.sql — support image/chart chunks alongside text chunks.
-- Adds chunk_type (default 'text') and image_meta (nullable JSONB) so VLM-described
-- images from marker extraction can be stored as searchable chunks.

alter table public.chunks
    add column if not exists chunk_type text not null default 'text';

alter table public.chunks
    add column if not exists image_meta jsonb;

comment on column public.chunks.chunk_type is
    'Chunk type: "text" (default) or "image" (VLM-described chart/figure/diagram).';
comment on column public.chunks.image_meta is
    'Structured metadata for image chunks: {chart_type, title, description, key_data_points, ...}. NULL for text chunks.';
