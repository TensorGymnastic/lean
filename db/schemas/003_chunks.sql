-- 003_chunks.sql — chunk table with pgvector embeddings
-- Each chunk belongs to a document; embedding is 1024-dim (LFM2.5-Embedding-350M).

create table if not exists public.chunks (
    id            uuid primary key default uuid_generate_v4(),
    document_id   uuid not null references public.documents(id) on delete cascade,
    chunk_index   int  not null check (chunk_index >= 0),
    section_path  text not null,
    heading_text  text,
    page_start    int,
    page_end      int,
    token_count   int  not null check (token_count > 0),
    content       text not null,
    embedding     vector(1024) not null,
    unique (document_id, chunk_index)
);

-- ivfflat index for approximate nearest-neighbor cosine search.
-- lists=100 is a reasonable default for up to ~100k chunks; tune if corpus grows.
-- IMPORTANT: ivfflat builds K-means clusters at creation time. For optimal recall,
-- run `REINDEX INDEX chunks_embedding_idx` after bulk ingest.
create index if not exists chunks_embedding_idx
    on public.chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Fast lookup of all chunks for a document (used by list_documents, delete cascade).
create index if not exists chunks_document_id_idx
    on public.chunks (document_id);

-- Trigram index for section_path substring search (e.g. WHERE section_path ILIKE '%DMAIC%').
create index if not exists chunks_section_path_trgm_idx
    on public.chunks using gin (section_path gin_trgm_ops);

comment on table public.chunks is
    'Retrievable text chunks with Liquid LMF 1024-dim embeddings.';
comment on column public.chunks.section_path is
    'Hierarchical location: "Chapter 3 > DMAIC > Measure > Data Collection".';
comment on column public.chunks.embedding is
    '1024-dim normalized vector from LiquidAI/LFM2.5-Embedding-350M (prompt_name="document").';
