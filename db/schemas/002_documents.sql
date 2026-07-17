-- 002_documents.sql — document metadata table
-- One row per ingested PDF. Deduplication via source_sha256.

create table if not exists public.documents (
    id                    uuid primary key default uuid_generate_v4(),
    source_path           text not null,
    source_sha256         text not null unique,
    title                 text,
    authors               text[]   default '{}',
    publisher             text,
    year                  int,
    page_count            int,
    extraction_method     text not null
        check (extraction_method in ('unlimited_ocr', 'markitdown')),
    source_storage_path   text not null,
    markdown_storage_path text not null,
    markdown_content      text,
    ingested_at           timestamptz not null default now(),
    reingested_at         timestamptz,
    metadata              jsonb not null default '{}'::jsonb
);

comment on table public.documents is
    'One row per ingested PDF in the Lean Six Sigma corpus.';
comment on column public.documents.source_sha256 is
    'SHA-256 of the original PDF bytes; unique dedup key.';
comment on column public.documents.extraction_method is
    'Which extractor produced the markdown: unlimited_ocr (vLLM) or markitdown (fallback).';
