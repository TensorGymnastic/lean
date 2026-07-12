-- seed.sql — create storage buckets for source PDFs and extracted markdown.
-- Run after schema migrations. Idempotent.

insert into storage.buckets (id, name, public)
values
    ('sources', 'sources', false),
    ('markdown', 'markdown', false)
on conflict (id) do nothing;
