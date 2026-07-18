-- Drop unused markdown_content column (markdown is reconstructed from chunks by
-- services/corpus.py:get_document_markdown — no Supabase Storage upload was ever wired).
alter table public.documents drop column if exists markdown_content;
