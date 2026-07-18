-- Drop vestigial columns that were never wired up:
--   query_logs.agent_id       — plumbed through services/search.py + store/analytics.py
--                                but no transport (CLI/MCP/REST) ever populated it.
--   documents.source_storage_path  — write-only (zero SELECTs); intended for Supabase
--   documents.markdown_storage_path  Storage upload that was never implemented. Markdown
--                                is reconstructed from chunks by services/corpus.py.
alter table public.query_logs drop column if exists agent_id;
alter table public.documents drop column if exists source_storage_path;
alter table public.documents drop column if exists markdown_storage_path;
