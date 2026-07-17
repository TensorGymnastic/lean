-- Drop unused markdown_content column (markdown stored in Supabase Storage, not DB)
alter table public.documents drop column if exists markdown_content;
