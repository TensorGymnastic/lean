-- Add text_file and web_url extraction methods for the code and web domains.
-- Previously these were recorded as 'markitdown', corrupting provenance data.

-- Drop and re-add the CHECK constraint to include new values.
alter table public.documents drop constraint if exists documents_extraction_method_check;
alter table public.documents add constraint documents_extraction_method_check
    check (extraction_method in ('unknown', 'marker', 'unlimited_ocr', 'markitdown', 'text_file', 'web_url'));

-- Backfill: text files previously stored as 'markitdown' that are not PDFs.
update documents
set extraction_method = 'text_file'
where extraction_method = 'markitdown'
  and source_path not like '%.pdf';

-- Backfill: URLs previously stored as 'markitdown'.
update documents
set extraction_method = 'web_url'
where extraction_method = 'markitdown'
  and (source_path like 'http://%' or source_path like 'https://%');
