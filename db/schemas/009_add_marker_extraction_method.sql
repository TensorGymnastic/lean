-- Add 'marker' to the extraction_method CHECK constraint (migration 009)
alter table public.documents drop constraint if exists documents_extraction_method_check;
alter table public.documents add constraint documents_extraction_method_check
    check (extraction_method in ('marker', 'unlimited_ocr', 'markitdown'));
