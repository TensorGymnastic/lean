-- Add text_file and web_url extraction methods for the code and web domains.
-- Previously these were recorded as 'markitdown', corrupting provenance data.

ALTER TYPE extraction_method ADD VALUE IF NOT EXISTS 'text_file';
ALTER TYPE extraction_method ADD VALUE IF NOT EXISTS 'web_url';

-- Backfill: text files previously stored as 'markitdown' that are not PDFs.
UPDATE documents
SET extraction_method = 'text_file'
WHERE extraction_method = 'markitdown'
  AND source_path NOT LIKE '%.pdf';

-- Backfill: URLs previously stored as 'markitdown'.
UPDATE documents
SET extraction_method = 'web_url'
WHERE extraction_method = 'markitdown'
  AND (source_path LIKE 'http%' OR source_path LIKE 'ftp%');
