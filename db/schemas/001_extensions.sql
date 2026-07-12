-- 001_extensions.sql — required Postgres extensions for lean
-- Applied first; all subsequent schema depends on these.

create extension if not exists vector;        -- pgvector for 1024-dim cosine search
create extension if not exists "uuid-ossp";   -- uuid_generate_v4() for primary keys
create extension if not exists pg_trgm;       -- trigram index for section_path substring search
