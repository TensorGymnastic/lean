-- 012_add_chunk_type_check.sql — enforce chunk_type values at the DB level.
-- The application layer validates via Literal["text", "image"], but a CHECK
-- constraint catches direct inserts, typos in migrations, and drift across
-- services that bypass the Pydantic schema.

do $$
begin
    -- Drop any existing constraint first (idempotent for re-runs).
    alter table public.chunks drop constraint if exists chunks_chunk_type_check;
    -- Add the CHECK constraint.
    alter table public.chunks
        add constraint chunks_chunk_type_check
        check (chunk_type in ('text', 'image'));
exception
    when others then
        raise notice 'constraint already satisfied or table missing: %', sqlerrm;
end $$;

comment on constraint chunks_chunk_type_check on public.chunks is
    'Restricts chunk_type to ''text'' or ''image'' — prevents typos and invalid values.';
