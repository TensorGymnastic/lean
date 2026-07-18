# Supabase (local development)

Lean runs Supabase **locally via Docker** — no cloud account, zero cost.
pgvector is pre-installed.

## What lean uses

- **`docker compose up -d supabase-db`** — start Postgres with pgvector
- **`make db-init`** — apply SQL migrations from `db/schemas/`
- **`SUPABASE_DB_URL`** — direct Postgres DSN (not pooler)

Lean does **not** use Supabase Auth, Storage, Realtime, or Edge Functions.
The vestigial `supabase/config.toml [storage]` block and `storage.*`
path-prefix columns are not wired up — see [`limitations.md`](../limitations.md#dedup-is-by-sha-256-not-by-path).

## Local stack (Docker Compose)

```yaml
services:
  supabase-db:
    image: supabase/postgres:15.6.1.143
    ports:
      - "54322:5432"
    environment:
      POSTGRES_PASSWORD: postgres
    volumes:
      - ./db/schemas:/docker-entrypoint-initdb.d  # runs on FIRST init only
```

The `db/schemas` bind-mount only runs on first DB init. For subsequent
schema changes, run `make db-init` explicitly.

## Migration workflow

Lean uses **plain SQL files** (`db/schemas/001_extensions.sql` through
`007_drop_markdown_content.sql`), applied by `lean db-init`. Not the
Supabase CLI migration system.

```bash
# Apply all migrations (idempotent — CREATE TABLE IF NOT EXISTS)
make db-init

# Wipe and re-apply from scratch
make db-reset  # runs `supabase db reset`
```

## CLI (when you need it)

The Supabase CLI is useful for inspecting the local stack, generating
TypeScript types, or deploying to a remote project. Install:

```bash
# macOS
brew install supabase/tap/supabase

# Linux / WSL
curl -fsSL https://raw.githubusercontent.com/supabase/cli/main/install.sh | sh
```

Useful commands (not currently wired into lean):

```bash
supabase init          # scaffold a supabase/ config dir
supabase start         # start full local stack (Postgres + Auth + Storage + Studio)
supabase migration up  # apply pending migrations
supabase db reset      # wipe + re-apply all migrations
supabase gen types     # generate TypeScript types from schema
```

Lean uses raw Postgres DSN (`postgresql://postgres:postgres@127.0.0.1:54322/postgres`),
not the Supabase Python client. We talk to Postgres directly via `psycopg`.

## Gotchas

- **Studio web UI:** `supabase start` brings up a Studio web UI on
  `http://127.0.0.1:54323`. Useful for inspecting data manually.
- **Port conflicts:** default `54322` for Postgres, `54323` for Studio.
  Override via env if needed.
- **Connection pooling:** the local stack doesn't have Supavisor by
  default. For multi-process apps, use a pooler; for lean's single-process
  MCP server, direct connections are fine.
- **Don't commit `.env` with real Supabase keys.** Only the local
  `postgresql://postgres:postgres@127.0.0.1:54322/postgres` is safe to
  document.

## Resources

- Docs: <https://supabase.com/docs/guides/local-development>
- lean compose: `docker-compose.yml`
- lean migrations: `db/schemas/`
- Lean uses pgvector via raw SQL — see [`pgvector.md`](pgvector.md)
