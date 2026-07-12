#!/usr/bin/env bash
set -euo pipefail
: "${SUPABASE_DB_URL:?SUPABASE_DB_URL must be set}"
psql "$SUPABASE_DB_URL" -c "select extname from pg_extension where extname='vector'" -t | grep -q vector
echo "pgvector extension OK"
