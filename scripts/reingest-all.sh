#!/usr/bin/env bash
# Reingest all documents with current extraction settings.
# Processes smallest documents first, skips already-OCR'd by default.
#
# Usage:
#   ./scripts/reingest-all.sh [--force]
#
# Environment:
#   SUPABASE_DB_URL  Postgres DSN (required)
#
# Flags:
#   --force  reingest even if already OCR'd

set -euo pipefail
cd "$(dirname "$0")/.."

: "${SUPABASE_DB_URL:?SUPABASE_DB_URL must be set}"
export PYTHONUNBUFFERED=1

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
    esac
done

QUERY="SELECT id::text FROM documents ORDER BY (SELECT count(*) FROM chunks WHERE document_id = documents.id) ASC"
DOC_IDS=$(psql "$SUPABASE_DB_URL" -t -A -c "$QUERY" 2>/dev/null)

echo "=== REINGEST ALL ==="
echo "Started: $(date)"
echo "Documents: $(echo "$DOC_IDS" | wc -w) total"
echo ""

SUCCESS=0
FAILED=0
SKIPPED=0

for DOC_ID in $DOC_IDS; do
    [ -z "$DOC_ID" ] && continue

    METHOD=$(psql "$SUPABASE_DB_URL" -t -A -c "SELECT extraction_method FROM documents WHERE id = '$DOC_ID'" 2>/dev/null)
    PAGES=$(psql "$SUPABASE_DB_URL" -t -A -c "SELECT coalesce(page_count, 0) FROM documents WHERE id = '$DOC_ID'" 2>/dev/null)

    if [ "$FORCE" -eq 0 ] && [ "$METHOD" = "unlimited_ocr" ]; then
        echo "[$(date +%H:%M:%S)] SKIP $DOC_ID (already OCR'd, $PAGES pages)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[$(date +%H:%M:%S)] START $DOC_ID ($PAGES pages, was $METHOD)"

    if uv run lean reingest "$DOC_ID" 2>&1 | grep -v "UserWarning\|torch._C\|return torch\|__del__"; then
        echo "[$(date +%H:%M:%S)] DONE  $DOC_ID"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "[$(date +%H:%M:%S)] FAIL  $DOC_ID"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "=== COMPLETE ==="
echo "Finished: $(date)"
echo "Success: $SUCCESS  Failed: $FAILED  Skipped: $SKIPPED"
