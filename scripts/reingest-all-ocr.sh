#!/usr/bin/env bash
# Batch reingest all documents with Unlimited-OCR.
# Runs sequentially, smallest first, logs progress.
# Usage: nohup ./scripts/reingest-all-ocr.sh > /tmp/ocr-reingest.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

export SUPABASE_DB_URL="${SUPABASE_DB_URL:-postgresql://postgres:postgres@127.0.0.1:54322/postgres}"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONUNBUFFERED=1

# Get all document IDs sorted by chunk count (smallest first)
DOC_IDS=$(docker exec supabase_db_lean psql -U postgres -t -c "
    SELECT d.id::text FROM documents d
    ORDER BY (SELECT count(*) FROM chunks WHERE document_id = d.id) ASC
" 2>/dev/null | xargs)

echo "=== OCR BATCH REINGEST ==="
echo "Started: $(date)"
echo "Documents: $(echo $DOC_IDS | wc -w) total"
echo ""

SUCCESS=0
FAILED=0
SKIPPED=0

for DOC_ID in $DOC_IDS; do
    # Check if already OCR'd
    METHOD=$(docker exec supabase_db_lean psql -U postgres -t -c "
        SELECT extraction_method FROM documents WHERE id = '$DOC_ID'
    " 2>/dev/null | tr -d ' \n')

    if [ "$METHOD" = "unlimited_ocr" ]; then
        echo "[$(date +%H:%M:%S)] SKIP $DOC_ID (already OCR'd)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    PAGES=$(docker exec supabase_db_lean psql -U postgres -t -c "
        SELECT coalesce(page_count, 0) FROM documents WHERE id = '$DOC_ID'
    " 2>/dev/null | tr -d ' \n')

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
