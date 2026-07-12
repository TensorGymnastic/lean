#!/usr/bin/env bash
# reingest-all.sh — Reingest all documents with Unlimited-OCR (after 3080 Ti is up).
#
# Usage:
#   ./scripts/reingest-all.sh
#
# Prerequisites:
#   - .env configured with VLLM_BASE_URL pointing to the 3080 Ti
#   - vLLM serving Unlimited-OCR and healthy (make vllm-health)

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Checking vLLM health ==="
./scripts/smoke-vllm.sh

echo "=== Reingesting all documents ==="
uv run python -c "
import asyncio
from lean.mcp_server.tools import list_documents, reingest

async def main():
    docs = await list_documents()
    print(f'Found {len(docs)} documents to reingest.')
    for d in docs:
        print(f'  reingesting: {d.title or d.source_path}...')
        result = await reingest(d.id)
        print(f'    → {result.chunk_count} chunks, {result.elapsed_seconds:.1f}s, {result.extraction_method}')
        if result.warnings:
            print(f'    warnings: {result.warnings}')

asyncio.run(main())
print('=== REINGEST COMPLETE ===')
"
