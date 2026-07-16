#!/usr/bin/env bash
# Health check for the remote OCR server (Unlimited-OCR).
set -euo pipefail
: "${OCR_BASE_URL:?OCR_BASE_URL must be set (e.g. http://gpu-host:8000)}"
curl -fsS --max-time 10 "${OCR_BASE_URL%/}/health" >/dev/null
echo "OCR server healthcheck OK: ${OCR_BASE_URL}"
