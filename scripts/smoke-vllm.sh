#!/usr/bin/env bash
set -euo pipefail
: "${VLLM_BASE_URL:?VLLM_BASE_URL must be set}"
curl -fsS --max-time 10 "${VLLM_BASE_URL%/}/health" >/dev/null
echo "vLLM healthcheck OK: ${VLLM_BASE_URL}"
