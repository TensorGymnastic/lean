#!/usr/bin/env bash
# setup-3080ti.sh — Automated vLLM + Unlimited-OCR setup for the 3080 Ti.
#
# Usage from your local machine:
#   ssh user@3080ti-host 'bash -s' < scripts/setup-3080ti.sh
#
# Or directly on the 3080 Ti:
#   bash scripts/setup-3080ti.sh
#
# Prerequisites on the 3080 Ti:
#   - NVIDIA driver installed (nvidia-smi works)
#   - Python 3.10+
#   - HF_TOKEN environment variable set, or pass --hf-token=xxx

set -euo pipefail

HF_TOKEN_VAL="${HF_TOKEN:-}"
PORT=8000

for arg in "$@"; do
    case "$arg" in
        --hf-token=*) HF_TOKEN_VAL="${arg#*=}" ;;
        --port=*) PORT="${arg#*=}" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

if [ -z "$HF_TOKEN_VAL" ]; then
    echo "ERROR: HF_TOKEN not set. Pass --hf-token=xxx or export HF_TOKEN=xxx"
    exit 1
fi

echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    echo "ERROR: nvidia-smi failed. Install NVIDIA driver first."
    exit 1
}

echo "=== Python Check ==="
python3 --version || { echo "ERROR: Python 3 required"; exit 1; }

echo "=== Creating vLLM Virtual Environment ==="
python3 -m venv "$HOME/vllm-lean-env"
# shellcheck disable=SC1091
source "$HOME/vllm-lean-env/bin/activate"

echo "=== Installing vLLM ==="
pip install --upgrade pip
pip install vllm

echo "=== Pre-downloading baidu/Unlimited-OCR ==="
HF_TOKEN="$HF_TOKEN_VAL" python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('baidu/Unlimited-OCR', token='$HF_TOKEN_VAL')
print('Model downloaded successfully.')
" || echo "WARNING: Pre-download failed. vLLM will download on first serve."

echo "=== Creating systemd Service ==="
sudo tee /etc/systemd/system/vllm-unlimited-ocr.service > /dev/null << EOF
[Unit]
Description=vLLM serving baidu/Unlimited-OCR for lean corpus
After=network.target

[Service]
Type=simple
User=$(whoami)
Environment=HF_TOKEN=$HF_TOKEN_VAL
ExecStart=$HOME/vllm-lean-env/bin/vllm serve baidu/Unlimited-OCR \\
    --host 0.0.0.0 \\
    --port $PORT \\
    --trust-remote-code \\
    --max-model-len 32768
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable vllm-unlimited-ocr

echo "=== Starting vLLM Service ==="
sudo systemctl start vllm-unlimited-ocr

echo "=== Waiting for vLLM to become healthy (may take 60-120s on first start) ==="
for i in $(seq 1 24); do
    if curl -fsS --max-time 5 "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "vLLM is healthy after ${i}x5s waits!"
        break
    fi
    echo "  attempt $i/24: not ready yet..."
    sleep 5
done

if curl -fsS --max-time 5 "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo ""
    echo "=== SUCCESS ==="
    echo "vLLM serving baidu/Unlimited-OCR on port $PORT"
    echo ""
    echo "On your local machine, set in .env:"
    echo "  VLLM_BASE_URL=http://$(hostname -I | awk '{print $1}'):$PORT"
    echo "  HF_TOKEN=$HF_TOKEN_VAL"
    echo ""
    echo "Then run: make vllm-health"
else
    echo ""
    echo "=== vLLM NOT YET HEALTHY ==="
    echo "Check logs: sudo journalctl -u vllm-unlimited-ocr -f"
    echo "The model may still be loading (3B params, cold start 60-120s)."
    echo "Wait 2 minutes then run: curl http://localhost:$PORT/health"
fi
