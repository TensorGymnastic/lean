# 3080 Ti vLLM Setup Guide

One-time setup to run `baidu/Unlimited-OCR` via vLLM on the remote 3080 Ti server.

## Prerequisites

- SSH access to the 3080 Ti host
- NVIDIA driver installed (`nvidia-smi` works)
- Python 3.12+
- 12GB+ VRAM (3080 Ti has 12GB — fits Unlimited-OCR's ~6GB footprint)

## Step 1: Install vLLM

```bash
ssh user@3080ti-host
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install vllm
```

Alternatively, use the official Docker image:

```bash
docker pull vllm/vllm-openai:latest
```

## Step 2: Set HuggingFace token

```bash
export HF_TOKEN=your-hf-token-here
```

## Step 3: Start vLLM serving Unlimited-OCR

```bash
vllm serve baidu/Unlimited-OCR \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 32768
```

Wait for `Application startup complete` (30-60s cold start).

## Step 4: Verify from the local host

```bash
curl http://3080ti-host:8000/health
curl http://3080ti-host:8000/v1/models
```

## Step 5: Configure lean

Set in `.env`:

```
VLLM_BASE_URL=http://3080ti-host:8000
```

## Optional: systemd service for persistence

Create `/etc/systemd/system/vllm-unlimited-ocr.service`:

```ini
[Unit]
Description=vLLM serving Unlimited-OCR
After=network.target

[Service]
Type=simple
User=youruser
Environment=HF_TOKEN=your-token
ExecStart=/home/youruser/vllm-env/bin/vllm serve baidu/Unlimited-OCR --host 0.0.0.0 --port 8000 --trust-remote-code --max-model-len 32768
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-unlimited-ocr
```

## Troubleshooting

- **CUDA version mismatch**: vLLM needs CUDA 12.x. Check `nvidia-smi` shows CUDA 12.0+. If driver is too old, upgrade it.
- **OOM on startup**: Unlimited-OCR needs ~6GB VRAM. Close other GPU processes (`nvidia-smi`).
- **`trust-remote-code` error**: Ensure `--trust-remote-code` flag is passed. The model has custom Python code.
