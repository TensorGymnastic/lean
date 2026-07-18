# Marker Server Deployment

Pure-Python HTTP wrapper around `marker-pdf`'s `PdfConverter`. Runs on the
GPU host, accepts raw PDF bytes, returns JSON with markdown + base64 images.

**44× faster than local CPU** (14s vs 626s for a 29-page PDF on RTX 3080 Ti).

## Prerequisites

- NVIDIA GPU (12GB+ VRAM)
- Python 3.12+
- `marker-pdf` installed: `pip install marker-pdf`
- CUDA-enabled PyTorch (marker needs GPU for surya OCR + texify)

## Deployment

### 1. Copy the server script

From the lean repo root:

```bash
scp scripts/marker_server.py gpu-host:/opt/marker-serve/server.py
```

### 2. Create a systemd service

`/etc/systemd/system/marker-serve.service`:

```ini
[Unit]
Description=Marker PDF Extraction Server
After=network.target

[Service]
Type=simple
User=serve
WorkingDirectory=/opt/marker-serve
ExecStart=/opt/marker-serve/.venv/bin/python server.py --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

### 3. Enable + start

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now marker-serve
sudo systemctl status marker-serve
```

### 4. Verify

```bash
curl http://gpu-host:8000/health
# → {"status":"ok"}

curl -X POST --data-binary @test.pdf http://gpu-host:8000/extract | python -m json.tool | head
```

## Configure lean

In `src/lean/config/config.yaml`:

```yaml
marker:
  force_ocr: false
  remote_url: "http://gpu-host:8000"   # empty = run locally on CPU
```

## Operational notes

- **Singleton converter**: models load once on first request (~30s), stay in
  GPU memory for subsequent requests. First request is slow; rest are fast.
- **Single-threaded**: `HTTPServer` processes one request at a time. Concurrent
  ingests serialize. Acceptable for single-operator use; for parallel ingest,
  run multiple instances behind a load balancer.
- **No auth**: assumes LAN-only trust. For external exposure, front with a
  reverse proxy (Caddy/Traefik) with bearer auth or mTLS.
- **No retry**: lean's client has a 600s timeout. A network blip fails the
  extraction and falls through to OCR/markitdown per the 3-stage pipeline.
- **Temp files**: written to `/tmp/`, cleaned up in a `finally` block.

## API

### `POST /extract`

Request body: raw PDF bytes (`Content-Type: application/pdf` or no content-type).

Response 200:
```json
{
  "markdown": "# Extracted text...",
  "page_count": 29,
  "images": {"_page_0_Picture_1": "<base64 PNG>", ...}
}
```

Response 500:
```json
{"error": "extraction failed: <reason>"}
```

### `GET /health`

Returns `{"status":"ok"}`. Use for healthchecks.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| First request takes 30s+ | Model loading | Normal; singleton warmup |
| OOM on GPU | Other process using VRAM | `nvidia-smi` to check, free up |
| 500 on every request | marker-pdf not installed | `pip install marker-pdf` |
| 500 "CUDA out of memory" | Batch too large | Reduce `force_ocr` or restart service |
| lean falls through to OCR | Network timeout | Check `marker.remote_url` + GPU host reachable |
