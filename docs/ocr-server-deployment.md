# OCR Server Deployment (baidu/Unlimited-OCR via transformers)

## Critical configuration

The OCR server's `server.py` MUST pass `no_repeat_ngram_size=3` to
`model.infer_multi()`. Without it, the model enters infinite repetition
loops (generating the same phrase hundreds of times), producing massive
garbage output that breaks chunking and embedding.

## Server setup (on GPU host)

```bash
mkdir -p ~/ocr-serve && cd ~/ocr-serve
uv init --name ocr-serve
uv add "transformers>=4.57.1,<5" torch accelerate PyMuPDF \
  fastapi uvicorn pillow torchvision addict matplotlib easydict einops

# Download model weights (6.67 GB)
uv run python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('baidu/Unlimited-OCR')"

# Create server.py — key section:
#   result = model.infer_multi(
#       tokenizer, prompt=prompt_text, image_files=image_paths,
#       output_path=output_dir, save_results=False,
#       max_length=32768, temperature=0.0,
#       no_repeat_ngram_size=3,  # CRITICAL: prevents repetition loops
#   )
```

## Ollama embedding model (32K context)

```bash
ollama create lfm2.5-embed-32k -f - << 'EOF'
FROM hf.co/LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0
PARAMETER num_ctx 32768
EOF
```

## Systemd services

Both run as systemd services on the GPU host:

```bash
# OCR server (port 8001)
sudo systemctl status ocr-serve

# Ollama (port 11434)
sudo systemctl status ollama
```

## Firewall

```bash
sudo ufw allow from 192.168.2.0/24 to any port 8000  # OCR
sudo ufw allow from 192.168.2.0/24 to any port 11434 # Ollama
```

## Known issues

- OCR speed: ~5 sec/page with batching (20 pages/batch)
- Large PDFs (300+ pages) require page batching to avoid HTTP 500
- vLLM does not support `UnlimitedOCRForCausalLM` architecture in any version
- Transformers-based serving is ~10x slower than vLLM but the only option
