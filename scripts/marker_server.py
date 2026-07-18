#!/usr/bin/env python3
"""Marker PDF extraction server — runs marker-pdf on GPU.

Pure Python (stdlib http.server + marker-pdf). Singleton PdfConverter
stays loaded in GPU memory after first call. Accepts raw PDF bytes via
POST /extract, returns JSON with markdown + base64-encoded images.

Deployment: run on the GPU host. Configure lean via ``marker.remote_url``
in config.yaml to point at this server.

Usage:
    python3 scripts/marker_server.py [--host 0.0.0.0] [--port 8000]

Requires marker-pdf installed in the environment:
    pip install marker-pdf   # or: uv pip install marker-pdf

Health check:
    curl http://<host>:<port>/health   # → {"status":"ok"}

Extract:
    curl -X POST --data-binary @paper.pdf http://<host>:<port>/extract
    # → {"markdown": "...", "page_count": N, "images": {"_page_0_Picture_1": "<b64 png>"}}
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("marker-serve")

_converter = None


def get_converter(force_ocr: bool = False):
    """Singleton PdfConverter — models load once, stay in GPU memory."""
    global _converter
    if _converter is None:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        logger.info("loading marker models (first call, ~30s)…")
        config = ConfigParser({"output_format": "markdown", "force_ocr": force_ocr})
        _converter = PdfConverter(
            config=config.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config.get_processors(),
            renderer=config.get_renderer(),
        )
        logger.info("marker models loaded")
    return _converter


class MarkerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/extract":
            self.send_error(404, "not found")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_error(400, "empty body")
            return

        pdf_bytes = self.rfile.read(content_length)
        logger.info("received %d bytes, extracting…", len(pdf_bytes))

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            temp_path = f.name

        try:
            from marker.output import text_from_rendered

            converter = get_converter()
            rendered = converter(temp_path)
            text, _, images = text_from_rendered(rendered)

            meta = rendered.metadata if hasattr(rendered, "metadata") else {}
            page_count = len(meta.get("page_stats", [1])) if isinstance(meta, dict) else 1

            images_b64: dict[str, str] = {}
            for name, img in images.items():
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                images_b64[name] = base64.b64encode(buf.getvalue()).decode()

            response = json.dumps(
                {"markdown": text, "page_count": page_count, "images": images_b64}
            ).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            logger.info(
                "extracted: %d pages, %d chars, %d images",
                page_count,
                len(text),
                len(images_b64),
            )
        except Exception as e:
            logger.error("extraction failed: %s", e, exc_info=True)
            error = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error)))
            self.end_headers()
            self.wfile.write(error)
        finally:
            os.unlink(temp_path)

    def do_GET(self):
        if self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Marker PDF extraction server")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 — LAN server by design
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR on all pages")
    args = parser.parse_args()

    get_converter(force_ocr=args.force_ocr)
    server = HTTPServer((args.host, args.port), MarkerHandler)
    logger.info("marker-serve listening on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
