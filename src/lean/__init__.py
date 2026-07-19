"""lean — universal MCP+pgvector corpus server.

A single library, YAML-driven, that builds ingestion + search + retrieval
over any document corpus. Built-in domains include:

    pdf_lss    PDFs (with marker-pdf / OCR / markitdown fallback chain)
    code       Source code / markdown files (no extraction pipeline)
    web        URL-scraped pages

Add a domain by writing a YAML manifest + (optionally) Python adapter
modules referenced by dotted path. See `lean.domains.pdf_lss` for the
canonical example.
"""

from __future__ import annotations

__version__ = "0.2.0"
