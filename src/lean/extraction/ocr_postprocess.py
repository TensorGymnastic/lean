"""Post-process Unlimited-OCR output to clean markdown.

The model emits structured annotations alongside text:
  <|det|>type [x1, y1, x2, y2]<|/det|>  — bounding box
  <|ref|>label<|/ref|>                    — reference
  <|grounding|>                            — grounding marker
  <PAGE> / </PAGE>                         — page boundaries
  <image>                                  — image token placeholder

This module strips annotations and returns clean markdown text.
"""

from __future__ import annotations

import re

_DET_RE = re.compile(r"<\|det\|>[^<]*<\|/det\|>")
_REF_RE = re.compile(r"<\|ref\|>[^<]*<\|/ref\|>")
_PAGE_RE = re.compile(r"</?PAGE>")
_GROUNDING_RE = re.compile(r"<\|grounding\|>")
_IMAGE_TOKEN_RE = re.compile(r"<image>")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def clean_ocr_output(raw: str) -> str:
    """Strip Unlimited-OCR annotations and return clean markdown."""
    text = _DET_RE.sub("", raw)
    text = _REF_RE.sub("", text)
    text = _PAGE_RE.sub("", text)
    text = _GROUNDING_RE.sub("", text)
    text = _IMAGE_TOKEN_RE.sub("", text)
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()
