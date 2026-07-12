"""Recursive text chunker that respects section boundaries.

Splits each section's content into chunks between ``target_min`` and
``hard_cap`` tokens. Splits on paragraph boundaries first, then sentence
boundaries, then word boundaries as a last resort.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from lean.chunker.markdown_ast import Section

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class ChunkResult:
    """One chunk with its section context."""

    section_path: str
    heading_text: str
    chunk_index: int  # 0-based within the section
    content: str
    token_count: int


def chunk_sections(
    sections: list[Section],
    *,
    target_min: int,
    target_max: int,
    hard_cap: int,
) -> list[ChunkResult]:
    """Split each section's content into token-bounded chunks.

    Each section produces chunks with ``chunk_index`` starting at 0.
    Sections shorter than ``target_min`` produce a single chunk.
    """
    results: list[ChunkResult] = []
    for section in sections:
        chunks = _split_text(
            section.content,
            target_min=target_min,
            target_max=target_max,
            hard_cap=hard_cap,
        )
        for i, (content, token_count) in enumerate(chunks):
            results.append(
                ChunkResult(
                    section_path=section.path,
                    heading_text=section.heading,
                    chunk_index=i,
                    content=content,
                    token_count=token_count,
                )
            )
    return results


def _split_text(
    text: str, *, target_min: int, target_max: int, hard_cap: int
) -> list[tuple[str, int]]:
    """Split text into chunks of approximately target_max tokens."""
    if not text.strip():
        return []
    paragraphs = text.split("\n\n")
    chunks: list[tuple[str, int]] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)
        if para_tokens > hard_cap:
            # Flush current first
            if current:
                joined = "\n\n".join(current)
                chunks.append((joined, _count_tokens(joined)))
                current = []
                current_tokens = 0
            # Split the big paragraph by sentence
            for sentence_chunk in _split_by_sentences(para, target_max, hard_cap):
                chunks.append(sentence_chunk)
        elif current_tokens + para_tokens > target_max and current:
            joined = "\n\n".join(current)
            chunks.append((joined, _count_tokens(joined)))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        joined = "\n\n".join(current)
        chunks.append((joined, _count_tokens(joined)))
    return chunks


def _split_by_sentences(text: str, target_max: int, hard_cap: int) -> list[tuple[str, int]]:
    """Split a too-long paragraph by sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[tuple[str, int]] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        if sent_tokens > hard_cap:
            # Flush current
            if current:
                joined = " ".join(current)
                chunks.append((joined, _count_tokens(joined)))
                current = []
                current_tokens = 0
            # Last resort: split by words
            for word_chunk in _split_by_words(sent, hard_cap):
                chunks.append(word_chunk)
        elif current_tokens + sent_tokens > target_max and current:
            joined = " ".join(current)
            chunks.append((joined, _count_tokens(joined)))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens
    if current:
        joined = " ".join(current)
        chunks.append((joined, _count_tokens(joined)))
    return chunks


def _split_by_words(text: str, hard_cap: int) -> list[tuple[str, int]]:
    """Hard-split by words when sentences are too long."""
    words = text.split()
    chunks: list[tuple[str, int]] = []
    buf: list[str] = []
    buf_tokens = 0
    for w in words:
        w_tokens = _count_tokens(w)
        if buf_tokens + w_tokens > hard_cap:
            joined = " ".join(buf)
            chunks.append((joined, buf_tokens))
            buf = [w]
            buf_tokens = w_tokens
        else:
            buf.append(w)
            buf_tokens += w_tokens
    if buf:
        joined = " ".join(buf)
        chunks.append((joined, buf_tokens))
    return chunks


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))
