"""Recursive text chunker that respects section boundaries.

Splits each section's content into chunks bounded by ``hard_cap`` tokens.
Splits on paragraph boundaries first, then sentence boundaries, then word
boundaries as a last resort.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from lean.chunker.markdown_ast import Section

_ENCODINGS: dict[str, tiktoken.Encoding] = {}


def _get_encoding(name: str = "cl100k_base") -> tiktoken.Encoding:
    if name not in _ENCODINGS:
        _ENCODINGS[name] = tiktoken.get_encoding(name)
    return _ENCODINGS[name]


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
    target_max: int,
    hard_cap: int,
    overlap: int = 0,
    encoding: str = "cl100k_base",
) -> list[ChunkResult]:
    """Split each section's content into token-bounded chunks.

    When ``overlap > 0``, adjacent chunks within the same section share
    ``overlap`` tokens of trailing/leading text. This prevents context loss
    at chunk boundaries — a standard RAG technique.
    """
    enc = _get_encoding(encoding)
    results: list[ChunkResult] = []
    for section in sections:
        chunks = _split_text(
            section.content,
            target_max=target_max,
            hard_cap=hard_cap,
            enc=enc,
        )
        if overlap > 0 and len(chunks) > 1:
            chunks = _apply_overlap(chunks, overlap, enc)
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
    text: str,
    *,
    target_max: int,
    hard_cap: int,
    enc: tiktoken.Encoding | None = None,
) -> list[tuple[str, int]]:
    """Split text into chunks of approximately target_max tokens."""
    if enc is None:
        enc = _get_encoding()
    if not text.strip():
        return []
    paragraphs = text.split("\n\n")
    chunks: list[tuple[str, int]] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(enc.encode(para))
        if para_tokens > hard_cap:
            if current:
                joined = "\n\n".join(current)
                chunks.append((joined, len(enc.encode(joined))))
                current = []
                current_tokens = 0
            for sentence_chunk in _split_by_sentences(para, target_max, hard_cap, enc=enc):
                chunks.append(sentence_chunk)
        elif current_tokens + para_tokens > target_max and current:
            joined = "\n\n".join(current)
            chunks.append((joined, len(enc.encode(joined))))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        joined = "\n\n".join(current)
        chunks.append((joined, len(enc.encode(joined))))
    return chunks


def _split_by_sentences(
    text: str,
    target_max: int,
    hard_cap: int,
    *,
    enc: tiktoken.Encoding | None = None,
) -> list[tuple[str, int]]:
    """Split a too-long paragraph by sentence boundaries."""
    if enc is None:
        enc = _get_encoding()
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[tuple[str, int]] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        sent_tokens = len(enc.encode(sent))
        if sent_tokens > hard_cap:
            if current:
                joined = " ".join(current)
                chunks.append((joined, len(enc.encode(joined))))
                current = []
                current_tokens = 0
            for word_chunk in _split_by_words(sent, hard_cap, enc=enc):
                chunks.append(word_chunk)
        elif current_tokens + sent_tokens > target_max and current:
            joined = " ".join(current)
            chunks.append((joined, len(enc.encode(joined))))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens
    if current:
        joined = " ".join(current)
        chunks.append((joined, len(enc.encode(joined))))
    return chunks


def _split_by_words(
    text: str,
    hard_cap: int,
    *,
    enc: tiktoken.Encoding | None = None,
) -> list[tuple[str, int]]:
    """Hard-split by words when sentences are too long."""
    if enc is None:
        enc = _get_encoding()
    words = text.split()
    chunks: list[tuple[str, int]] = []
    buf: list[str] = []
    buf_tokens = 0
    for w in words:
        w_tokens = len(enc.encode(w))
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


def _apply_overlap(
    chunks: list[tuple[str, int]],
    overlap: int,
    enc: tiktoken.Encoding,
) -> list[tuple[str, int]]:
    """Prepend the last ``overlap`` tokens of each chunk to the next chunk.

    This creates sliding windows so that information near chunk boundaries
    appears in both chunks, improving retrieval continuity.
    """
    if len(chunks) <= 1 or overlap <= 0:
        return chunks
    result: list[tuple[str, int]] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1][0]
        prev_tokens = enc.encode(prev_text)
        tail_tokens = prev_tokens[-overlap:] if len(prev_tokens) > overlap else prev_tokens
        tail_text = enc.decode(tail_tokens)
        merged = f"{tail_text}\n\n{chunks[i][0]}"
        merged_tokens = len(enc.encode(merged))
        result.append((merged, merged_tokens))
    return result
