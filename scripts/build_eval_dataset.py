"""Build a curated eval dataset from the running corpus.

Queries chunks with substantive headings, cleans marker's HTML artifacts,
strips numbered prefixes, and filters generic sections.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from lean.store.base import StoreConnection
from psycopg.rows import dict_row

_GENERIC_HEADINGS = frozenset(
    {
        "introduction",
        "contents",
        "table of contents",
        "preface",
        "foreword",
        "acknowledgments",
        "about the author",
        "index",
        "references",
        "bibliography",
        "appendix",
        "glossary",
        "conclusion",
        "summary",
        "images",
        "figures",
        "tables",
        "copyright",
        "dedication",
        "errata",
    }
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LEADING_NUM_RE = re.compile(r"^\d+(\.\d+)*\.?\s*")


def _clean_heading(raw: str) -> str | None:
    heading = _HTML_TAG_RE.sub("", raw).strip()
    heading = _LEADING_NUM_RE.sub("", heading)
    if len(heading) < 8 or len(heading) > 80:
        return None
    if heading.lower() in _GENERIC_HEADINGS:
        return None
    if heading.lower().startswith(("chapter ", "figure ", "table ", "image ")):
        return None
    if heading.isdigit():
        return None
    return heading


def _formulate_query(heading: str) -> str:
    lower = heading.lower()
    if "phase" in lower:
        return f"What happens during the {heading.lower()}?"
    if " vs " in lower or "versus" in lower:
        return f"What is the difference: {heading}?"
    if lower.startswith(("how ", "what ", "why ", "when ", "where ")):
        return heading
    return f"What is {heading}?"


def build_dataset(sample_size: int = 30, seed: int = 42) -> list[dict]:
    random.seed(seed)
    conn = StoreConnection.from_env()
    try:
        with conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                select c.id, c.heading_text, c.section_path, c.content, d.title
                from public.chunks c
                join public.documents d on c.document_id = d.id
                where c.heading_text is not null
                  and length(c.heading_text) > 5
                  and c.token_count > 30
                order by random()
                limit %s
                """,
                (sample_size * 5,),
            )
            candidates = cur.fetchall()
    finally:
        conn.close()

    dataset = []
    seen_queries: set[str] = set()
    for row in candidates:
        heading = _clean_heading(row["heading_text"])
        if heading is None:
            continue
        query = _formulate_query(heading)
        if query in seen_queries:
            continue
        seen_queries.add(query)
        dataset.append(
            {
                "query": query,
                "expected_chunk_id": str(row["id"]),
                "heading": heading,
                "section": row["section_path"],
                "doc": (row["title"] or "")[:80],
            }
        )
        if len(dataset) >= sample_size:
            break

    return dataset


if __name__ == "__main__":
    dataset = build_dataset(sample_size=30)
    out = Path("data/curated_eval_dataset.json")
    out.write_text(json.dumps(dataset, indent=2))
    print(f"Wrote {len(dataset)} eval samples to {out}")
    for i, item in enumerate(dataset[:8]):
        print(f"  [{i}] q={item['query'][:70]}")
        print(f"       doc={item['doc'][:50]}")
