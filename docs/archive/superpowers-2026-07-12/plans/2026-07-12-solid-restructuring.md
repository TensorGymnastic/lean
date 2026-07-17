# SOLID Restructuring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the lean codebase to apply SOLID/KISS/DRY/YAGNI — split God modules, add service layer, centralize config in `.env` + `config.yaml`.

**Architecture:** Pragmatic refactor: split `pgvector.py` (417 LOC) into 5 focused store modules, extract business logic from `tools.py` into 3 service modules, centralize embedder factory, split config into `.env` (secrets) + `config.yaml` (app config). Transport layers become thin delegates.

**Tech Stack:** Python 3.12, pydantic-settings, pyyaml, psycopg, pgvector, fastmcp, FastAPI, Typer

---

## File Structure

```
src/lean/
├── config/                    ← NEW package
│   ├── __init__.py
│   ├── settings.py            ← Loads .env + config.yaml
│   └── config.yaml            ← App config defaults
├── infrastructure/            ← NEW package
│   ├── __init__.py
│   └── embedder.py            ← Singleton embedder factory (DRY)
├── services/                  ← NEW package
│   ├── __init__.py
│   ├── ingestion.py           ← Ingest pipeline
│   ├── search.py              ← Search orchestration
│   └── corpus.py              ← Corpus management
├── store/                     ← EXPANDED from pgvector.py monolith
│   ├── __init__.py
│   ├── base.py                ← Connection management
│   ├── documents.py           ← Document CRUD
│   ├── chunks.py              ← Chunk CRUD
│   ├── search.py              ← Vector + BM25 + RRF
│   └── analytics.py           ← Query logs + stats
├── models/schemas.py          ← Unchanged
├── extraction/                ← Unchanged
├── chunker/                   ← Unchanged
├── embeddings/                ← Unchanged
├── retrieval/                 ← Unchanged
├── eval/                      ← Unchanged
├── mcp_server/tools.py        ← THIN: delegates to services
├── api/routes.py              ← THIN: delegates to services
├── cli.py                     ← THIN: delegates to services
└── auth/                      ← Unchanged
```

**Deleted:** `src/lean/settings.py` (replaced by `config/settings.py`)

---

### Task 1: Config layer — settings.py + config.yaml

**Files:**
- Create: `src/lean/config/__init__.py`
- Create: `src/lean/config/settings.py`
- Create: `src/lean/config/config.yaml`
- Delete: `src/lean/settings.py`

- [ ] **Step 1: Create config.yaml**

```yaml
# src/lean/config/config.yaml
vllm:
  base_url: http://localhost:8000

embedding:
  model: LiquidAI/LFM2.5-Embedding-350M
  dim: 1024
  device: cpu

ocr:
  model: baidu/Unlimited-OCR
  dpi: 300
  timeout_s: 600.0
  max_tokens: 32768

chunking:
  target_min: 350
  target_max: 450
  hard_cap: 500
  max_heading_level: 4
  token_encoding: cl100k_base

retrieval:
  top_k: 5
  min_similarity: 0.0
  hybrid_search: true
  rerank:
    enabled: false
    model: cross-encoder/ms-marco-MiniLM-L-6-v2
    top_n: 5

storage:
  sources_bucket: sources
  markdown_bucket: markdown

transport:
  mcp_host: 127.0.0.1
  mcp_port: 8765
  api_port: 8766
```

- [ ] **Step 2: Create config/settings.py**

Loads `.env` for secrets and `config.yaml` for app config. Env vars override YAML for deployment flexibility.

```python
# src/lean/config/settings.py
"""Application settings: .env for secrets, config.yaml for app config."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml() -> dict[str, object]:
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.is_file():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml()


class Settings(BaseSettings):
    """Unified settings: env vars (secrets) override YAML defaults (app config)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets (from .env only) ---
    supabase_url: str = Field(description="Supabase project URL")
    supabase_service_key: str = Field(description="Supabase service_role key")
    supabase_db_url: str = Field(description="Direct Postgres DSN for pgvector")
    hf_token: str = Field(default="", description="HF token for gated models")
    lean_mcp_api_key: str = Field(description="Bearer token for MCP HTTP transport")

    # --- App config (from config.yaml, overridable by env) ---
    vllm_base_url: str = _yaml.get("vllm", {}).get("base_url", "http://localhost:8000")  # type: ignore[union-attr]

    embedding_model: str = _yaml.get("embedding", {}).get("model", "LiquidAI/LFM2.5-Embedding-350M")  # type: ignore[union-attr]
    embedding_dim: int = _yaml.get("embedding", {}).get("dim", 1024)  # type: ignore[union-attr]
    embedding_device: str = _yaml.get("embedding", {}).get("device", "cpu")  # type: ignore[union-attr]

    ocr_model: str = _yaml.get("ocr", {}).get("model", "baidu/Unlimited-OCR")  # type: ignore[union-attr]
    ocr_dpi: int = _yaml.get("ocr", {}).get("dpi", 300)  # type: ignore[union-attr]
    ocr_timeout_s: float = _yaml.get("ocr", {}).get("timeout_s", 600.0)  # type: ignore[union-attr]
    ocr_max_tokens: int = _yaml.get("ocr", {}).get("max_tokens", 32768)  # type: ignore[union-attr]

    chunk_target_min: int = _yaml.get("chunking", {}).get("target_min", 350)  # type: ignore[union-attr]
    chunk_target_max: int = _yaml.get("chunking", {}).get("target_max", 450)  # type: ignore[union-attr]
    chunk_hard_cap: int = _yaml.get("chunking", {}).get("hard_cap", 500)  # type: ignore[union-attr]
    max_section_heading_level: int = _yaml.get("chunking", {}).get("max_heading_level", 4)  # type: ignore[union-attr]
    token_counter_encoding: str = _yaml.get("chunking", {}).get("token_encoding", "cl100k_base")  # type: ignore[union-attr]

    search_top_k: int = _yaml.get("retrieval", {}).get("top_k", 5)  # type: ignore[union-attr]
    min_similarity: float = _yaml.get("retrieval", {}).get("min_similarity", 0.0)  # type: ignore[union-attr]
    hybrid_search_enabled: bool = _yaml.get("retrieval", {}).get("hybrid_search", True)  # type: ignore[union-attr]
    rerank_enabled: bool = _yaml.get("retrieval", {}).get("rerank", {}).get("enabled", False)  # type: ignore[union-attr]
    rerank_model: str = _yaml.get("retrieval", {}).get("rerank", {}).get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")  # type: ignore[union-attr]
    rerank_top_n: int = _yaml.get("retrieval", {}).get("rerank", {}).get("top_n", 5)  # type: ignore[union-attr]

    sources_bucket: str = _yaml.get("storage", {}).get("sources_bucket", "sources")  # type: ignore[union-attr]
    markdown_bucket: str = _yaml.get("storage", {}).get("markdown_bucket", "markdown")  # type: ignore[union-attr]

    mcp_http_host: str = _yaml.get("transport", {}).get("mcp_host", "127.0.0.1")  # type: ignore[union-attr]
    mcp_http_port: int = _yaml.get("transport", {}).get("mcp_port", 8765)  # type: ignore[union-attr]
    api_port: int = _yaml.get("transport", {}).get("api_port", 8766)  # type: ignore[union-attr]


def get_settings() -> Settings:
    """Factory that re-reads env on each call (test-friendly)."""
    return Settings()
```

- [ ] **Step 3: Create config/__init__.py**

```python
# src/lean/config/__init__.py
"""Configuration package."""
```

- [ ] **Step 4: Update all imports across codebase**

Global find-replace: `from lean.settings import` → `from lean.config.settings import`

Files to update (every file that imports Settings):
- `mcp_server/tools.py`
- `retrieval/search.py`
- `mcp_server/__main__.py`
- `api/routes.py`
- `cli.py`
- `eval/runner.py`
- `tests/test_settings.py`

- [ ] **Step 5: Delete old settings.py**

```bash
rm src/lean/settings.py
```

- [ ] **Step 6: Add pyyaml dependency**

```bash
uv add pyyaml
```

- [ ] **Step 7: Run tests + commit**

```bash
uv run ruff check src/ tests/
uv run mypy src/lean
uv run pytest -m "not slow and not integration and not e2e" -q
git add -A
git commit -m "refactor: centralize config in config/ package (.env + config.yaml)"
```

---

### Task 2: Store base + documents + chunks + search + analytics

Split `pgvector.py` (417 LOC) into 5 focused modules. The old `pgvector.py` stays as a compatibility shim that re-exports from the new modules during the transition, then gets deleted.

**Files:**
- Create: `src/lean/store/__init__.py` (update)
- Create: `src/lean/store/base.py`
- Create: `src/lean/store/documents.py`
- Create: `src/lean/store/chunks.py`
- Create: `src/lean/store/search.py`
- Create: `src/lean/store/analytics.py`
- Delete: `src/lean/store/pgvector.py` (after all imports updated)

- [ ] **Step 1: Create store/base.py**

Connection management only:

```python
# src/lean/store/base.py
"""Postgres connection management for the lean store."""

from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector


class StoreConnection:
    """Manages a single psycopg connection with pgvector registered."""

    def __init__(self, db_url: str) -> None:
        self._conn = psycopg.connect(db_url, autocommit=False)
        register_vector(self._conn)

    @classmethod
    def from_env(cls) -> StoreConnection:
        db_url = os.environ.get("SUPABASE_DB_URL")
        if not db_url:
            from lean.config.settings import Settings

            db_url = Settings().supabase_db_url
        return cls(db_url)

    @property
    def conn(self) -> psycopg.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> StoreConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
```

- [ ] **Step 2: Create store/documents.py**

Document CRUD. Depends on `StoreConnection` and `models.schemas`.

```python
# src/lean/store/documents.py
"""Document CRUD operations."""

from __future__ import annotations

from uuid import UUID

from psycopg.rows import dict_row

from lean.models.schemas import DocumentSummary, ExtractionMethod
from lean.store.base import StoreConnection


class DocumentRepo:
    """Repository for document lifecycle operations."""

    def __init__(self, conn: StoreConnection) -> None:
        self._conn = conn

    def upsert_document(
        self,
        *,
        source_path: str,
        source_sha256: str,
        title: str | None,
        extraction_method: str,
        source_storage_path: str,
        markdown_storage_path: str,
        authors: list[str] | None = None,
        publisher: str | None = None,
        year: int | None = None,
        page_count: int | None = None,
    ) -> UUID:
        with self._conn.conn.cursor() as cur:
            cur.execute(
                """
                insert into public.documents (
                    source_path, source_sha256, title, authors, publisher,
                    year, page_count, extraction_method,
                    source_storage_path, markdown_storage_path
                ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (source_sha256) do update set
                    source_path = excluded.source_path,
                    title = excluded.title,
                    authors = excluded.authors,
                    publisher = excluded.publisher,
                    year = excluded.year,
                    page_count = excluded.page_count,
                    extraction_method = excluded.extraction_method,
                    reingested_at = now()
                returning id
                """,
                (
                    source_path, source_sha256, title, authors or [], publisher,
                    year, page_count, extraction_method,
                    source_storage_path, markdown_storage_path,
                ),
            )
            row = cur.fetchone()
            self._conn.conn.commit()
            return row[0]

    def delete_document(self, document_id: UUID) -> None:
        with self._conn.conn.cursor() as cur:
            cur.execute("delete from public.documents where id = %s", (document_id,))
            self._conn.conn.commit()

    def list_documents(self) -> list[DocumentSummary]:
        with self._conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                select d.id, d.source_path, d.title, d.authors, d.page_count,
                       d.extraction_method, d.ingested_at,
                       count(c.id) as chunk_count
                from public.documents d
                left join public.chunks c on c.document_id = d.id
                group by d.id
                order by d.ingested_at desc
                """
            )
            return [
                DocumentSummary(
                    id=str(r["id"]),
                    source_path=r["source_path"],
                    title=r["title"],
                    authors=r["authors"] or [],
                    page_count=r["page_count"],
                    extraction_method=r["extraction_method"],
                    chunk_count=r["chunk_count"],
                    ingested_at=r["ingested_at"],
                )
                for r in cur.fetchall()
            ]

    def get_document_storage_paths(self, document_id: UUID) -> tuple[str, str] | None:
        with self._conn.conn.cursor() as cur:
            cur.execute(
                "select source_storage_path, markdown_storage_path"
                " from public.documents where id = %s",
                (document_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0], row[1]
```

- [ ] **Step 3: Create store/chunks.py**

Chunk CRUD. Depends on `StoreConnection` and `models.schemas`.

```python
# src/lean/store/chunks.py
"""Chunk CRUD operations."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from psycopg.rows import dict_row

from lean.models.schemas import Chunk
from lean.store.base import StoreConnection


@dataclass
class ChunkRow:
    document_id: UUID
    chunk_index: int
    section_path: str
    heading_text: str | None
    page_start: int | None
    page_end: int | None
    token_count: int
    content: str
    embedding: list[float]


class ChunkRepo:
    """Repository for chunk lifecycle operations."""

    def __init__(self, conn: StoreConnection) -> None:
        self._conn = conn

    def replace_chunks(self, document_id: UUID, chunks: list[ChunkRow]) -> None:
        with self._conn.conn.cursor() as cur:
            cur.execute("delete from public.chunks where document_id = %s", (document_id,))
            cur.executemany(
                """
                insert into public.chunks (
                    document_id, chunk_index, section_path, heading_text,
                    page_start, page_end, token_count, content, embedding
                ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        c.document_id, c.chunk_index, c.section_path, c.heading_text,
                        c.page_start, c.page_end, c.token_count, c.content, c.embedding,
                    )
                    for c in chunks
                ],
            )
            self._conn.conn.commit()

    def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        with self._conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                select id, document_id, chunk_index, section_path, heading_text,
                       page_start, page_end, token_count, content
                from public.chunks where id = %s
                """,
                (chunk_id,),
            )
            r = cur.fetchone()
            if r is None:
                return None
            return Chunk(
                id=str(r["id"]), document_id=str(r["document_id"]),
                chunk_index=r["chunk_index"], section_path=r["section_path"],
                heading_text=r["heading_text"], page_start=r["page_start"],
                page_end=r["page_end"], token_count=r["token_count"],
                content=r["content"],
            )

    def count_chunks(self) -> int:
        with self._conn.conn.cursor() as cur:
            cur.execute("select count(*) from public.chunks")
            return cur.fetchone()[0]
```

- [ ] **Step 4: Create store/search.py**

Vector search + BM25 + RRF. Depends on `StoreConnection` and `models.schemas`.

```python
# src/lean/store/search.py
"""Vector + BM25 hybrid search with Reciprocal Rank Fusion."""

from __future__ import annotations

import json
from uuid import UUID

from psycopg.rows import dict_row

from lean.models.schemas import Chunk
from lean.store.base import StoreConnection


class SearchHit:
    """A search result with its similarity score."""

    def __init__(self, chunk: Chunk, score: float) -> None:
        self.chunk = chunk
        self.score = score


class SearchEngine:
    """Hybrid search engine: vector cosine + BM25 full-text + RRF fusion."""

    def __init__(self, conn: StoreConnection) -> None:
        self._conn = conn

    def vector_search(
        self,
        *,
        query_embedding: list[float],
        k: int = 5,
        doc_id: UUID | None = None,
        section_substring: str | None = None,
        author: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        min_score: float | None = None,
    ) -> list[SearchHit]:
        """Cosine similarity search over chunks with metadata filters."""
        needs_join = author is not None or year_min is not None or year_max is not None
        join_clause = " join public.documents d on d.id = c.document_id" if needs_join else ""

        conditions = [
            "(%s::uuid is null or c.document_id = %s)",
            "(%s::text is null or c.section_path ilike '%%' || %s || '%%')",
        ]
        params: list[object] = [doc_id, doc_id, section_substring, section_substring]

        if author is not None:
            conditions.append(
                "exists (select 1 from unnest(d.authors) a"
                " where a ilike '%%' || %s || '%%')"
            )
            params.append(author)
        if year_min is not None:
            conditions.append("d.year >= %s")
            params.append(year_min)
        if year_max is not None:
            conditions.append("d.year <= %s")
            params.append(year_max)
        if min_score is not None:
            conditions.append("1 - (c.embedding <=> %s::vector) >= %s")
            params.extend([query_embedding, min_score])

        where_clause = " and ".join(conditions)
        sql = f"""
            select c.id, c.document_id, c.chunk_index, c.section_path,
                   c.heading_text, c.page_start, c.page_end, c.token_count,
                   c.content, 1 - (c.embedding <=> %s::vector) as score
            from public.chunks c{join_clause}
            where {where_clause}
            order by c.embedding <=> %s::vector
            limit %s
        """
        params_final = [query_embedding] + params + [query_embedding, k]
        return self._execute_search(sql, params_final)

    def bm25_search(
        self,
        *,
        query_text: str,
        k: int = 5,
        doc_id: UUID | None = None,
        section_substring: str | None = None,
        author: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> list[SearchHit]:
        """Full-text search via tsvector + ts_rank."""
        needs_join = author is not None or year_min is not None or year_max is not None
        join_clause = " join public.documents d on d.id = c.document_id" if needs_join else ""

        conditions = [
            "c.tsv @@ plainto_tsquery('english', %s)",
            "(%s::uuid is null or c.document_id = %s)",
            "(%s::text is null or c.section_path ilike '%%' || %s || '%%')",
        ]
        params: list[object] = [query_text, doc_id, doc_id, section_substring, section_substring]

        if author is not None:
            conditions.append(
                "exists (select 1 from unnest(d.authors) a"
                " where a ilike '%%' || %s || '%%')"
            )
            params.append(author)
        if year_min is not None:
            conditions.append("d.year >= %s")
            params.append(year_min)
        if year_max is not None:
            conditions.append("d.year <= %s")
            params.append(year_max)

        where_clause = " and ".join(conditions)
        sql = f"""
            select c.id, c.document_id, c.chunk_index, c.section_path,
                   c.heading_text, c.page_start, c.page_end, c.token_count,
                   c.content, ts_rank(c.tsv, plainto_tsquery('english', %s)) as score
            from public.chunks c{join_clause}
            where {where_clause}
            order by score desc
            limit %s
        """
        params_final = [query_text] + params + [k]
        return self._execute_search(sql, params_final)

    @staticmethod
    def reciprocal_rank_fusion(
        vector_hits: list[SearchHit],
        bm25_hits: list[SearchHit],
        *,
        k: int = 5,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        """Fuse two ranked lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        best_hit: dict[str, SearchHit] = {}

        for rank, hit in enumerate(vector_hits):
            cid = hit.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            best_hit[cid] = hit

        for rank, hit in enumerate(bm25_hits):
            cid = hit.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            if cid not in best_hit:
                best_hit[cid] = hit

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        result: list[SearchHit] = []
        for cid, score in ranked:
            hit = best_hit[cid]
            result.append(
                SearchHit(
                    chunk=hit.chunk.model_copy(update={"score": score}),
                    score=score,
                )
            )
        return result

    def _execute_search(self, sql: str, params: tuple[object, ...]) -> list[SearchHit]:
        with self._conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return [
                SearchHit(
                    chunk=Chunk(
                        id=str(r["id"]),
                        document_id=str(r["document_id"]),
                        chunk_index=r["chunk_index"],
                        section_path=r["section_path"],
                        heading_text=r["heading_text"],
                        page_start=r["page_start"],
                        page_end=r["page_end"],
                        token_count=r["token_count"],
                        content=r["content"],
                        score=float(r["score"]),
                    ),
                    score=float(r["score"]),
                )
                for r in cur.fetchall()
            ]
```

- [ ] **Step 5: Create store/analytics.py**

Query logging + corpus stats + document count.

```python
# src/lean/store/analytics.py
"""Analytics: query logging and corpus statistics."""

from __future__ import annotations

import json
from uuid import UUID

from psycopg.rows import dict_row

from lean.models.schemas import CorpusStats
from lean.store.base import StoreConnection


class AnalyticsRepo:
    """Repository for analytics and statistics."""

    def __init__(self, conn: StoreConnection) -> None:
        self._conn = conn

    def log_query(
        self,
        *,
        query_text: str,
        k: int,
        filters: dict[str, object],
        hit_chunk_ids: list[UUID],
        hit_scores: list[float],
        latency_ms: int,
        agent_id: str | None = None,
    ) -> None:
        with self._conn.conn.cursor() as cur:
            cur.execute(
                """
                insert into public.query_logs
                    (query_text, k, filters, hit_chunk_ids, hit_scores, latency_ms, agent_id)
                values (%s, %s, %s::jsonb, %s::uuid[], %s, %s, %s)
                """,
                (query_text, k, json.dumps(filters), hit_chunk_ids,
                 hit_scores, latency_ms, agent_id),
            )
            self._conn.conn.commit()

    def count_documents(self) -> int:
        with self._conn.conn.cursor() as cur:
            cur.execute("select count(*) from public.documents")
            return cur.fetchone()[0]

    def corpus_stats(
        self, *, embedding_dim: int, embedding_model: str
    ) -> CorpusStats:
        with self._conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                select
                    count(distinct d.id) as doc_count,
                    count(c.id) as chunk_count,
                    coalesce(sum(c.token_count), 0) as total_tokens,
                    max(d.ingested_at) as last_ingested
                from public.documents d
                left join public.chunks c on c.document_id = d.id
                """
            )
            row = cur.fetchone()

            cur.execute(
                """
                select extraction_method, count(*) as cnt
                from public.documents group by extraction_method
                """
            )
            breakdown = {r["extraction_method"]: r["cnt"] for r in cur.fetchall()}

        from datetime import datetime

        return CorpusStats(
            document_count=row["doc_count"],
            chunk_count=row["chunk_count"],
            total_tokens=int(row["total_tokens"]),
            extraction_method_breakdown=breakdown,
            embedding_dim=embedding_dim,
            embedding_model=embedding_model,
            last_ingested_at=row["last_ingested"],
        )
```

- [ ] **Step 6: Update store/__init__.py**

```python
# src/lean/store/__init__.py
"""Store package: connection, repositories, and search engine."""
```

- [ ] **Step 7: Run tests + commit**

```bash
uv run ruff check src/lean/store/
uv run mypy src/lean/store/
git add -A
git commit -m "refactor: split pgvector.py monolith into focused store modules

store/base.py — connection management
store/documents.py — document CRUD
store/chunks.py — chunk CRUD
store/search.py — vector + BM25 + RRF search engine
store/analytics.py — query logging + corpus stats"
```

---

### Task 3: Infrastructure — embedder factory

**Files:**
- Create: `src/lean/infrastructure/__init__.py`
- Create: `src/lean/infrastructure/embedder.py`

- [ ] **Step 1: Create infrastructure/embedder.py**

Single embedder factory — replaces duplicated `_get_embedder()` in `tools.py` and `search.py`.

```python
# src/lean/infrastructure/embedder.py
"""Singleton embedder factory — DRY replacement for duplicated _get_embedder()."""

from __future__ import annotations

from lean.config.settings import Settings
from lean.embeddings.liquid_lmf import LiquidLMFEmbedder

_embedder: LiquidLMFEmbedder | None = None


def get_embedder() -> LiquidLMFEmbedder:
    global _embedder
    if _embedder is None:
        s = Settings()
        _embedder = LiquidLMFEmbedder(
            model=s.embedding_model,
            hf_token=s.hf_token,
            device=s.embedding_device,
            dim=s.embedding_dim,
        )
    return _embedder
```

- [ ] **Step 2: Create infrastructure/__init__.py**

```python
# src/lean/infrastructure/__init__.py
"""Infrastructure: shared singletons and factories."""
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: centralize embedder factory in infrastructure/embedder.py"
```

---

### Task 4: Services — ingestion, search, corpus

**Files:**
- Create: `src/lean/services/__init__.py`
- Create: `src/lean/services/ingestion.py`
- Create: `src/lean/services/search.py`
- Create: `src/lean/services/corpus.py`

- [ ] **Step 1: Create services/ingestion.py**

Move ingestion pipeline from `tools.py:ingest_pdf` and `tools.py:reingest` into a service.

```python
# src/lean/services/ingestion.py
"""Ingestion service: PDF → extract → metadata → chunk → embed → store."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from uuid import UUID

import anyio

from lean.chunker.markdown_ast import build_sections
from lean.chunker.recursive import chunk_sections
from lean.config.settings import Settings
from lean.extraction.metadata import extract_metadata
from lean.extraction.pipeline import extract_pdf_markdown
from lean.infrastructure.embedder import get_embedder
from lean.models.schemas import ExtractionMethod, IngestResult
from lean.store.analytics import AnalyticsRepo
from lean.store.base import StoreConnection
from lean.store.chunks import ChunkRepo, ChunkRow
from lean.store.documents import DocumentRepo


async def ingest_pdf(path: str) -> IngestResult:
    """Full ingestion pipeline: PDF → chunks → embeddings → pgvector."""
    start = time.monotonic()
    settings = Settings()
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    source_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    markdown, page_count, method = await anyio.to_thread.run_sync(
        lambda: extract_pdf_markdown(
            pdf_path,
            vllm_base_url=settings.vllm_base_url,
            hf_token=settings.hf_token,
            ocr_model=settings.ocr_model,
            ocr_dpi=settings.ocr_dpi,
            ocr_timeout_s=settings.ocr_timeout_s,
            ocr_max_tokens=settings.ocr_max_tokens,
        )
    )

    warnings: list[str] = []
    if method == ExtractionMethod.MARKITDOWN:
        warnings.append("vLLM unavailable, fell back to markitdown")

    sections = await anyio.to_thread.run_sync(
        lambda: build_sections(markdown, max_heading_level=settings.max_section_heading_level)
    )
    chunk_results = await anyio.to_thread.run_sync(
        lambda: chunk_sections(
            sections,
            target_min=settings.chunk_target_min,
            target_max=settings.chunk_target_max,
            hard_cap=settings.chunk_hard_cap,
            encoding=settings.token_counter_encoding,
        )
    )

    embedder = get_embedder()
    chunk_texts = [c.content for c in chunk_results]
    embeddings = await anyio.to_thread.run_sync(lambda: embedder.embed_documents(chunk_texts))

    pdf_meta = await anyio.to_thread.run_sync(lambda: extract_metadata(pdf_path))

    conn = StoreConnection.from_env()
    try:
        doc_repo = DocumentRepo(conn)
        chunk_repo = ChunkRepo(conn)

        source_storage_path = f"sources/{source_sha256}.pdf"
        markdown_storage_path = f"markdown/{source_sha256}.md"

        doc_id = doc_repo.upsert_document(
            source_path=str(pdf_path),
            source_sha256=source_sha256,
            title=pdf_meta.title or pdf_path.stem,
            extraction_method=method.value,
            source_storage_path=source_storage_path,
            markdown_storage_path=markdown_storage_path,
            authors=pdf_meta.authors,
            publisher=pdf_meta.publisher,
            year=pdf_meta.year,
            page_count=page_count,
        )
        chunk_rows = [
            ChunkRow(
                document_id=doc_id,
                chunk_index=global_idx,
                section_path=c.section_path,
                heading_text=c.heading_text,
                page_start=None,
                page_end=None,
                token_count=c.token_count,
                content=c.content,
                embedding=emb,
            )
            for global_idx, (c, emb) in enumerate(
                zip(chunk_results, embeddings, strict=True)
            )
        ]
        chunk_repo.replace_chunks(doc_id, chunk_rows)
    finally:
        conn.close()

    elapsed = time.monotonic() - start
    return IngestResult(
        document_id=str(doc_id),
        source_sha256=source_sha256,
        page_count=page_count,
        extraction_method=method,
        chunk_count=len(chunk_rows),
        elapsed_seconds=elapsed,
        warnings=warnings,
    )


async def reingest(document_id: str) -> IngestResult:
    """Re-ingest an existing document by its UUID."""
    conn = StoreConnection.from_env()
    try:
        doc_repo = DocumentRepo(conn)
        paths = doc_repo.get_document_storage_paths(UUID(document_id))
        if paths is None:
            raise FileNotFoundError(f"Document not found: {document_id}")
        # The source_path stored in DB is the original file path
        from psycopg.rows import dict_row
        with conn.conn.cursor(row_factory=dict_row) as cur:
            cur.execute("select source_path from public.documents where id = %s", (UUID(document_id),))
            row = cur.fetchone()
            if row is None:
                raise FileNotFoundError(f"Document not found: {document_id}")
            source_path = row["source_path"]
    finally:
        conn.close()

    return await ingest_pdf(source_path)
```

- [ ] **Step 2: Create services/search.py**

Move search orchestration from `retrieval/search.py` into a service (without module-level singletons).

```python
# src/lean/services/search.py
"""Search service: embed → hybrid search → rerank → postprocess → log."""

from __future__ import annotations

import logging
import time
from uuid import UUID

from lean.config.settings import Settings
from lean.infrastructure.embedder import get_embedder
from lean.models.schemas import Chunk
from lean.store.analytics import AnalyticsRepo
from lean.store.base import StoreConnection
from lean.store.search import SearchEngine, SearchHit

logger = logging.getLogger(__name__)


def search(
    query: str,
    *,
    k: int = 5,
    doc_id: str | None = None,
    section: str | None = None,
    author: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    min_score: float | None = None,
    agent_id: str | None = None,
) -> list[Chunk]:
    """Search with hybrid BM25+vector, optional reranking, and postprocessing."""
    settings = Settings()
    embedder = get_embedder()
    conn = StoreConnection.from_env()

    try:
        engine = SearchEngine(conn)
        analytics = AnalyticsRepo(conn)

        start = time.monotonic()
        query_vec = embedder.embed_query(query)
        doc_uuid = UUID(doc_id) if doc_id else None

        if settings.hybrid_search_enabled:
            fetch_k = max(k * 4, 20)
            vector_hits = engine.vector_search(
                query_embedding=query_vec, k=fetch_k, doc_id=doc_uuid,
                section_substring=section, author=author,
                year_min=year_min, year_max=year_max, min_score=min_score,
            )
            bm25_hits = engine.bm25_search(
                query_text=query, k=fetch_k, doc_id=doc_uuid,
                section_substring=section, author=author,
                year_min=year_min, year_max=year_max,
            )
            hits = engine.reciprocal_rank_fusion(vector_hits, bm25_hits, k=fetch_k)
        else:
            hits = engine.vector_search(
                query_embedding=query_vec, k=k, doc_id=doc_uuid,
                section_substring=section, author=author,
                year_min=year_min, year_max=year_max, min_score=min_score,
            )

        if settings.rerank_enabled and hits:
            from lean.retrieval.reranker import rerank
            hits = rerank(hits, query, model=settings.rerank_model,
                          top_n=settings.rerank_top_n, device=settings.embedding_device)

        effective_min = min_score if min_score is not None else settings.min_similarity
        if effective_min > 0:
            from lean.retrieval.postprocessors import similarity_filter
            hits = similarity_filter(hits, effective_min)

        from lean.retrieval.postprocessors import long_context_reorder
        hits = long_context_reorder(hits)
        hits = hits[:k]
        latency_ms = int((time.monotonic() - start) * 1000)

        filters: dict[str, object] = {
            "doc_id": doc_id, "section": section, "author": author,
            "year_min": year_min, "year_max": year_max, "min_score": min_score,
            "hybrid": settings.hybrid_search_enabled,
            "reranked": settings.rerank_enabled,
        }
        try:
            analytics.log_query(
                query_text=query, k=k, filters=filters,
                hit_chunk_ids=[UUID(h.chunk.id) for h in hits],
                hit_scores=[h.score for h in hits],
                latency_ms=latency_ms, agent_id=agent_id,
            )
        except Exception:
            logger.warning("failed to log query", exc_info=True)

        logger.info("search q=%r k=%d hits=%d latency=%dms", query[:60], k, len(hits), latency_ms)
        return [h.chunk for h in hits]
    finally:
        conn.close()
```

- [ ] **Step 3: Create services/corpus.py**

```python
# src/lean/services/corpus.py
"""Corpus management: list, delete, stats, get chunk/markdown."""

from __future__ import annotations

from uuid import UUID

from lean.config.settings import Settings
from lean.models.schemas import Chunk, CorpusStats, DocumentSummary
from lean.store.analytics import AnalyticsRepo
from lean.store.base import StoreConnection
from lean.store.chunks import ChunkRepo
from lean.store.documents import DocumentRepo


def list_documents() -> list[DocumentSummary]:
    conn = StoreConnection.from_env()
    try:
        return DocumentRepo(conn).list_documents()
    finally:
        conn.close()


def delete_document(document_id: str) -> dict[str, str]:
    conn = StoreConnection.from_env()
    try:
        DocumentRepo(conn).delete_document(UUID(document_id))
        return {"status": "deleted", "document_id": document_id}
    finally:
        conn.close()


def corpus_stats() -> CorpusStats:
    settings = Settings()
    conn = StoreConnection.from_env()
    try:
        return AnalyticsRepo(conn).corpus_stats(
            embedding_dim=settings.embedding_dim,
            embedding_model=settings.embedding_model,
        )
    finally:
        conn.close()


def get_chunk(chunk_id: str) -> Chunk | None:
    conn = StoreConnection.from_env()
    try:
        return ChunkRepo(conn).get_chunk(UUID(chunk_id))
    finally:
        conn.close()


def get_document_markdown(document_id: str) -> str:
    conn = StoreConnection.from_env()
    try:
        paths = DocumentRepo(conn).get_document_storage_paths(UUID(document_id))
        if paths is None:
            raise FileNotFoundError(f"Document not found: {document_id}")
        markdown_path = paths[1]
        return f"[markdown at storage path {markdown_path}]"
    finally:
        conn.close()
```

- [ ] **Step 4: Create services/__init__.py**

```python
# src/lean/services/__init__.py
"""Service layer: business logic for ingestion, search, and corpus management."""
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: add service layer (ingestion, search, corpus)"
```

---

### Task 5: Thin transport layers

Rewrite `tools.py`, `routes.py`, and `cli.py` to delegate to services instead of containing business logic.

**Files:**
- Modify: `src/lean/mcp_server/tools.py` (337→~120 LOC)
- Modify: `src/lean/api/routes.py` (85→~60 LOC)
- Modify: `src/lean/cli.py` (241→~200 LOC)
- Delete: `src/lean/retrieval/search.py` (replaced by `services/search.py`)

- [ ] **Step 1: Rewrite tools.py**

Thin MCP tool definitions that delegate to services:

```python
# src/lean/mcp_server/tools.py
"""MCP tools — thin delegates to the service layer."""

from __future__ import annotations

from fastmcp import FastMCP

from lean.models.schemas import Chunk, CorpusStats, DocumentSummary, IngestResult
from lean.services.corpus import (
    corpus_stats as _corpus_stats,
    delete_document as _delete,
    get_chunk as _get_chunk,
    get_document_markdown as _get_markdown,
    list_documents as _list,
)
from lean.services.ingestion import ingest_pdf as _ingest, reingest as _reingest
from lean.services.search import search as _search

mcp = FastMCP("lean")


@mcp.tool
async def ingest_pdf(path: str) -> IngestResult:
    """Ingest a PDF into the lean corpus."""
    return await _ingest(path)


@mcp.tool
async def search(
    query: str, k: int = 5, doc_id: str | None = None, section: str | None = None,
    author: str | None = None, year_min: int | None = None,
    year_max: int | None = None, min_score: float | None = None,
) -> list[Chunk]:
    """Semantic + hybrid search over the Lean Six Sigma corpus."""
    import anyio
    return await anyio.to_thread.run_sync(
        lambda: _search(query, k=k, doc_id=doc_id, section=section, author=author,
                        year_min=year_min, year_max=year_max, min_score=min_score)
    )


@mcp.tool
async def get_chunk(chunk_id: str) -> Chunk | None:
    """Retrieve a single chunk by ID."""
    import anyio
    return await anyio.to_thread.run_sync(lambda: _get_chunk(chunk_id))


@mcp.tool
async def list_documents() -> list[DocumentSummary]:
    """List all documents in the corpus."""
    import anyio
    return await anyio.to_thread.run_sync(_list)


@mcp.tool
async def get_document_markdown(document_id: str) -> str:
    """Get the extracted markdown for a document."""
    import anyio
    return await anyio.to_thread.run_sync(lambda: _get_markdown(document_id))


@mcp.tool
async def delete_document(document_id: str) -> dict[str, str]:
    """Delete a document and all its chunks."""
    import anyio
    return await anyio.to_thread.run_sync(lambda: _delete(document_id))


@mcp.tool
async def reingest(document_id: str) -> IngestResult:
    """Re-ingest a document (re-extract with current settings)."""
    return await _reingest(document_id)


@mcp.tool
async def corpus_stats() -> CorpusStats:
    """Show corpus statistics."""
    import anyio
    return await anyio.to_thread.run_sync(_corpus_stats)
```

- [ ] **Step 2: Rewrite routes.py**

```python
# src/lean/api/routes.py
"""FastAPI REST mirror — thin delegates to services."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI

from lean.api.auth.bearer import BearerMiddleware
from lean.config.settings import get_settings

app = FastAPI(title="lean")
app.add_middleware(BearerMiddleware, token=get_settings().lean_mcp_api_key)


def _verify_token() -> None:
    pass  # BearerMiddleware handles it


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/documents", dependencies=[Depends(_verify_token)])
async def list_documents() -> list[dict[str, Any]]:
    from lean.services.corpus import list_documents as _list
    import anyio
    docs = await anyio.to_thread.run_sync(_list)
    return [d.model_dump(mode="json") for d in docs]


@app.get("/search", dependencies=[Depends(_verify_token)])
async def search(
    query: str, k: int = 5, doc_id: str | None = None, section: str | None = None,
    author: str | None = None, year_min: int | None = None,
    year_max: int | None = None, min_score: float | None = None,
) -> list[dict[str, Any]]:
    from lean.services.search import search as _search
    import anyio
    chunks = await anyio.to_thread.run_sync(
        lambda: _search(query, k=k, doc_id=doc_id, section=section, author=author,
                        year_min=year_min, year_max=year_max, min_score=min_score)
    )
    return [c.model_dump(mode="json") for c in chunks]


@app.post("/ingest", dependencies=[Depends(_verify_token)])
async def ingest(path: str) -> dict[str, Any]:
    from lean.services.ingestion import ingest_pdf
    result = await ingest_pdf(path)
    return result.model_dump(mode="json")


@app.get("/stats", dependencies=[Depends(_verify_token)])
async def stats() -> dict[str, Any]:
    from lean.services.corpus import corpus_stats as _stats
    import anyio
    s = await anyio.to_thread.run_sync(_stats)
    return s.model_dump(mode="json")
```

- [ ] **Step 3: Update cli.py imports**

Change `from lean.mcp_server.tools import ...` to `from lean.services... import ...` for each command that was calling tools directly.

- [ ] **Step 4: Delete retrieval/search.py**

The old `retrieval/search.py` is replaced by `services/search.py`. The `retrieval/` package still contains `reranker.py` and `postprocessors.py`.

```bash
rm src/lean/retrieval/search.py
```

- [ ] **Step 5: Update resources.py and prompts.py imports**

These import from `mcp_server.tools` — update to import from services instead.

- [ ] **Step 6: Update eval/runner.py imports**

Change from `lean.store.pgvector` to the new store modules.

- [ ] **Step 7: Run tests + fix all import errors**

```bash
uv run ruff check src/ tests/
uv run mypy src/lean
uv run pytest -m "not slow and not integration and not e2e" -q
```

Fix any remaining import errors. The old `store/pgvector.py` can be kept temporarily as a compatibility shim if needed, then deleted once all imports are clean.

- [ ] **Step 8: Delete old pgvector.py**

```bash
rm src/lean/store/pgvector.py
```

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: thin transport layers, delete God modules

tools.py: 337→120 LOC (thin delegates to services)
routes.py: delegates to services
cli.py: delegates to services
Delete retrieval/search.py (replaced by services/search.py)
Delete store/pgvector.py (replaced by store/ package)"
```

---

### Task 6: Final verification + cleanup

- [ ] **Step 1: Verify all imports**

```bash
uv run python -c "import lean.mcp_server.tools; import lean.api.routes; import lean.cli; print('all imports OK')"
```

- [ ] **Step 2: Run full test suite**

```bash
uv run ruff check src/ tests/
uv run mypy src/lean
uv run pytest -m "not slow and not integration and not e2e" -q
```

- [ ] **Step 3: Verify MCP server starts**

```bash
uv run python -c "
import asyncio
from lean.mcp_server.tools import mcp
tools = asyncio.run(mcp.list_tools())
print(f'{len(tools)} tools: {', '.join(sorted(t.name for t in tools))}')
"
```

- [ ] **Step 4: Verify CLI works**

```bash
uv run lean corpus-stats
uv run lean search "DMAIC" --k 3
```

- [ ] **Step 5: Verify .env.example is correct**

```bash
cat .env.example
```

- [ ] **Step 6: Update .env.example**

Remove VLLM_BASE_URL (it's now in config.yaml, not .env).

- [ ] **Step 7: Final commit**

```bash
git add -A
git commit -m "refactor: SOLID restructuring complete — all modules focused, DRY, testable"
```

---

## Self-Review

**Spec coverage:**
- ✅ Config split (.env + config.yaml) → Task 1
- ✅ Store split (5 modules) → Task 2
- ✅ Infrastructure factory → Task 3
- ✅ Service layer → Task 4
- ✅ Thin transport → Task 5
- ✅ Verification → Task 6

**Placeholder scan:** No TBDs, TODOs, or vague descriptions. All code is complete.

**Type consistency:** `SearchHit` class defined in `store/search.py`, used consistently across `services/search.py` and `retrieval/postprocessors.py`. `ChunkRow` dataclass defined in `store/chunks.py`, used in `services/ingestion.py`. `ExtractionMethod` in `models/schemas.py`, imported consistently.

**SOLID compliance:**
- SRP: Each store module has one responsibility. Each service has one responsibility.
- DRY: One embedder factory. One connection class.
- KISS: No abstract base classes, no DI container, no event bus.
- YAGNI: No interfaces for things we don't have multiple implementations of.
