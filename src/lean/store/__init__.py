"""Postgres + pgvector persistence, split into focused repos.

Modules:
    base       — ``StoreConnection``: owns the psycopg connection.
    documents  — ``DocumentRepo``: documents table CRUD.
    chunks     — ``ChunkRepo`` + ``ChunkRow``: chunks table CRUD.
    search     — ``SearchEngine`` + ``SearchHit``: vector/BM25 search + RRF.
    analytics  — ``AnalyticsRepo``: query logging + corpus stats.
"""

from __future__ import annotations
