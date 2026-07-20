# ADR-0005: Document dedup is by SHA-256 of source bytes, not by path

## Status

Accepted

## Date

2026-07-20

## Context

`lean` deduplicates documents on ingest to prevent the same PDF from
filling the corpus twice (intentional re-ingest for side-by-side
experiments, accidental drag-and-drop duplicates, restarts after
extraction failure). The dedup key was historically a path string;
that breaks under two common cases:

1. **Renamed PDF.** User deletes `data/dmaic_v1.pdf`, replaces it
   with `data/dmaic_v2.pdf` containing the *same bytes*. Path-dedup
   treats them as distinct, doubling the corpus.
2. **Edited PDF.** User fixes a typo in `data/dmaic.pdf`. SHA-256
   changes, so a SHA-dedup treats them as distinct again.

The converse — *"two PDFs with the same content but legitimately
different metadata"* — is rare; the project serves search, not
provenance tracking, so collapsing by content is the right default.

The `documents.source_sha256` column was added in
`db/schemas/002_documents.sql` (commit history predates AGENTS.md)
to make dedup by content primary, but only the embeddings layer
checked it. The old path-keyed upsert left orphan rows whenever a
file was renamed.

## Decision

`DocumentRepo.upsert(doc)` matches incoming documents on
`source_sha256` (NOT NULL, unique-constrained in migration 002). A
re-ingest of a path that points at bytes already in the corpus
**merges into the existing document**: chunk rows are deleted and
replaced (atomic txn, see `services/ingestion.py:_persist_ingest`),
provenance metadata is updated to the latest ingest timestamp, and
the `source_path` column is updated to the new path.

A re-ingest with the *same* path but different SHA (edited file) is
treated as a new document. Old chunks are left orphaned; the operator
is responsible for `lean delete <old_doc_id>` plus optional REINDEX
(see `docs/operations.md §post-ingest-reindex`).

A re-ingest after partial failure: same SHA as the previous attempt,
chunk rows are atomically replaced. No duplicate-document risk.

## Alternatives considered

### Path-keyed dedup

- Pros: simpler mental model; humans understand paths.
- Cons: the rename case doubles the corpus silently.
- Rejected: silent duplication is worse than orphan rows.

### Content-only (no SHA at all)

- Pros: byte-exact dedup.
- Cons: identical with SHA — the current design already does this.
- Rejected: equivalent.

### SHA + path composite key

- Pros: a renamed-but-identical file gets a new row, preserving
  ingestion history for both paths.
- Cons: doubles the corpus for the common rename-during-update
  case; doesn't help anyone; complicates dedup.
- Rejected: the rename case is rare; when it happens, the operator
  can resolve via explicit `delete` + `reingest`.

### Per-chunk dedup (cross-document)

- Pros: catches duplicated passages.
- Cons: complicates provenance; the use case doesn't justify the
  implementation cost.
- Rejected: scope creep.

## Consequences

- Operators who edit a PDF must `delete` the old doc_id; the ingest
  log should warn when this is needed (`lean ingest <edited.pdf>`
  returns `warnings=["document replaced — old_doc_id=…"]`).
- The `documents.source_path` column is mutable now — it's the
  *latest* path, not the original. Reproducing an ingest always
  works (re-runs hit the SHA and no-op the chunk replace).
- A migration path from path-keyed corpora is required if users
  have existing data; the simplest is to leave the old `documents`
  rows alone and dedup *new* ingests by SHA, with a flag-day
  `ReingestAll` to migrate.

## References

- Code: `src/lean/core/store/documents.py:DocumentRepo.upsert`,
  `src/lean/core/services/ingestion.py:_persist_ingest`
- Schema: `db/schemas/002_documents.sql` (`source_sha256 UNIQUE NOT NULL`)
- Tests: `tests/test_store_pgvector.py` (`test_reingested_at_updates_on_upsert`)
- Doc: `docs/limitations.md § Dedup is by SHA-256, not by path`,
  `docs/operations.md § reingest-semantics`
- Audit finding: this is pre-existing behavior (not opened in the
  audit) but flagged by `docs/limitations.md:41`
