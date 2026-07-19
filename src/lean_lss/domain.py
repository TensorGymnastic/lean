"""Lean Six Sigma domain — wires lean-core to LSS-specific behavior.

A thin layer that:
1. Subclasses ``CoreSettings`` to add LSS-only knobs.
2. Builds the LSS extraction pipeline (marker → OCR → markitdown).
3. Registers LSS-specific MCP tools, FastAPI routes, and Typer commands.
4. Provides the VLM prompt + parser for chart/image enrichment.

This is what the rest of lean-lss looks like: ~200 LOC of glue on top
of lean-core.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lean_core.config import CoreSettings
from lean_core.extraction import (
    MarkerExtractor,
    MarkitdownExtractor,
    Pipeline,
    UnlimitedOCRExtractor,
)
from lean_core.extraction.pipeline_helpers import hash_image
from lean_core.models import Chunk, CorpusStats, DocumentSummary, IngestResult
from lean_core.transports import DomainRegistration
from lean_core.vlm import OpenAICompatibleVLM
from pydantic import Field

from lean_lss.prompts import CHART_EXTRACTION_PROMPT, parse_description

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastmcp import FastMCP
    from typer import Typer

logger = logging.getLogger(__name__)


class LssSettings(CoreSettings):
    """LSS-specific settings. Adds knobs the universal core doesn't know about."""

    lss_corpus_label: str = Field(default="lean-six-sigma", description="Display label")
    lss_pdf_filename_regex: str = Field(
        default=r"^(?P<title>.+?)(?:\s*-\s*(?P<authors>.+?))?(?:\s*\((?P<year>\d{4})\))?",
        description="Regex to extract title/authors/year from PDF filename",
    )


def _lss_extractors(settings: LssSettings) -> list[Any]:
    """Build the LSS extraction pipeline: marker → OCR → markitdown."""
    from lean_core.extraction import Extractor

    extractors: list[Extractor] = []
    if settings.domain_config.get("marker", {}).get("remote_url"):
        extractors.append(
            MarkerExtractor(
                force_ocr=settings.domain_config.get("marker", {}).get("force_ocr", False),
                remote_url=settings.domain_config["marker"]["remote_url"],
            )
        )
    if settings.domain_config.get("ocr", {}).get("base_url"):
        ocr = settings.domain_config["ocr"]
        extractors.append(
            UnlimitedOCRExtractor(
                base_url=ocr["base_url"],
                hf_token=settings.hf_token,
                model=ocr.get("model", "baidu/Unlimited-OCR"),
                dpi=ocr.get("dpi", 300),
                timeout=ocr.get("timeout_s", 1800.0),
                max_tokens=ocr.get("max_tokens", 32768),
                batch_size=ocr.get("batch_size", 20),
            )
        )
    extractors.append(MarkerExtractor(force_ocr=False, remote_url=""))
    extractors.append(MarkitdownExtractor())
    return extractors


class LssDomain(DomainRegistration[LssSettings]):
    """Lean Six Sigma corpus domain — the full LSS surface."""

    name = "lean-lss"
    version = "0.1.0"
    description = "Lean Six Sigma PDF corpus MCP server"

    def build_settings(self) -> LssSettings:
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.is_file():
            instance = LssSettings.from_yaml(config_path)
        else:
            instance = LssSettings.from_yaml(None)
        return instance  # type: ignore[return-value]

    def build_pipeline(self, settings: LssSettings) -> Pipeline:
        return Pipeline(_lss_extractors(settings))

    def describe_one_image(
        self, name: str, image: Any, settings: LssSettings, prompt: str
    ) -> tuple[str, dict[str, object], str]:
        """LSS-specific image description: chart_extraction prompt + JSON parser."""
        vlm = OpenAICompatibleVLM(
            base_url=settings.vlm_base_url,
            model=settings.vlm_model,
            api_key=settings.vlm_api_key,
            timeout=settings.vlm_timeout_s,
            detail=settings.vlm_detail,
            disable_thinking=settings.vlm_disable_thinking,
        )
        try:
            raw = vlm.describe_image(
                image, prompt=prompt or CHART_EXTRACTION_PROMPT, max_tokens=settings.vlm_max_tokens
            )
            parsed = parse_description(raw)
            embed_text = str(parsed.get("description", raw.strip()))
            if parsed.get("title"):
                embed_text = f"{parsed['title']}\n\n{embed_text}"
            if parsed.get("key_data_points"):
                points = parsed["key_data_points"]
                if isinstance(points, list):
                    embed_text += "\n\nKey data: " + "; ".join(str(p) for p in points)
            img_hash = hash_image(image)
            return embed_text, parsed, img_hash
        finally:
            vlm.close()

    def image_chunk_heading(self, image_meta: dict[str, Any], idx: int) -> str:
        return str(image_meta.get("title") or f"Image {idx + 1}")

    def register_mcp(self, mcp: FastMCP, services: dict[str, Any], settings: LssSettings) -> None:
        import anyio

        @mcp.tool
        async def ingest_pdf(path: str) -> IngestResult:
            return await services["ingest_pdf"](path)  # type: ignore[no-any-return]

        @mcp.tool
        async def search(
            query: str,
            k: int = 5,
            doc_id: str | None = None,
            section: str | None = None,
            author: str | None = None,
            year_min: int | None = None,
            year_max: int | None = None,
            min_score: float | None = None,
            chunk_type: str | None = None,
        ) -> list[Chunk]:
            return await anyio.to_thread.run_sync(
                lambda: services["search"](
                    query,
                    k=k,
                    doc_id=doc_id,
                    section=section,
                    author=author,
                    year_min=year_min,
                    year_max=year_max,
                    min_score=min_score,
                    chunk_type=chunk_type,
                )
            )

        @mcp.tool
        async def get_chunk(chunk_id: str) -> Chunk | None:
            return await anyio.to_thread.run_sync(lambda: services["get_chunk"](chunk_id))

        @mcp.tool
        async def list_documents() -> list[DocumentSummary]:
            return await anyio.to_thread.run_sync(services["list_documents"])

        @mcp.tool
        async def get_document_markdown(document_id: str) -> str:
            return await anyio.to_thread.run_sync(
                lambda: services["get_document_markdown"](document_id)
            )

        @mcp.tool
        async def delete_document(document_id: str) -> dict[str, str]:
            return await anyio.to_thread.run_sync(lambda: services["delete_document"](document_id))

        @mcp.tool
        async def reingest(document_id: str) -> IngestResult:
            return await services["reingest"](document_id)  # type: ignore[no-any-return]

        @mcp.tool
        async def corpus_stats() -> CorpusStats:
            return await anyio.to_thread.run_sync(services["corpus_stats"])

    def register_api(self, app: FastAPI, services: dict[str, Any], settings: LssSettings) -> None:
        """REST routes. Lean-core wires bearer-auth on every endpoint."""
        from fastapi import HTTPException, status
        from pydantic import BaseModel

        class IngestRequest(BaseModel):
            path: str

        @app.get("/search")
        async def search(
            query: str,
            k: int = 5,
            doc_id: str | None = None,
            section: str | None = None,
            author: str | None = None,
            year_min: int | None = None,
            year_max: int | None = None,
            min_score: float | None = None,
            chunk_type: str | None = None,
        ) -> list[dict[str, Any]]:
            import anyio

            chunks = await anyio.to_thread.run_sync(
                lambda: services["search"](
                    query,
                    k=k,
                    doc_id=doc_id,
                    section=section,
                    author=author,
                    year_min=year_min,
                    year_max=year_max,
                    min_score=min_score,
                    chunk_type=chunk_type,
                )
            )
            return [c.model_dump(mode="json") for c in chunks]

        @app.post("/ingest")
        async def ingest(req: IngestRequest) -> dict[str, Any]:
            result = await services["ingest_pdf"](req.path)
            return result.model_dump(mode="json")  # type: ignore[no-any-return]

        @app.get("/documents")
        async def list_documents() -> list[dict[str, Any]]:
            import anyio

            docs = await anyio.to_thread.run_sync(services["list_documents"])
            return [d.model_dump(mode="json") for d in docs]

        @app.get("/chunks/{chunk_id}")
        async def get_chunk(chunk_id: str) -> dict[str, Any]:
            import anyio

            chunk = await anyio.to_thread.run_sync(lambda: services["get_chunk"](chunk_id))
            if chunk is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="chunk not found")
            return chunk.model_dump(mode="json")  # type: ignore[no-any-return]

        @app.get("/documents/{document_id}/markdown")
        async def get_document_markdown(document_id: str) -> dict[str, str]:
            import anyio

            try:
                markdown = await anyio.to_thread.run_sync(
                    lambda: services["get_document_markdown"](document_id)
                )
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="document not found"
                ) from None
            return {"document_id": document_id, "markdown": markdown}

        @app.delete("/documents/{document_id}")
        async def delete_document(document_id: str) -> dict[str, str]:
            import anyio

            return await anyio.to_thread.run_sync(lambda: services["delete_document"](document_id))

        @app.get("/stats")
        async def stats() -> dict[str, Any]:
            import anyio

            result = await anyio.to_thread.run_sync(services["corpus_stats"])
            return result.model_dump(mode="json")  # type: ignore[no-any-return]

    def register_cli(self, app: Typer, services: dict[str, Any], settings: LssSettings) -> None:
        """CLI commands specific to lean-lss (universal commands are added by lean-core)."""
        import json

        import typer

        @app.command()
        def ingest(path: str, json_output: bool = typer.Option(False, "--json")) -> None:
            """Ingest a PDF into the corpus."""
            result = asyncio.run(services["ingest_pdf"](path))
            if json_output:
                typer.echo(result.model_dump_json(indent=2))
            else:
                typer.echo(
                    f"ingested {result.document_id} ({result.chunk_count} chunks, "
                    f"{result.elapsed_seconds:.1f}s)"
                )

        @app.command()
        def search(
            query: str,
            k: int = typer.Option(5, help="Number of results"),
            doc_id: str = typer.Option(None, help="Filter by document UUID"),
            section: str = typer.Option(None, help="Filter by section substring"),
            author: str = typer.Option(None, help="Filter by author substring"),
            year_min: int = typer.Option(None, help="Filter by min publication year"),
            year_max: int = typer.Option(None, help="Filter by max publication year"),
            min_score: float = typer.Option(None, help="Minimum cosine similarity"),
            chunk_type: str = typer.Option(None, help="Filter by chunk type: text|image"),
            json_output: bool = typer.Option(False, "--json"),
        ) -> None:
            """Semantic + hybrid search over the corpus."""
            chunks = services["search"](
                query,
                k=k,
                doc_id=doc_id,
                section=section,
                author=author,
                year_min=year_min,
                year_max=year_max,
                min_score=min_score,
                chunk_type=chunk_type,
            )
            if json_output:
                typer.echo(
                    json.dumps([c.model_dump(mode="json") for c in chunks], indent=2, default=str)
                )
                return
            if not chunks:
                typer.echo("No results found.")
                return
            for c in chunks:
                score_str = f"[{c.score:.4f}] " if c.score else ""
                typer.echo(
                    f"{score_str}{c.section_path} (chunk {c.chunk_index})\n  {c.content[:200]}...\n"
                )

        @app.command()
        def reingest(doc_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
            """Re-ingest a document (re-extract with current settings)."""
            result = asyncio.run(services["reingest"](doc_id))
            if json_output:
                typer.echo(result.model_dump_json(indent=2))
            else:
                typer.echo(f"reingested {result.document_id} ({result.chunk_count} chunks)")

        @app.command(name="corpus-stats")
        def corpus_stats_cmd(json_output: bool = typer.Option(False, "--json")) -> None:
            """Show corpus statistics."""
            stats = services["corpus_stats"]()
            if json_output:
                typer.echo(stats.model_dump_json(indent=2, default=str))
            else:
                typer.echo(f"documents: {stats.document_count}")
                typer.echo(f"chunks:    {stats.chunk_count}")
                typer.echo(f"tokens:    {stats.total_tokens}")
                typer.echo(f"embed:     {stats.embedding_model} ({stats.embedding_dim}-dim)")

        @app.command(name="list-documents")
        def list_documents_cmd(
            json_output: bool = typer.Option(False, "--json"),
        ) -> None:
            """List all documents in the corpus."""
            docs = services["list_documents"]()
            if json_output:
                typer.echo(json.dumps([d.model_dump(mode="json") for d in docs], indent=2))
            else:
                for d in docs:
                    auth = f"  {', '.join(d.authors)}" if d.authors else ""
                    typer.echo(f"{d.id}  {d.title or '(untitled)'}  [{d.chunk_count} chunks]{auth}")

        @app.command(name="get-chunk")
        def get_chunk_cmd(
            chunk_id: str,
            json_output: bool = typer.Option(False, "--json"),
        ) -> None:
            """Retrieve a single chunk by ID."""
            chunk = services["get_chunk"](chunk_id)
            if chunk is None:
                typer.echo("Chunk not found.", err=True)
                raise typer.Exit(1)
            if json_output:
                typer.echo(chunk.model_dump_json(indent=2))
            else:
                typer.echo(chunk.content)

        @app.command(name="get-markdown")
        def get_markdown_cmd(
            doc_id: str,
            json_output: bool = typer.Option(False, "--json"),
        ) -> None:
            """Get the extracted markdown for a document."""
            md = services["get_document_markdown"](doc_id)
            if json_output:
                typer.echo(json.dumps({"doc_id": doc_id, "markdown": md}, indent=2))
            else:
                typer.echo(md)

        @app.command()
        def delete(doc_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
            """Delete a document and all its chunks."""
            result = services["delete_document"](doc_id)
            if json_output:
                typer.echo(json.dumps(result, indent=2))
            else:
                typer.echo(f"Deleted: {result['deleted']}")

        @app.command(name="reingest-all")
        def reingest_all_cmd(
            force: bool = typer.Option(False, "--force"),
            json_output: bool = typer.Option(False, "--json"),
        ) -> None:
            """Reingest all documents (smallest first, skips OCR'd unless --force)."""
            docs = sorted(services["list_documents"](), key=lambda d: d.chunk_count)
            results: list[dict[str, object]] = []
            success = failed = skipped = 0
            for doc in docs:
                if not force and doc.extraction_method == "unlimited_ocr":
                    skipped += 1
                    results.append({"id": doc.id, "status": "skipped"})
                    typer.echo(f"SKIP  {doc.id}  (already OCR'd, {doc.chunk_count} chunks)")
                    continue
                typer.echo(
                    f"START {doc.id}  ({doc.page_count or '?'} pages, was {doc.extraction_method})"
                )
                try:
                    res = asyncio.run(services["reingest"](doc.id))
                    success += 1
                    results.append(
                        {
                            "id": doc.id,
                            "status": "ok",
                            "chunks": res.chunk_count,
                            "seconds": res.elapsed_seconds,
                        }
                    )
                    typer.echo(
                        f"DONE  {doc.id}  ({res.chunk_count} chunks, {res.elapsed_seconds:.1f}s)"
                    )
                except Exception as exc:
                    failed += 1
                    results.append({"id": doc.id, "status": "failed", "error": str(exc)})
                    typer.echo(f"FAIL  {doc.id}  ({exc})")
            if json_output:
                typer.echo(
                    json.dumps(
                        {
                            "success": success,
                            "failed": failed,
                            "skipped": skipped,
                            "details": results,
                        },
                        indent=2,
                        default=str,
                    )
                )
            else:
                typer.echo(f"\nSuccess: {success}  Failed: {failed}  Skipped: {skipped}")


__all__ = ["LssDomain", "LssSettings"]
