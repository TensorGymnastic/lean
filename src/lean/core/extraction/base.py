"""Universal extraction pipeline.

The ``Extractor`` Protocol defines the contract every backend
implements. ``Pipeline`` runs a sequence of extractors, falling
through on typed exceptions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from lean.core.models import ExtractionMethod

logger = logging.getLogger(__name__)


class BackendUnavailable(RuntimeError):
    """Raised when an extraction backend cannot be reached or is not installed.

    The pipeline falls through to the next backend when this is raised.
    """


class ExtractorUnavailable(BackendUnavailable):
    """Raised when an extraction backend is reachable but a request failed."""


class ExtractionResult:
    """The output of a successful extraction.

    ``method`` records which extractor produced the result so the
    document row can store it. ``images`` is backend-specific
    (marker returns PIL.Image dicts; OCR returns empty; markitdown
    returns empty). ``block_metas`` carries provenance metadata for
    chunk↔block alignment (empty on remote backends).
    """

    def __init__(
        self,
        *,
        markdown: str,
        page_count: int,
        method: ExtractionMethod,
        images: dict[str, Any] | None = None,
        block_metas: list[Any] | None = None,
    ) -> None:
        self.markdown = markdown
        self.page_count = page_count
        self.method = method
        self.images = images or {}
        self.block_metas = block_metas or []


@runtime_checkable
class Extractor(Protocol):
    """Contract every extraction backend implements.

    Implementations should:
    - Raise ``BackendUnavailable`` when the backend cannot be reached
      (missing deps, offline server).
    - Raise ``ExtractorUnavailable`` when the backend is reachable
      but a single request fails.
    - Return an ``ExtractionResult`` on success.
    """

    @property
    def method(self) -> ExtractionMethod:
        """The ExtractionMethod value to record on documents produced by this backend."""
        ...

    def is_configured(self) -> bool:
        """Whether this backend is configured (e.g. remote_url set).

        Always-configured backends return True.
        """
        ...

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Run extraction on a single PDF. Returns ExtractionResult on success."""
        ...


class Pipeline:
    """Runs a sequence of extractors, falling through on typed exceptions.

    The first configured extractor wins. On ``BackendUnavailable`` or
    ``ExtractorUnavailable`` the next extractor is tried. Any other
    exception propagates.
    """

    def __init__(self, extractors: list[Extractor]) -> None:
        if not extractors:
            raise ValueError("Pipeline requires at least one extractor")
        self._extractors = list(extractors)

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Try each configured extractor in order. Returns the first success."""
        last_error: Exception | None = None
        for extractor in self._extractors:
            if not extractor.is_configured():
                continue
            try:
                result = extractor.extract(pdf_path)
                logger.info(
                    "extraction: %s via %s (%d pages, %d chars)",
                    pdf_path.name,
                    extractor.method.value,
                    result.page_count,
                    len(result.markdown),
                )
                return result
            except BackendUnavailable as exc:
                logger.warning(
                    "extractor %s unavailable (%s); trying next backend",
                    extractor.method.value,
                    exc,
                )
                last_error = exc
                continue
        raise BackendUnavailable(f"all extractors exhausted for {pdf_path.name}: {last_error}")


_pipeline_cache: Pipeline | None = None


def set_pipeline(pipeline: Pipeline) -> None:
    """Install the domain's pipeline as the singleton.

    Called once at startup by the domain's transport-factory entrypoint.
    Subsequent calls replace the previous pipeline (used in tests).
    """
    global _pipeline_cache
    _pipeline_cache = pipeline


def get_pipeline() -> Pipeline:
    """Return the singleton pipeline, or raise if none is registered.

    Domains must call ``set_pipeline()`` at startup. The ingestion
    service reads from here so it doesn't need to know which backends
    are configured.
    """
    if _pipeline_cache is None:
        raise RuntimeError(
            "no extraction pipeline registered. "
            "Call lean.core.extraction.set_pipeline(...) at startup."
        )
    return _pipeline_cache


def clear_pipeline_cache() -> None:
    """Reset the pipeline singleton — for tests."""
    global _pipeline_cache
    _pipeline_cache = None


__all__ = [
    "BackendUnavailable",
    "ExtractorUnavailable",
    "ExtractionResult",
    "Extractor",
    "Pipeline",
    "set_pipeline",
    "get_pipeline",
    "clear_pipeline_cache",
]
