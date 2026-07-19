"""Universal extraction pipeline + backends.

Domain plugins compose these into a ``Pipeline([...])`` ordered by
preference (e.g. marker → OCR → markitdown for LSS PDFs).
"""

from lean.core.extraction.base import (
    BackendUnavailable,
    ExtractionResult,
    Extractor,
    ExtractorUnavailable,
    Pipeline,
    clear_pipeline_cache,
    get_pipeline,
    set_pipeline,
)
from lean.core.extraction.marker import BlockMeta, MarkerExtractor
from lean.core.extraction.markitdown import MarkitdownExtractor
from lean.core.extraction.metadata import PdfMetadata, extract_metadata
from lean.core.extraction.ocr import UnlimitedOCRExtractor
from lean.core.models import ExtractionMethod

__all__ = [
    "BackendUnavailable",
    "ExtractorUnavailable",
    "ExtractionResult",
    "Extractor",
    "Pipeline",
    "set_pipeline",
    "get_pipeline",
    "clear_pipeline_cache",
    "BlockMeta",
    "MarkerExtractor",
    "UnlimitedOCRExtractor",
    "MarkitdownExtractor",
    "PdfMetadata",
    "extract_metadata",
    "ExtractionMethod",
]
