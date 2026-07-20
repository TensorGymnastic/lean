"""PDF extraction backend adapters for the pdf_lss domain.

Each class is a thin wrapper around a lean ``Extractor`` with
the LSS-specific defaults (model names, timeouts, etc.) pre-baked.
The YAML manifest references these by dotted path.
"""

from __future__ import annotations

from lean.core.extraction import MarkerExtractor as _MarkerExtractor
from lean.core.extraction import MarkitdownExtractor as _MarkitdownExtractor
from lean.core.extraction import UnlimitedOCRExtractor as _UnlimitedOCRExtractor


class MarkerAdapter(_MarkerExtractor):
    """Marker-pdf with LSS defaults."""

    def __init__(
        self,
        *,
        force_ocr: bool = False,
        remote_url: str = "",
    ) -> None:
        super().__init__(force_ocr=force_ocr, remote_url=remote_url)


class UnlimitedOCRAdapter(_UnlimitedOCRExtractor):
    """baidu/Unlimited-OCR with LSS defaults."""

    def __init__(
        self,
        *,
        base_url: str,
        hf_token: str | None = None,
        model: str = "baidu/Unlimited-OCR",
        dpi: int = 300,
        timeout: float = 1800.0,
        max_tokens: int = 32768,
        batch_size: int = 20,
    ) -> None:
        super().__init__(
            base_url=base_url,
            hf_token=hf_token,
            model=model,
            dpi=dpi,
            timeout=timeout,
            max_tokens=max_tokens,
            batch_size=batch_size,
        )


class MarkitdownAdapter(_MarkitdownExtractor):
    """Microsoft MarkItDown — pure-Python fallback."""

    def __init__(self) -> None:
        super().__init__()
