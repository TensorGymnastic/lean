"""Coverage tests for the Extraction Pipeline base + adapter classes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lean.core.extraction.base import (
    BackendUnavailable,
    ExtractorUnavailable,
    Pipeline,
)
from lean.core.extraction.markitdown import MarkitdownExtractor
from lean.core.models import ExtractionMethod


def test_pipeline_raises_on_empty_extractor_list() -> None:
    """Pipeline with no extractors raises ValueError at construction."""
    with pytest.raises(ValueError, match="at least one extractor"):
        Pipeline([])


def test_pipeline_uses_first_configured_extractor(tmp_path: Path) -> None:
    """Pipeline returns the first configured extractor's result."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"fake")
    extractor1 = MagicMock()
    extractor1.is_configured.return_value = True
    extractor1.extract.return_value = MagicMock(markdown="hello")
    pipeline = Pipeline([extractor1])
    result = pipeline.extract(pdf)
    assert result.markdown == "hello"


def test_pipeline_falls_through_when_first_not_configured(tmp_path: Path) -> None:
    """If the first extractor reports ``is_configured() == False``, try the next."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"fake")
    extractor1 = MagicMock()
    extractor1.is_configured.return_value = False
    extractor2 = MagicMock()
    extractor2.is_configured.return_value = True
    extractor2.extract.return_value = MagicMock(markdown="from #2")
    pipeline = Pipeline([extractor1, extractor2])
    result = pipeline.extract(pdf)
    assert result.markdown == "from #2"
    extractor1.extract.assert_not_called()
    extractor2.extract.assert_called_once()


def test_pipeline_falls_through_on_backend_unavailable(tmp_path: Path) -> None:
    """Pipeline catches ``BackendUnavailable`` and tries the next extractor."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"fake")
    extractor1 = MagicMock()
    extractor1.is_configured.return_value = True
    extractor1.extract.side_effect = BackendUnavailable("first failed")
    extractor2 = MagicMock()
    extractor2.is_configured.return_value = True
    extractor2.extract.return_value = MagicMock(markdown="from #2")
    pipeline = Pipeline([extractor1, extractor2])
    result = pipeline.extract(pdf)
    assert result.markdown == "from #2"


def test_pipeline_falls_through_on_extractor_unavailable(tmp_path: Path) -> None:
    """Pipeline catches ``ExtractorUnavailable`` and tries the next extractor."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"fake")
    extractor1 = MagicMock()
    extractor1.is_configured.return_value = True
    extractor1.extract.side_effect = ExtractorUnavailable("first failed")
    extractor2 = MagicMock()
    extractor2.is_configured.return_value = True
    extractor2.extract.return_value = MagicMock(markdown="from #2")
    pipeline = Pipeline([extractor1, extractor2])
    result = pipeline.extract(pdf)
    assert result.markdown == "from #2"


def test_pipeline_raises_when_all_extractors_fail(tmp_path: Path) -> None:
    """If every configured extractor fails, the last error propagates."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"fake")
    extractor1 = MagicMock()
    extractor1.is_configured.return_value = True
    extractor1.extract.side_effect = BackendUnavailable("first")
    extractor2 = MagicMock()
    extractor2.is_configured.return_value = True
    extractor2.extract.side_effect = BackendUnavailable("second")
    pipeline = Pipeline([extractor1, extractor2])
    with pytest.raises(BackendUnavailable, match="second"):
        pipeline.extract(pdf)


def test_pipeline_raises_when_no_extractor_configured(tmp_path: Path) -> None:
    """If every extractor reports not configured, Pipeline raises RuntimeError."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"fake")
    extractor1 = MagicMock()
    extractor1.is_configured.return_value = False
    pipeline = Pipeline([extractor1])
    with pytest.raises(RuntimeError, match="all extractors exhausted"):
        pipeline.extract(pdf)


# ---------- MarkitdownExtractor ----------


def test_markitdown_extractor_method_is_markitdown() -> None:
    """MarkitdownExtractor.method is ExtractionMethod.MARKITDOWN."""
    assert MarkitdownExtractor().method == ExtractionMethod.MARKITDOWN


def test_markitdown_extractor_is_always_configured() -> None:
    """Markitdown has no external deps; is_configured() always returns True."""
    with patch.dict("sys.modules", {"markitdown": MagicMock()}):
        assert MarkitdownExtractor().is_configured() is True


def test_markitdown_extractor_raises_on_missing_file(tmp_path: Path) -> None:
    """Extracting from a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        MarkitdownExtractor().extract(tmp_path / "nope.pdf")
