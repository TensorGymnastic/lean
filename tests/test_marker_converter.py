"""Tests for ``MarkerExtractor`` in core/extraction/marker.py.

Covers the universal ``Extractor`` protocol behaviour plus the
``_extract_remote`` HTTP path (mocked via respx).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import patch

import pytest
import respx

from lean.core.extraction.base import ExtractorUnavailable
from lean.core.extraction.marker import (
    MAX_IMAGES_PER_DOC,
    MarkerExtractor,
    _extract_remote,
)


def test_marker_extractor_is_not_configured_when_marker_missing() -> None:
    """``is_configured`` returns False when marker-pdf is not installed."""
    with patch.dict("sys.modules", {"marker": None}):
        ext = MarkerExtractor(remote_url="")
        assert ext.is_configured() is False


def test_marker_extractor_is_configured_when_remote_url_set() -> None:
    """A non-empty ``remote_url`` makes ``is_configured`` return True regardless."""
    ext = MarkerExtractor(remote_url="http://gpu:8000")
    assert ext.is_configured() is True


def test_marker_extractor_method_is_marker() -> None:
    """``MarkerExtractor.method`` is the canonical enum value."""
    from lean.core.models import ExtractionMethod

    assert MarkerExtractor.method == ExtractionMethod.MARKER


def test_extract_remote_success(tmp_path: Path) -> None:
    """``_extract_remote`` returns ExtractionResult on HTTP 200."""
    from PIL import Image

    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(
            json={"markdown": "# Hello", "page_count": 3, "images": {"fig1": img_b64}}
        )
        result = _extract_remote(pdf, "http://gpu-test:8000")

    assert result.markdown == "# Hello"
    assert result.page_count == 3
    assert "fig1" in result.images


def test_extract_remote_raises_on_server_error(tmp_path: Path) -> None:
    """``_extract_remote`` raises ``ExtractorUnavailable`` on HTTP 500."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(status_code=500)
        with pytest.raises(ExtractorUnavailable, match="HTTP 500"):
            _extract_remote(pdf, "http://gpu-test:8000")


def test_extract_remote_raises_on_client_error(tmp_path: Path) -> None:
    """``_extract_remote`` raises ``ExtractorUnavailable`` on HTTP 400."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(status_code=400, text="bad request")
        with pytest.raises(ExtractorUnavailable, match="HTTP 400"):
            _extract_remote(pdf, "http://gpu-test:8000")


def test_extract_remote_raises_on_error_key(tmp_path: Path) -> None:
    """``_extract_remote`` raises ``ExtractorUnavailable`` when response JSON has 'error' key."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(json={"error": "model load failed"})
        with pytest.raises(ExtractorUnavailable, match="model load failed"):
            _extract_remote(pdf, "http://gpu-test:8000")


def test_extract_remote_raises_on_missing_markdown_key(tmp_path: Path) -> None:
    """``_extract_remote`` raises ``ExtractorUnavailable`` when response is missing 'markdown'."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(json={"page_count": 1, "images": {}})
        with pytest.raises(ExtractorUnavailable, match="missing key"):
            _extract_remote(pdf, "http://gpu-test:8000")


def test_extract_remote_raises_on_non_json_response(tmp_path: Path) -> None:
    """``_extract_remote`` raises ``ExtractorUnavailable`` when response is not JSON."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(content=b"<html>error</html>")
        with pytest.raises(ExtractorUnavailable, match="non-JSON"):
            _extract_remote(pdf, "http://gpu-test:8000")


def test_extract_remote_rejects_too_many_images(tmp_path: Path) -> None:
    """_extract_remote raises when response has more than MAX_IMAGES_PER_DOC images."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    too_many = {f"img_{i}": "" for i in range(MAX_IMAGES_PER_DOC + 1)}
    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(
            json={"markdown": "text", "page_count": 1, "images": too_many}
        )
        with pytest.raises(ExtractorUnavailable, match=f"{MAX_IMAGES_PER_DOC + 1} images"):
            _extract_remote(pdf, "http://gpu-test:8000")


def test_extract_remote_accepts_exactly_max_images(tmp_path: Path) -> None:
    """``_extract_remote`` accepts exactly ``MAX_IMAGES_PER_DOC`` images (boundary)."""
    from PIL import Image

    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    buf = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    images = {f"img_{i}": img_b64 for i in range(MAX_IMAGES_PER_DOC)}
    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(json={"markdown": "text", "page_count": 1, "images": images})
        result = _extract_remote(pdf, "http://gpu-test:8000")

    assert len(result.images) == MAX_IMAGES_PER_DOC
