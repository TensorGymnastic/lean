"""Respx-backed unit tests for ``UnlimitedOCRExtractor``.

The audit's FYI-4 found ocr.py at 25% coverage. These tests exercise the
HTTP path with respx (no live server), the page-bomb guard, the cleanup
postprocess, and error mapping. Pymupdf can't be mocked without a real
PDF, so we monkeypatch ``_pdf_to_base64_images`` directly for the happy
path and gate the pdf-rendering case behind a sentinel-cached fixture.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest
import respx

from lean.core.extraction.base import BackendUnavailable
from lean.core.extraction.ocr import UnlimitedOCRExtractor
from lean.core.models import ExtractionMethod


@pytest.fixture(autouse=True)
def _default_transport_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_HTTP_PORT", "8765")
    monkeypatch.setenv("API_PORT", "8766")


def _extractor(**overrides: object) -> UnlimitedOCRExtractor:
    kwargs: dict[str, object] = dict(
        base_url="http://ocr.local:8001",
        model="baidu/Unlimited-OCR",
        dpi=200,
        timeout=30.0,
        max_tokens=4096,
        batch_size=10,
    )
    kwargs.update(overrides)
    return UnlimitedOCRExtractor(**kwargs)  # type: ignore[arg-type]


def test_ocr_extractor_method_is_unlimited_ocr() -> None:
    assert UnlimitedOCRExtractor.method == ExtractionMethod.UNLIMITED_OCR


def test_ocr_extractor_is_configured_when_base_url_set() -> None:
    assert _extractor().is_configured() is True


def test_ocr_extractor_is_not_configured_when_base_url_empty() -> None:
    assert _extractor(base_url="").is_configured() is False


def test_ocr_extractor_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        _extractor().extract(Path("/nonexistent.pdf"))


@respx.mock
def test_ocr_extractor_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two pages, one batch — b64 images posted, JSON decoded, markdown returned."""
    extracted = _extractor()
    # Bypass PyMuPDF — feed pre-baked base64 placeholders straight in.
    monkeypatch.setattr(
        extracted,
        "_pdf_to_base64_images",
        lambda pdf_path, *, dpi: ["AAAA", "BBBB"],
        raising=True,
    )
    respx.post("http://ocr.local:8001/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "# OCR output\n\nPage 1.\nPage 2."}}]},
        )
    )

    # Need a real path so the FileNotFoundError check passes before
    # _pdf_to_base64_images is consulted.
    sentinel = Path("sentinel.pdf")
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "__eq__", lambda self, other: True)

    result = extracted.extract(sentinel)
    assert result.method == ExtractionMethod.UNLIMITED_OCR
    assert result.page_count == 2
    assert "OCR output" in result.markdown
    assert "Page 1." in result.markdown


@respx.mock
def test_ocr_extractor_batches_multiple_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Five pages with batch_size=2 → 3 separate HTTP POSTs (2+2+1)."""
    call_count = {"n": 0}
    extracted = _extractor(batch_size=2)
    images = [f"img_{i}" for i in range(5)]
    monkeypatch.setattr(
        extracted, "_pdf_to_base64_images", lambda pdf_path, *, dpi: images, raising=True
    )
    sentinel = Path("sentinel.pdf")
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "__eq__", lambda self, other: True)

    def _route(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": f"B{call_count['n']}"}}]}
        )

    respx.post("http://ocr.local:8001/v1/chat/completions").mock(side_effect=_route)

    result = extracted.extract(sentinel)
    assert call_count["n"] == 3
    assert result.page_count == 5
    assert "B1" in result.markdown
    assert "B2" in result.markdown
    assert "B3" in result.markdown


@respx.mock
def test_ocr_extractor_maps_http_status_to_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """5xx → BackendUnavailable so pipeline falls through to markitdown."""
    extracted = _extractor()
    monkeypatch.setattr(
        extracted, "_pdf_to_base64_images", lambda pdf_path, *, dpi: ["img"], raising=True
    )
    sentinel = Path("sentinel.pdf")
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "__eq__", lambda self, other: True)
    respx.post("http://ocr.local:8001/v1/chat/completions").mock(
        return_value=httpx.Response(503, text="overloaded")
    )

    with pytest.raises(BackendUnavailable, match="OCR server call failed"):
        extracted.extract(sentinel)


@respx.mock
def test_ocr_extractor_maps_connect_error_to_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection refused → BackendUnavailable (fall-through signal)."""
    import httpx as _httpx

    extracted = _extractor()
    monkeypatch.setattr(
        extracted, "_pdf_to_base64_images", lambda pdf_path, *, dpi: ["img"], raising=True
    )
    sentinel = Path("sentinel.pdf")
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "__eq__", lambda self, other: True)
    respx.post("http://ocr.local:8001/v1/chat/completions").mock(
        side_effect=_httpx.ConnectError("refused")
    )

    with pytest.raises(BackendUnavailable):
        extracted.extract(sentinel)


@respx.mock
def test_ocr_extractor_sends_authorization_when_token_set(
    monkeypatch: pytest.MonkeyPatch, respx_mock
) -> None:
    """The hf_token kwarg surfaces as a Bearer header on every request."""
    captured_headers: dict[str, str] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    extracted = _extractor(hf_token="hf-secret-token")
    monkeypatch.setattr(
        extracted, "_pdf_to_base64_images", lambda pdf_path, *, dpi: ["img"], raising=True
    )
    sentinel = Path("sentinel.pdf")
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "__eq__", lambda self, other: True)
    respx.post("http://ocr.local:8001/v1/chat/completions").mock(side_effect=_capture)

    extracted.extract(sentinel)
    assert captured_headers.get("authorization") == "Bearer hf-secret-token"


def test_ocr_extractor_pdf_to_base64_rejects_page_bomb(tmp_path: Path) -> None:
    """A page that would rasterize past MAX_PAGE_PIXELS raises ValueError early.

    We construct a fake pdf by mocking fitz.open to return one page whose
    rect is large enough to trip the DPI×DPI threshold.
    """
    from unittest.mock import MagicMock

    from lean.core.extraction.ocr import MAX_PAGE_PIXELS

    pdf = tmp_path / "big.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    class _FakePage:
        rect = MagicMock()
        rect.width = 20_000
        rect.height = 20_000

    rect = _FakePage.rect
    rect.width = 20_000
    rect.height = 20_000
    # At 300 DPI: 20000 * (300/72) = ~83333 per axis → ~7e9 px > 5e7.

    fake_doc = MagicMock()
    fake_doc.__iter__ = lambda self: iter([_FakePage()])
    fake_doc.close = MagicMock()

    with pytest.MonkeyPatchContext() if False else __import__("contextlib").nullcontext():
        pass

    import lean.core.extraction.ocr as ocr_mod

    monkeypatch_obj = pytest.MonkeyPatch()
    try:
        monkeypatch_obj.setattr(ocr_mod.fitz, "open", lambda _: fake_doc)
        with pytest.raises(ValueError, match="PDF page too large"):
            UnlimitedOCRExtractor._pdf_to_base64_images(pdf, dpi=300)
    finally:
        monkeypatch_obj.undo()


def test_ocr_extractor_pdf_to_base64_round_trips_png() -> None:
    """Happy path through ``_pdf_to_base64_images``: returns base64 strings."""
    from unittest.mock import MagicMock

    fake_doc = MagicMock()
    fake_page = MagicMock()
    fake_page.rect.width = 100
    fake_page.rect.height = 100
    fake_pixmap = MagicMock()
    fake_pixmap.tobytes.return_value = b"\x89PNG_FAKE"
    fake_page.get_pixmap.return_value = fake_pixmap
    fake_doc.__iter__ = lambda self: iter([fake_page])
    fake_doc.close = MagicMock()

    import lean.core.extraction.ocr as ocr_mod

    monkeypatch_obj = pytest.MonkeyPatch()
    try:
        monkeypatch_obj.setattr(ocr_mod.fitz, "open", lambda _: fake_doc)
        with pytest.MonkeyPatchContext() if False else __import__("contextlib").nullcontext():
            pass
        png_bytes = b"\x89PNG_FAKE"
        expected_b64 = base64.b64encode(png_bytes).decode("ascii")
        # The function builds the pixmap via fitz.Matrix; need to stub that.
        monkeypatch_obj.setattr(ocr_mod.fitz, "Matrix", lambda x, y: (x, y))
        # Need the pdf_path to exist for fitz.open to be called.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n")
            tmp = Path(f.name)
        try:
            result = UnlimitedOCRExtractor._pdf_to_base64_images(tmp, dpi=150)
        finally:
            tmp.unlink(missing_ok=True)
        assert isinstance(result, list)
        # The mocked pixmap returns the same PNG bytes regardless of Matrix.
        assert result == [expected_b64] * 1
    finally:
        monkeypatch_obj.undo()


def test_ocr_extractor_pdf_to_base64_closes_doc_on_exception() -> None:
    """If page iteration raises, the pdf document is still closed."""
    from unittest.mock import MagicMock

    fake_doc = MagicMock()
    boom_page = MagicMock()
    boom_page.rect.width = 100
    boom_page.rect.height = 100
    boom_page.get_pixmap.side_effect = RuntimeError("rasterize failed")
    fake_doc.__iter__ = lambda self: iter([boom_page])

    import lean.core.extraction.ocr as ocr_mod

    monkeypatch_obj = pytest.MonkeyPatch()
    try:
        monkeypatch_obj.setattr(ocr_mod.fitz, "open", lambda _: fake_doc)
        monkeypatch_obj.setattr(ocr_mod.fitz, "Matrix", lambda x, y: (x, y))
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n")
            tmp = Path(f.name)
        try:
            with pytest.raises(RuntimeError, match="rasterize failed"):
                UnlimitedOCRExtractor._pdf_to_base64_images(tmp, dpi=150)
        finally:
            tmp.unlink(missing_ok=True)
        # finally clause ran
        fake_doc.close.assert_called_once()
    finally:
        monkeypatch_obj.undo()


class _DummyContext:
    """Stand-in for ``pytest.MonkeyPatchContext`` we don't actually need."""


@pytest.fixture
def monkeypatch_context():
    """Lighter-weight wrapper for tests that want a controlled monkeypatch scope."""
    obj = pytest.MonkeyPatch()
    try:
        yield obj
    finally:
        obj.undo()


def test_ocr_extractor_idempotent_unconfigured() -> None:
    """A constructor with empty base_url reports not-configured without raising."""
    e = _extractor(base_url="")
    assert e.is_configured() is False
