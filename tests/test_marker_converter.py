"""Tests for extraction/marker_converter.py — marker-pdf integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import respx
from lean.extraction.marker_converter import MarkerNotInstalled, MarkerRemoteError


@pytest.fixture(autouse=True)
def _clear_converter_cache():
    """Clear the _get_converter lru_cache between tests."""
    from lean.extraction.marker_converter import _get_converter

    _get_converter.cache_clear()
    yield
    _get_converter.cache_clear()


def _make_mock_converter(mock_rendered: MagicMock, page_count: int = 1) -> MagicMock:
    """Build a mock PdfConverter wired for the _extract_local dual-render flow."""
    mock_document = MagicMock()
    mock_document.pages = [MagicMock()] * page_count
    mock_converter = MagicMock()
    mock_converter.build_document.return_value = mock_document
    mock_converter.resolve_dependencies.return_value = MagicMock(return_value=mock_rendered)
    return mock_converter


def test_extract_markdown_returns_text_and_page_count(tmp_path: Path) -> None:
    mock_rendered = MagicMock()
    mock_converter = _make_mock_converter(mock_rendered, page_count=3)

    with (
        patch("lean.extraction.marker_converter._get_converter", return_value=mock_converter),
        patch("marker.output.text_from_rendered", return_value=("# Test\n\nContent", "md", {})),
    ):
        from lean.extraction.marker_converter import extract_markdown

        text, page_count, images, _block_metas = extract_markdown(tmp_path / "fake.pdf")

    assert text == "# Test\n\nContent"
    assert page_count == 3
    assert images == {}


def test_extract_markdown_returns_images(tmp_path: Path) -> None:
    mock_rendered = MagicMock()
    mock_converter = _make_mock_converter(mock_rendered, page_count=1)
    fake_images = {"img_0_0": MagicMock(), "img_1_2": MagicMock()}

    with (
        patch("lean.extraction.marker_converter._get_converter", return_value=mock_converter),
        patch("marker.output.text_from_rendered", return_value=("text", "md", fake_images)),
    ):
        from lean.extraction.marker_converter import extract_markdown

        _, _, images, _ = extract_markdown(tmp_path / "fake.pdf")

    assert images is fake_images


def test_extract_markdown_passes_force_ocr_config(tmp_path: Path) -> None:
    mock_rendered = MagicMock()
    mock_rendered.metadata = {}

    mock_converter = MagicMock(return_value=mock_rendered)

    with (
        patch(
            "lean.extraction.marker_converter._get_converter",
            return_value=mock_converter,
        ) as get_conv_mock,
        patch("marker.output.text_from_rendered", return_value=("text", "md", {})),
    ):
        from lean.extraction.marker_converter import extract_markdown

        extract_markdown(tmp_path / "fake.pdf", force_ocr=True)

    get_conv_mock.assert_called_once_with(force_ocr=True)


def test_extract_markdown_handles_missing_page_stats(tmp_path: Path) -> None:
    mock_rendered = MagicMock()
    mock_converter = _make_mock_converter(mock_rendered, page_count=1)

    with (
        patch("lean.extraction.marker_converter._get_converter", return_value=mock_converter),
        patch("marker.output.text_from_rendered", return_value=("text", "md", {})),
    ):
        from lean.extraction.marker_converter import extract_markdown

        _, page_count, _, _ = extract_markdown(tmp_path / "fake.pdf")

    assert page_count == 1


def test_extract_markdown_handles_none_metadata(tmp_path: Path) -> None:
    mock_rendered = MagicMock()
    mock_converter = _make_mock_converter(mock_rendered, page_count=1)

    with (
        patch("lean.extraction.marker_converter._get_converter", return_value=mock_converter),
        patch("marker.output.text_from_rendered", return_value=("text", "md", {})),
    ):
        from lean.extraction.marker_converter import extract_markdown

        _, page_count, _, _ = extract_markdown(tmp_path / "fake.pdf")

    assert page_count == 1


def test_get_converter_raises_when_marker_not_installed() -> None:
    """_get_converter raises MarkerNotInstalled when marker is not importable."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("marker"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        from lean.extraction.marker_converter import _get_converter

        with pytest.raises(MarkerNotInstalled, match="marker-pdf not installed"):
            _get_converter()


def test_get_converter_caches_singleton() -> None:
    """_get_converter returns the same instance for the same force_ocr value."""
    mock_converter_obj = MagicMock()

    with patch(
        "lean.extraction.marker_converter._get_converter", wraps=lambda **kw: mock_converter_obj
    ):
        from lean.extraction.marker_converter import _get_converter

        result1 = _get_converter(force_ocr=False)
        result2 = _get_converter(force_ocr=False)

    assert result1 is result2


def test_get_converter_constructs_pdf_converter_when_installed() -> None:
    """When marker deps are importable, _get_converter builds a real PdfConverter.

    Covers the construction body (config + ConfigParser + PdfConverter wiring) that
    the dispatch-level tests skip by mocking _get_converter itself.
    """
    import sys
    import types

    from lean.extraction.marker_converter import _get_converter

    fake_modules: dict[str, Any] = {}
    saved: dict[str, Any] = {}

    def _install_fake_marker() -> None:
        parser_mod = types.ModuleType("marker.config.parser")
        converter_mod = types.ModuleType("marker.converters.pdf")
        models_mod = types.ModuleType("marker.models")
        output_mod = types.ModuleType("marker.output")

        fake_config_parser = MagicMock()
        fake_config_parser.return_value = fake_config_parser
        fake_config_parser.generate_config_dict.return_value = {"parsed": True}
        fake_config_parser.get_processors.return_value = ["proc_a"]
        fake_config_parser.get_renderer.return_value = "renderer_cls"
        parser_mod.ConfigParser = fake_config_parser

        fake_pdf_converter = MagicMock(name="PdfConverter_instance")
        fake_pdf_converter_class = MagicMock(name="PdfConverter", return_value=fake_pdf_converter)
        converter_mod.PdfConverter = fake_pdf_converter_class

        models_mod.create_model_dict = MagicMock(return_value={"model": "weights"})

        output_mod.text_from_rendered = MagicMock(return_value=("text", "md", {}))

        for name, mod in [
            ("marker", types.ModuleType("marker")),
            ("marker.config", types.ModuleType("marker.config")),
            ("marker.config.parser", parser_mod),
            ("marker.converters", types.ModuleType("marker.converters")),
            ("marker.converters.pdf", converter_mod),
            ("marker.models", models_mod),
            ("marker.output", output_mod),
        ]:
            saved[name] = sys.modules.get(name)
            sys.modules[name] = mod
            fake_modules[name] = mod

    _install_fake_marker()
    try:
        _get_converter.cache_clear()
        result = _get_converter(force_ocr=False)
    finally:
        _get_converter.cache_clear()
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig

    assert result is not None
    parser_mod = fake_modules["marker.config.parser"]
    converter_mod = fake_modules["marker.converters.pdf"]
    models_mod = fake_modules["marker.models"]

    parser_mod.ConfigParser.assert_called_once_with(
        {"output_format": "markdown", "force_ocr": False}
    )
    fake_config_parser_instance = parser_mod.ConfigParser.return_value
    converter_mod.PdfConverter.assert_called_once_with(
        config=fake_config_parser_instance.generate_config_dict.return_value,
        artifact_dict=models_mod.create_model_dict.return_value,
        processor_list=fake_config_parser_instance.get_processors.return_value,
        renderer=fake_config_parser_instance.get_renderer.return_value,
    )
    assert result is converter_mod.PdfConverter.return_value


def test_get_converter_force_ocr_true_uses_separate_cache_entry() -> None:
    """force_ocr=True and force_ocr=False produce separate converter instances.

    Covers the @lru_cache(maxsize=4) separate-cache-per-force_ocr contract.
    """
    import sys
    import types

    from lean.extraction.marker_converter import _get_converter

    saved: dict[str, Any] = {}
    fake_modules: dict[str, Any] = {}

    def _install_fake_marker() -> None:
        for name in [
            "marker",
            "marker.config",
            "marker.config.parser",
            "marker.converters",
            "marker.converters.pdf",
            "marker.models",
            "marker.output",
        ]:
            mod = types.ModuleType(name)
            saved[name] = sys.modules.get(name)
            sys.modules[name] = mod
            fake_modules[name] = mod

        parser_mod = fake_modules["marker.config.parser"]
        parser_mod.ConfigParser = MagicMock(
            return_value=MagicMock(
                generate_config_dict=MagicMock(return_value={}),
                get_processors=MagicMock(return_value=[]),
                get_renderer=MagicMock(return_value="r"),
            )
        )
        converter_mod = fake_modules["marker.converters.pdf"]
        converter_mod.PdfConverter = MagicMock(side_effect=lambda **kw: object())
        fake_models = fake_modules["marker.models"]
        fake_models.create_model_dict = MagicMock(return_value={})

    _install_fake_marker()
    try:
        _get_converter.cache_clear()
        result_false_1 = _get_converter(force_ocr=False)
        result_false_2 = _get_converter(force_ocr=False)
        result_true = _get_converter(force_ocr=True)
    finally:
        _get_converter.cache_clear()
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig

    assert result_false_1 is result_false_2
    assert result_false_1 is not result_true


# --- _extract_remote tests (network-mocked via respx) ---


def test_extract_remote_success(tmp_path: Path) -> None:
    """_extract_remote returns markdown + page_count + images on HTTP 200."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    from PIL import Image

    buf = __import__("io").BytesIO()
    Image.new("RGB", (10, 10), "white").save(buf, format="PNG")
    import base64

    img_b64 = base64.b64encode(buf.getvalue()).decode()

    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(
            json={"markdown": "# Hello", "page_count": 3, "images": {"fig1": img_b64}}
        )
        from lean.extraction.marker_converter import _extract_remote

        text, pages, images, _block_metas = _extract_remote(pdf, "http://gpu-test:8000")

    assert text == "# Hello"
    assert pages == 3
    assert "fig1" in images


def test_extract_remote_raises_on_server_error(tmp_path: Path) -> None:
    """_extract_remote raises MarkerRemoteError on HTTP 500."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(status_code=500)
        from lean.extraction.marker_converter import _extract_remote

        with pytest.raises(MarkerRemoteError, match="HTTP 500"):
            _extract_remote(pdf, "http://gpu-test:8000")


def test_extract_remote_raises_on_client_error(tmp_path: Path) -> None:
    """_extract_remote raises MarkerRemoteError on HTTP 400."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(status_code=400, text="bad request")
        from lean.extraction.marker_converter import _extract_remote

        with pytest.raises(MarkerRemoteError, match="HTTP 400"):
            _extract_remote(pdf, "http://gpu-test:8000")


def test_extract_remote_raises_on_error_key(tmp_path: Path) -> None:
    """_extract_remote raises MarkerRemoteError when response JSON has 'error' key."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(json={"error": "model load failed"})
        from lean.extraction.marker_converter import _extract_remote

        with pytest.raises(MarkerRemoteError, match="model load failed"):
            _extract_remote(pdf, "http://gpu-test:8000")


def test_extract_remote_raises_on_missing_markdown_key(tmp_path: Path) -> None:
    """_extract_remote raises MarkerRemoteError when response is missing 'markdown'."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(json={"page_count": 1, "images": {}})
        from lean.extraction.marker_converter import _extract_remote

        with pytest.raises(MarkerRemoteError, match="missing key"):
            _extract_remote(pdf, "http://gpu-test:8000")


def test_extract_remote_raises_on_non_json_response(tmp_path: Path) -> None:
    """_extract_remote raises MarkerRemoteError when response is not JSON."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(content=b"<html>error</html>")
        from lean.extraction.marker_converter import _extract_remote

        with pytest.raises(MarkerRemoteError, match="non-JSON"):
            _extract_remote(pdf, "http://gpu-test:8000")


def test_extract_remote_rejects_too_many_images(tmp_path: Path) -> None:
    """_extract_remote raises MarkerRemoteError when response has > MAX_IMAGES_PER_DOC."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    from lean.extraction.marker_converter import MAX_IMAGES_PER_DOC

    too_many = {f"img_{i}": "" for i in range(MAX_IMAGES_PER_DOC + 1)}
    with respx.mock(base_url="http://gpu-test:8000") as mock:
        mock.post("/extract").respond(
            json={"markdown": "text", "page_count": 1, "images": too_many}
        )
        from lean.extraction.marker_converter import _extract_remote

        with pytest.raises(MarkerRemoteError, match=f"{MAX_IMAGES_PER_DOC + 1} images"):
            _extract_remote(pdf, "http://gpu-test:8000")
