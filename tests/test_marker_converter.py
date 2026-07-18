"""Tests for extraction/marker_converter.py — marker-pdf integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lean.extraction.marker_converter import MarkerNotInstalled


@pytest.fixture(autouse=True)
def _clear_converter_cache():
    """Clear the _get_converter lru_cache between tests."""
    from lean.extraction.marker_converter import _get_converter

    _get_converter.cache_clear()
    yield
    _get_converter.cache_clear()


def test_extract_markdown_returns_text_and_page_count(tmp_path: Path) -> None:
    """extract_markdown returns (markdown, page_count) from the rendered output."""
    mock_rendered = MagicMock()
    mock_rendered.metadata = {"page_stats": [{"page_id": 0}, {"page_id": 1}, {"page_id": 2}]}

    mock_converter = MagicMock(return_value=mock_rendered)

    with (
        patch("lean.extraction.marker_converter._get_converter", return_value=mock_converter),
        patch("marker.output.text_from_rendered", return_value=("# Test\n\nContent", None, None)),
    ):
        from lean.extraction.marker_converter import extract_markdown

        text, page_count = extract_markdown(tmp_path / "fake.pdf")

    assert text == "# Test\n\nContent"
    assert page_count == 3
    mock_converter.assert_called_once()


def test_extract_markdown_passes_force_ocr_config(tmp_path: Path) -> None:
    """force_ocr=True is passed to _get_converter for converter construction."""
    mock_rendered = MagicMock()
    mock_rendered.metadata = {}

    mock_converter = MagicMock(return_value=mock_rendered)

    with (
        patch(
            "lean.extraction.marker_converter._get_converter",
            return_value=mock_converter,
        ) as get_conv_mock,
        patch("marker.output.text_from_rendered", return_value=("text", None, None)),
    ):
        from lean.extraction.marker_converter import extract_markdown

        extract_markdown(tmp_path / "fake.pdf", force_ocr=True)

    get_conv_mock.assert_called_once_with(force_ocr=True)


def test_extract_markdown_handles_missing_page_stats(tmp_path: Path) -> None:
    """When metadata lacks page_stats, page_count defaults to 1."""
    mock_rendered = MagicMock()
    mock_rendered.metadata = {}

    mock_converter = MagicMock(return_value=mock_rendered)

    with (
        patch("lean.extraction.marker_converter._get_converter", return_value=mock_converter),
        patch("marker.output.text_from_rendered", return_value=("text", None, None)),
    ):
        from lean.extraction.marker_converter import extract_markdown

        _, page_count = extract_markdown(tmp_path / "fake.pdf")

    assert page_count == 1


def test_extract_markdown_handles_none_metadata(tmp_path: Path) -> None:
    """When rendered.metadata is None, page_count defaults to 1."""
    mock_rendered = MagicMock()
    mock_rendered.metadata = None

    mock_converter = MagicMock(return_value=mock_rendered)

    with (
        patch("lean.extraction.marker_converter._get_converter", return_value=mock_converter),
        patch("marker.output.text_from_rendered", return_value=("text", None, None)),
    ):
        from lean.extraction.marker_converter import extract_markdown

        _, page_count = extract_markdown(tmp_path / "fake.pdf")

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
