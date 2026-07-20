"""Unit tests for the web (URL-scraped) domain adapter.

The audit's FYI-1 found these domains had zero unit tests. These tests
pin ``WebPageExtractor`` behavior using ``respx`` to mock ``httpx`` so
network isn't exercised: HTML fetches are converted via the regex
fallback (no trafilatura dep required); a non-HTML content-type is
wrapped as a fenced body; an empty page surfaces as a placeholder;
an unreachable URL raises ``RuntimeError``; a non-URL path raises
``ValueError``.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from lean.core.models import ExtractionMethod


@pytest.fixture(autouse=True)
def _default_transport_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_HTTP_PORT", "8765")
    monkeypatch.setenv("API_PORT", "8766")


def test_web_extractor_method_is_markitdown() -> None:
    """Web pages run through the markitdown-style adapter slot."""
    from lean.domains.web.adapters import WebPageExtractor

    assert WebPageExtractor().method == ExtractionMethod.MARKITDOWN


def test_web_extractor_is_always_configured() -> None:
    """No external preconditions beyond network access."""
    from lean.domains.web.adapters import WebPageExtractor

    assert WebPageExtractor().is_configured() is True


def test_web_extractor_rejects_non_url() -> None:
    """A bare filesystem path is rejected so callers fail fast."""
    from lean.domains.web.adapters import WebPageExtractor

    with pytest.raises(ValueError, match="not a valid URL"):
        WebPageExtractor().extract(Path("/tmp/local.html"))  # noqa: S108


@respx.mock
def test_web_extractor_fetches_html_and_extracts_title() -> None:
    """HTML pages: title is parsed via the regex fallback; body becomes text."""
    from lean.domains.web.adapters import WebPageExtractor

    html = "<html><head><title>Hello</title></head><body><p>world</p></body></html>"
    respx.get("https://example.com/").mock(
        return_value=httpx.Response(200, text=html, headers={"content-type": "text/html"})
    )

    result = WebPageExtractor().extract(Path("https://example.com/"))
    assert result.page_count == 1
    assert "# Hello" in result.markdown
    assert "world" in result.markdown


@respx.mock
def test_web_extractor_wraps_non_html_body() -> None:
    """JSON / text content-type: raw body inside a fenced code block."""
    from lean.domains.web.adapters import WebPageExtractor

    respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(
            200,
            text='{"key": "value"}',
            headers={"content-type": "application/json"},
        )
    )

    result = WebPageExtractor().extract(Path("https://api.example.com/data"))
    assert "# https://api.example.com/data" in result.markdown
    assert "```" in result.markdown
    assert '"key": "value"' in result.markdown


@respx.mock
def test_web_extractor_wraps_non_html_plain_text() -> None:
    """Plain text content-type: still wrapped (no markdown conversion)."""
    from lean.domains.web.adapters import WebPageExtractor

    respx.get("https://example.com/robots.txt").mock(
        return_value=httpx.Response(
            200,
            text="User-agent: *\nDisallow: /",
            headers={"content-type": "text/plain"},
        )
    )

    result = WebPageExtractor().extract(Path("https://example.com/robots.txt"))
    assert "User-agent: *" in result.markdown
    assert "# https://example.com/robots.txt" in result.markdown


@respx.mock
def test_web_extractor_raises_on_http_error() -> None:
    """A 5xx response surfaces as ``RuntimeError`` so callers can retry."""
    from lean.domains.web.adapters import WebPageExtractor

    respx.get("https://example.com/down").mock(return_value=httpx.Response(503, text="overloaded"))

    with pytest.raises(RuntimeError, match="fetch failed for https://example.com/down"):
        WebPageExtractor().extract(Path("https://example.com/down"))


@respx.mock
def test_web_extractor_raises_on_connection_error() -> None:
    """Network exception (DNS, timeout) surfaces as ``RuntimeError``."""
    import httpx as _httpx

    from lean.domains.web.adapters import WebPageExtractor

    respx.get("https://broken.invalid/").mock(side_effect=_httpx.ConnectError("nope"))

    with pytest.raises(RuntimeError, match="fetch failed for https://broken.invalid"):
        WebPageExtractor().extract(Path("https://broken.invalid/"))


@respx.mock
def test_web_extractor_html_strips_scripts_and_styles() -> None:
    """Script + style blocks are removed before markdown conversion."""
    from lean.domains.web.adapters import WebPageExtractor

    html = (
        "<html><head>"
        "<style>body {color: red}</style>"
        "</head><body>"
        "<script>alert('x')</script>"
        "<p>visible body</p>"
        "</body></html>"
    )
    respx.get("https://example.com/").mock(
        return_value=httpx.Response(200, text=html, headers={"content-type": "text/html"})
    )

    result = WebPageExtractor().extract(Path("https://example.com/"))
    assert "alert" not in result.markdown
    assert "color" not in result.markdown
    assert "visible body" in result.markdown


@respx.mock
def test_web_extractor_falls_back_to_url_when_no_title() -> None:
    """HTML with no <title> falls back to the URL as the section heading."""
    from lean.domains.web.adapters import WebPageExtractor

    html = "<html><body><p>content only</p></body></html>"
    respx.get("https://no-title.example.com/").mock(
        return_value=httpx.Response(200, text=html, headers={"content-type": "text/html"})
    )

    result = WebPageExtractor().extract(Path("https://no-title.example.com/"))
    assert result.markdown.startswith("# https://no-title.example.com")


def test_web_extractor_timeout_configurable() -> None:
    """The ``timeout_s`` constructor kwarg is recorded on the instance."""
    from lean.domains.web.adapters import WebPageExtractor

    assert WebPageExtractor(timeout_s=10.0)._timeout == 10.0
    assert WebPageExtractor()._timeout == 30.0  # default
