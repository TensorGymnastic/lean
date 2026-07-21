"""URL → markdown extraction for the web domain.

Uses ``trafilatura`` for clean text extraction (optional dep).
Falls back to raw HTML if trafilatura isn't installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from lean.core.extraction.base import ExtractionResult
from lean.core.models import ExtractionMethod

logger = logging.getLogger(__name__)


class WebPageExtractor:
    """Fetch a URL and convert to markdown. Output goes to a temp file.

    The framework's pipeline expects an ExtractionResult with markdown.
    For web pages, we download the page (HTML or markdown), convert to
    markdown, and return the result. The ``pdf_path`` arg is the URL.

    Configuration:
      timeout_s: HTTP request timeout
    """

    method = ExtractionMethod.MARKITDOWN

    def __init__(self, *, timeout_s: float = 30.0) -> None:
        self._timeout = timeout_s

    def is_configured(self) -> bool:
        return True

    @staticmethod
    def _normalize_url(path_or_str: Path | str) -> str:
        """Coerce the path arg back to a URL string without losing ``//``.

        ``str(Path("https://example.com/"))`` returns ``"https:/example.com"``
        because PosixPath collapses ``//``. Callers that pass a URL via
        CLI string hit this when the framework wraps it in Path first.
        Detect the scheme prefix and re-expand.
        """
        if isinstance(path_or_str, Path):
            raw = str(path_or_str)
            if not raw.startswith(("https:/", "http:/", "ftp:/")):
                return raw
            for single, double in (
                ("https:/", "https://"),
                ("http:/", "http://"),
                ("ftp:/", "ftp://"),
            ):
                if raw.startswith(single):
                    return double + raw[len(single) :]
        return str(path_or_str)

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Fetch URL ``pdf_path`` (str-like) and return markdown."""
        import httpx

        url = self._normalize_url(pdf_path)
        if not urlparse(url).scheme:
            raise ValueError(f"not a valid URL: {url}")

        try:
            resp = httpx.get(url, timeout=self._timeout, follow_redirects=True)
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"fetch failed for {url}: {exc}") from exc

        content_type = resp.headers.get("content-type", "")
        if "html" in content_type.lower():
            markdown = _html_to_markdown(resp.text, url)
        else:
            markdown = f"# {url}\n\n```\n{resp.text}\n```\n"

        title = _extract_title(resp.text) or url
        return ExtractionResult(
            markdown=f"# {title}\n\n{markdown}",
            page_count=1,
            method=self.method,
            images={},
            block_metas=[],
        )


def _html_to_markdown(html: str, url: str) -> str:
    """Best-effort HTML → markdown conversion.

    Uses trafilatura if available, else a simple regex-based fallback.
    """
    try:
        import trafilatura

        extracted = trafilatura.extract(html, include_links=False, include_images=False)
        if extracted:
            return str(extracted)
    except ImportError:
        pass

    import re

    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip() or "(empty)"


def _extract_title(html: str) -> str | None:
    """Pick the page title from an HTML document, if any."""
    import re

    match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


__all__ = ["WebPageExtractor"]
