"""URL metadata for the web domain."""

from __future__ import annotations

from urllib.parse import urlparse


def extract(url: str) -> dict[str, object]:
    """Return metadata inferred from a URL.

    The URL is passed as a Path-like object; we coerce to str.
    """
    url_str = str(url)
    parsed = urlparse(url_str)
    domain = parsed.netloc or ""
    path_parts = [p for p in parsed.path.split("/") if p]
    title = path_parts[-1].replace("-", " ").replace("_", " ").title() if path_parts else domain
    return {
        "title": title,
        "authors": [domain] if domain else [],
        "year": None,
        "publisher": domain or None,
        "keywords": [],
        "subject": None,
        "toc": [],
    }


__all__ = ["extract"]
