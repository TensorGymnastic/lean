"""File-path metadata for the code domain.

For a single-file ingest, metadata comes from the file path itself:
- author = file owner (when available)
- year = mtime year
- title = first heading OR filename
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def extract(path: Path) -> dict[str, object]:
    """Return metadata derived from a file path."""
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")

    stat = path.stat()
    year = datetime.fromtimestamp(stat.st_mtime).year if stat.st_mtime else None
    title = path.stem.replace("-", " ").replace("_", " ").title() or path.name

    authors: list[str] = []
    try:
        import pwd

        pw = pwd.getpwuid(stat.st_uid)
        authors = [pw.pw_gecos.split(",")[0] or pw.pw_name]
    except (ImportError, KeyError):
        pass

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "publisher": None,
        "keywords": [path.suffix.lstrip(".")] if path.suffix else [],
        "subject": None,
        "toc": [],
    }


__all__ = ["extract"]
