"""Parse the pdf_lss VLM response — JSON with markdown-fence handling."""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def parse(raw: str) -> dict[str, object]:
    """Parse the VLM response into a dict.

    Handles three cases:
      1. Valid JSON -> parsed dict
      2. JSON wrapped in markdown fence -> extracted and parsed
      3. Non-JSON text -> {"chart_type": "unknown", "description": raw}
    """
    stripped = (raw or "").strip()
    fence_match = _FENCE_RE.search(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return {"chart_type": "unknown", "description": (raw or "").strip()}


__all__ = ["parse"]
