"""Prompts and parsing for VLM chart/image extraction."""

from __future__ import annotations

import json
import re

CHART_EXTRACTION_PROMPT = """Analyze this image from a document and extract structured information.

Return a JSON object with these fields:
- "chart_type": one of "bar", "line", "pie", "scatter", "table",
  "diagram", "flowchart", "photo", "screenshot", "other"
- "title": the title or caption visible in the image, or null if none
- "description": a concise prose summary of what the image shows
  (2-4 sentences)
- "key_data_points": a list of 0-5 notable values, labels, or findings
  (e.g. "Q3 revenue: $4.2M", "Process step: Define")
- "axis_labels": {"x": "...", "y": "..."} if axes are visible, or null
- "source_text": any caption, legend, or annotation text, or null

Return ONLY the JSON object, no other text."""


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def parse_description(raw: str) -> dict[str, object]:
    """Parse VLM response into a structured dict.

    Handles three cases:
    1. Valid JSON → parsed dict
    2. JSON wrapped in markdown fence → extracted and parsed
    3. Non-JSON text → {"chart_type": "unknown", "description": raw}
    """
    stripped = raw.strip()

    fence_match = _FENCE_RE.search(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    return {"chart_type": "unknown", "description": raw.strip()}
