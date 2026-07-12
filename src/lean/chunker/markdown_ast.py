"""Walk markdown headers to build a list of (section_path, content) sections.

Uses mistune's AST renderer to parse markdown into tokens, then groups
body content under the most recent heading. Section paths join ancestor
headings with ' > '.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mistune


@dataclass
class Section:
    """A heading and its body content within the document hierarchy."""

    path: str  # "Chapter 1 > Section 1.1"
    level: int  # heading level (1-4)
    heading: str  # heading text
    content: str = ""  # body content under this heading


def build_sections(markdown: str, *, max_heading_level: int = 4) -> list[Section]:
    """Parse markdown into sections by walking H1-H{max_heading_level} headers."""
    ast_parser = mistune.create_markdown(renderer="ast", plugins=["speedup"])
    parsed = ast_parser.parse(markdown)
    tokens = parsed[0] if isinstance(parsed, tuple) else parsed

    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_section: Section | None = None
    body_parts: list[str] = []

    for token in tokens:
        if not isinstance(token, dict):
            continue
        token_type = token.get("type", "")

        if token_type == "heading":
            level = token.get("attrs", {}).get("level", 1)
            if level > max_heading_level:
                text = _extract_text(token)
                body_parts.append(text)
                continue

            # Flush previous section
            if current_section is not None:
                current_section.content = "\n\n".join(body_parts).strip()
                sections.append(current_section)
                body_parts = []

            text = _extract_text(token)

            # Pop stack to current level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))

            path = " > ".join(h for _, h in heading_stack)
            current_section = Section(path=path, level=level, heading=text)
        else:
            text = _extract_text(token)
            if text:
                if current_section is None:
                    # Content before first heading → front matter
                    current_section = Section(path="Front Matter", level=0, heading="Front Matter")
                body_parts.append(text)

    if current_section is not None:
        current_section.content = "\n\n".join(body_parts).strip()
        sections.append(current_section)

    return sections


def _extract_text(token: dict[str, Any]) -> str:
    """Extract plain text from a mistune AST token (handles nested children)."""
    if "raw" in token and isinstance(token["raw"], str):
        return token["raw"]
    if "children" in token and isinstance(token["children"], list):
        parts: list[str] = []
        for child in token["children"]:
            if isinstance(child, dict):
                parts.append(_extract_text(child))
            elif isinstance(child, str):
                parts.append(child)
        return " ".join(parts)
    return ""
