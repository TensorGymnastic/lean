"""MCP prompts for Lean Six Sigma Q&A with corpus retrieval."""

from __future__ import annotations

from lean.mcp_server.tools import mcp


@mcp.prompt
def lean_qa(topic: str) -> str:
    """Answer a question about a Lean Six Sigma topic with corpus citations."""
    return (
        f"Use the `search` tool to find chunks about '{topic}' (k=5). "
        f"Then answer the user's question about {topic}, citing "
        f"section_path for each source. If the corpus lacks relevant "
        f"content, say so explicitly."
    )


@mcp.prompt
def lean_glossary(term: str) -> str:
    """Produce a glossary entry for a Lean Six Sigma term using the corpus."""
    return (
        f"Use the `search` tool with query='{term} definition' (k=3). "
        f"Then write a concise glossary entry for '{term}' citing each "
        f"source by section_path."
    )


@mcp.prompt
def lean_compare_concepts(a: str, b: str) -> str:
    """Compare two Lean Six Sigma concepts using corpus evidence."""
    return (
        f"Use the `search` tool twice: once for '{a}' and once for '{b}' "
        f"(k=3 each). Then produce a comparison table with rows: "
        f"Definition, Purpose, When to use, Key steps. Cite section_path."
    )
