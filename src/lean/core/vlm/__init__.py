"""Universal Vision-Language Model sidecar.

The ``VLMClient`` Protocol + OpenAI-compatible transport live here.
Domain-specific prompts and response parsers belong in the domain
package — lean-lss ships CHART_EXTRACTION_PROMPT + parse_description
in ``lean_lss.prompts``.
"""

from lean.core.vlm.client import OpenAICompatibleVLM, VLMClient, VLMError

__all__ = ["VLMClient", "VLMError", "OpenAICompatibleVLM"]
