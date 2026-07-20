"""Universal Vision-Language Model sidecar.

The ``VLMClient`` Protocol + OpenAI-compatible transport live here.
Domain-specific prompts and response parsers belong in the domain
package; the default chart extraction prompt + JSON parser ship in
``lean.core.vlm.prompts`` and are reused by the pdf_lss domain.
"""

from lean.core.vlm.client import OpenAICompatibleVLM, VLMClient, VLMError
from lean.core.vlm.prompts import CHART_EXTRACTION_PROMPT, parse_description

__all__ = [
    "VLMClient",
    "VLMError",
    "OpenAICompatibleVLM",
    "CHART_EXTRACTION_PROMPT",
    "parse_description",
]
