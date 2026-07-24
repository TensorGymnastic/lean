"""Universal Vision-Language Model sidecar.

The OpenAI-compatible transport lives here. Domain-specific prompts and
response parsers belong in the domain package; the default chart
extraction prompt + JSON parser ship in ``lean.core.vlm.prompts`` and
are reused by the pdf_lss domain.
"""

from lean.core.vlm.client import OpenAICompatibleVLM, VLMError
from lean.core.vlm.prompts import CHART_EXTRACTION_PROMPT, parse_description

__all__ = [
    "VLMError",
    "OpenAICompatibleVLM",
    "CHART_EXTRACTION_PROMPT",
    "parse_description",
]
