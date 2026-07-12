import lean.mcp_server.prompts  # noqa: F401 — registers prompts on import
import lean.mcp_server.resources  # noqa: F401 — registers resources on import
from lean.mcp_server.tools import mcp

__all__ = ["mcp"]
