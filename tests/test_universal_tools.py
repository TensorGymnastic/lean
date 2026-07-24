"""Verify that importing universal tools makes them discoverable by the scanner."""

from __future__ import annotations


def test_all_15_universal_functions_have_lean_tool_kind() -> None:
    import lean.core.tools.universal as mod
    from lean.core.tools.universal import __all__ as universal_names

    for name in universal_names:
        fn = getattr(mod, name)
        assert hasattr(fn, "__lean_tool_kind__"), f"{name} missing __lean_tool_kind__"


def test_pdf_lss_domain_discovers_universal_tools() -> None:
    """pdf_lss.tools module exposes all 15 universal functions after import."""
    import lean.domains.pdf_lss.tools as mod
    from lean.core.tools.universal import __all__ as universal_names

    for name in universal_names:
        assert hasattr(mod, name), f"pdf_lss.tools missing universal function: {name}"


def test_code_domain_discovers_universal_tools() -> None:
    import lean.domains.code.tools as mod
    from lean.core.tools.universal import __all__ as universal_names

    for name in universal_names:
        assert hasattr(mod, name), f"code.tools missing universal function: {name}"


def test_web_domain_discovers_universal_tools() -> None:
    import lean.domains.web.tools as mod
    from lean.core.tools.universal import __all__ as universal_names

    for name in universal_names:
        assert hasattr(mod, name), f"web.tools missing universal function: {name}"
