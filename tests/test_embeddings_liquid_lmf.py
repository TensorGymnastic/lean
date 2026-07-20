"""Smoke tests for the Liquid LMF embedder.

The Liquid LMF embedder depends on optional ``sentence-transformers`` and
``transformers`` packages (``uv sync --extra local-models``). Tests are
skipped if these are not importable in the test environment.
"""

from __future__ import annotations

import importlib.util

import pytest

if (
    importlib.util.find_spec("sentence_transformers") is None
    or importlib.util.find_spec("transformers") is None
):
    pytest.skip(
        "sentence-transformers not installed; run `uv sync --extra local-models`",
        allow_module_level=True,
    )
