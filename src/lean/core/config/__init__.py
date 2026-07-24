"""Typed settings base for lean. Domains subclass ``CoreSettings``.

The YAML overlay lives in
``lean.core.transports.yaml_loader._apply_settings_overrides`` — call
``build_from_yaml(path)`` to wire a domain manifest end-to-end.
"""

from lean.core.config.settings import (
    CoreSettings,
    clear_settings_cache,
    get_settings,
)

__all__ = ["CoreSettings", "get_settings", "clear_settings_cache"]
