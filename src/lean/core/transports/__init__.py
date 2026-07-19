"""Universal transport layer — MCP, REST, CLI.

The framework ships two entry points:

- ``TransportBuilder`` — Python API; build a domain by subclassing
  ``DomainRegistration`` (used by Python adapter modules).
- ``build_from_yaml`` — YAML-driven; pass a path to a domain manifest
  and get a fully-wired ``TransportBuilder`` back.

Most users should reach for ``build_from_yaml``. ``TransportBuilder``
exists for programmatic domains that need full Python control.
"""

from lean.core.transports.builder import TransportBuilder
from lean.core.transports.registration import DomainRegistration
from lean.core.transports.yaml_loader import build_from_yaml

__all__ = ["DomainRegistration", "TransportBuilder", "build_from_yaml"]
