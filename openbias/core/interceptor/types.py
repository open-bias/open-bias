"""
Core types for the interceptor system.

Defines the dataclasses used throughout the interceptor module.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InterceptionResult:
    """Result of running interceptor pre_call or post_call."""

    allowed: bool
    modified_data: dict[str, Any] | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
