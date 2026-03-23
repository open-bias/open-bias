"""
Core types for the interceptor system.

Defines the enums and dataclasses used throughout the interceptor module.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CheckPhase(Enum):
    """When a checker runs in the request lifecycle."""

    PRE_CALL = "pre_call"    # Before LLM call
    POST_CALL = "post_call"  # After LLM call


class CheckerMode(Enum):
    """How a checker executes."""

    SYNC = "sync"    # Blocking, must complete before proceeding
    ASYNC = "async"  # Background, results applied on next request


@dataclass
class InterceptionResult:
    """Result of running interceptor pre_call or post_call."""

    allowed: bool
    modified_data: dict[str, Any] | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
