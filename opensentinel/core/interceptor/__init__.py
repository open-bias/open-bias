"""
Interceptor module for Open Sentinel.

Provides a clean abstraction for running checkers at different phases
of the LLM request lifecycle.
"""

from opensentinel.policy.protocols import Decision, EngineResult

from .adapters import PolicyEngineChecker
from .interceptor import Interceptor
from .types import (
    CheckerMode,
    CheckPhase,
    InterceptionResult,
)

__all__ = [
    # Types
    "CheckPhase",
    "CheckerMode",
    "Decision",
    "EngineResult",
    "InterceptionResult",
    # Classes
    "Interceptor",
    "PolicyEngineChecker",
]
