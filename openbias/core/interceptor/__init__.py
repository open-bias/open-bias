"""
Interceptor module for Open Bias.

Provides a clean abstraction for running checkers at different phases
of the LLM request lifecycle.
"""

from openbias.policy.protocols import Decision, EngineResult

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
