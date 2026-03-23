"""
Interceptor module for Open Bias.

Provides a clean abstraction for running policy engines at different phases
of the LLM request lifecycle.
"""

from openbias.policy.protocols import Decision, EngineResult

from .interceptor import Interceptor
from .types import InterceptionResult

__all__ = [
    # Types
    "Decision",
    "EngineResult",
    "InterceptionResult",
    # Classes
    "Interceptor",
]
