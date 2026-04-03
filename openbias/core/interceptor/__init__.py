"""
Interceptor module for Open Bias.

Provides a clean abstraction for running policy engines at different phases
of the LLM request lifecycle.
"""

from openbias.policy.protocols import (
    Decision,
    EvaluationResult,
    EvaluationStatus,
    ViolationRecord,
)

from .interceptor import Interceptor, SYNC_POST_REPLAY_KIND
from .types import InterceptionResult

__all__ = [
    # Types
    "Decision",
    "EvaluationResult",
    "EvaluationStatus",
    "ViolationRecord",
    "InterceptionResult",
    # Classes
    "Interceptor",
    "SYNC_POST_REPLAY_KIND",
]
