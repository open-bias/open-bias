"""
LLM-as-a-Judge Policy Engine.

Evaluates agent responses against configurable rules using LLM judges.

Usage:
    The engine is auto-registered as "judge" when this package is imported.
    Use PolicyEngineRegistry.create("judge") to instantiate.
"""

# Import engine to trigger @register_engine("judge")
from openbias.policy.engines.judge.engine import JudgePolicyEngine

# Re-export model types
from openbias.policy.engines.judge.models import (
    EvaluationScope,
    VerdictAction,
    JudgeScore,
    JudgeVerdict,
    JudgeSessionContext,
)

# Re-export components
from openbias.policy.engines.judge.client import JudgeClient
from openbias.policy.engines.judge.evaluator import JudgeEvaluator

__all__ = [
    "JudgePolicyEngine",
    "EvaluationScope",
    "VerdictAction",
    "JudgeScore",
    "JudgeVerdict",
    "JudgeSessionContext",
    "JudgeClient",
    "JudgeEvaluator",
]
