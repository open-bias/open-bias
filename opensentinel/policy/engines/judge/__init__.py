"""
LLM-as-a-Judge Policy Engine.

Evaluates agent responses and conversation trajectories against
configurable rubrics using LLM judges.

Usage:
    The engine is auto-registered as "judge" when this package is imported.
    Use PolicyEngineRegistry.create("judge") to instantiate.
"""

# Import engine to trigger @register_engine("judge")
from opensentinel.policy.engines.judge.engine import JudgePolicyEngine

# Re-export model types
from opensentinel.policy.engines.judge.models import (
    EvaluationType,
    EvaluationScope,
    ScoreScale,
    VerdictAction,
    RubricCriterion,
    Rubric,
    JudgeScore,
    JudgeVerdict,
    EvaluationRequest,
    JudgeSessionContext,
)

# Re-export components
from opensentinel.policy.engines.judge.client import JudgeClient
from opensentinel.policy.engines.judge.evaluator import JudgeEvaluator
from opensentinel.policy.engines.judge.rubrics import RubricRegistry
from opensentinel.policy.engines.judge.bias import randomize_positions

__all__ = [
    "JudgePolicyEngine",
    "EvaluationType",
    "EvaluationScope",
    "ScoreScale",
    "VerdictAction",
    "RubricCriterion",
    "Rubric",
    "JudgeScore",
    "JudgeVerdict",
    "EvaluationRequest",
    "JudgeSessionContext",
    "JudgeClient",
    "JudgeEvaluator",
    "RubricRegistry",
    "randomize_positions",
]
