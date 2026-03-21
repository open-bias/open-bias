"""
Type definitions for the LLM-as-a-Judge Policy Engine.

Contains all enums and dataclasses used by the judge engine
for rubric-based evaluation, scoring, and verdict generation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EvaluationType(Enum):
    """How the judge evaluates responses."""
    POINTWISE = "pointwise"        # Score a single response
    PAIRWISE = "pairwise"          # Compare two responses
    REFERENCE = "reference"        # Score against a reference answer


class EvaluationScope(Enum):
    """What gets judged."""
    TURN = "turn"                  # Latest assistant response only
    CONVERSATION = "conversation"  # Entire conversation trajectory


class ScoreScale(Enum):
    """Scoring scale for rubric criteria."""
    BINARY = "binary"              # 0 or 1
    LIKERT_3 = "likert_3"          # 1-3
    LIKERT_5 = "likert_5"          # 1-5

    @property
    def max_score(self) -> int:
        return {
            ScoreScale.BINARY: 1,
            ScoreScale.LIKERT_3: 3,
            ScoreScale.LIKERT_5: 5,
        }[self]

    @property
    def min_score(self) -> int:
        return 0 if self == ScoreScale.BINARY else 1


class VerdictAction(Enum):
    """Action to take based on judge verdict."""
    PASS = "pass"
    INTERVENE = "intervene"
    BLOCK = "block"


@dataclass
class RubricCriterion:
    """Single scoring dimension within a rubric."""
    name: str
    description: str
    scale: ScoreScale = ScoreScale.LIKERT_5
    weight: float = 1.0
    fail_threshold: float | None = None
    score_descriptions: dict[int, str] = field(default_factory=dict)


@dataclass
class Rubric:
    """Collection of criteria for evaluation."""
    name: str
    description: str
    criteria: list[RubricCriterion]
    evaluation_type: EvaluationType = EvaluationType.POINTWISE
    scope: EvaluationScope = EvaluationScope.TURN
    pass_threshold: float = 0.6
    prompt_overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class JudgeScore:
    """Per-criterion result from a single judge."""
    criterion: str
    score: int
    max_score: int
    reasoning: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 1.0
    corrective_actions: str | None = None

    @property
    def normalized(self) -> float:
        """Normalize score to 0-1 range."""
        if self.max_score == 0:
            return 0.0
        # For binary (0-1), normalize directly
        # For likert (1-N), normalize (score-1)/(max-1)
        if self.max_score == 1:
            return float(self.score)
        return (self.score - 1) / (self.max_score - 1) if self.max_score > 1 else 0.0


@dataclass
class JudgeVerdict:
    """Full verdict from a single judge evaluation."""
    scores: list[JudgeScore]
    composite_score: float  # 0-1 normalized weighted average
    action: VerdictAction
    summary: str
    judge_model: str
    latency_ms: float = 0.0
    token_usage: int = 0
    scope: EvaluationScope = EvaluationScope.TURN
    metadata: dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": [
                {
                    "criterion": s.criterion,
                    "score": s.score,
                    "max_score": s.max_score,
                    "normalized": s.normalized,
                    "reasoning": s.reasoning,
                    "evidence": s.evidence,
                    "confidence": s.confidence,
                    "corrective_actions": s.corrective_actions,
                }
                for s in self.scores
            ],
            "composite_score": self.composite_score,
            "action": self.action.value,
            "summary": self.summary,
            "judge_model": self.judge_model,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "scope": self.scope.value,
        }


@dataclass
class JudgeSessionContext:
    """Per-session state for the judge engine."""
    session_id: str
    evaluation_history: list[JudgeVerdict] = field(default_factory=list)
    score_trend: list[float] = field(default_factory=list)
    violation_counts: dict[str, int] = field(default_factory=dict)
    turn_count: int = 0
    total_tokens_used: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    last_updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    # Intervention tracking for escalation
    intervention_count: int = 0
    last_intervention_criteria: list[str] = field(default_factory=list)
    criterion_intervention_counts: dict[str, int] = field(default_factory=dict)

    def record_verdict(self, verdict: JudgeVerdict) -> None:
        """Record a verdict and update trends."""
        self.evaluation_history.append(verdict)
        self.score_trend.append(verdict.composite_score)
        self.total_tokens_used += verdict.token_usage
        self.turn_count += 1
        self.last_updated_at = datetime.now(tz=timezone.utc)

        if verdict.action != VerdictAction.PASS:
            action_key = verdict.action.value
            self.violation_counts[action_key] = self.violation_counts.get(action_key, 0) + 1

        # Track intervention criteria for escalation
        if verdict.action != VerdictAction.PASS:
            self.intervention_count += 1
            failed = verdict.metadata.get("criterion_failures", [])
            if failed:
                self.last_intervention_criteria = list(failed)
                for criterion in failed:
                    self.criterion_intervention_counts[criterion] = (
                        self.criterion_intervention_counts.get(criterion, 0) + 1
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "total_tokens_used": self.total_tokens_used,
            "score_trend": self.score_trend,
            "violation_counts": self.violation_counts,
            "intervention_count": self.intervention_count,
            "criterion_intervention_counts": self.criterion_intervention_counts,
            "evaluation_count": len(self.evaluation_history),
            "created_at": self.created_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
        }
