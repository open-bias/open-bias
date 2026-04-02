"""Type definitions for the simplified judge engine."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EvaluationScope(Enum):
    """Judge engine currently evaluates only at turn scope."""

    TURN = "turn"


class VerdictAction(Enum):
    """Action to take based on judge verdict."""

    PASS = "pass"
    INTERVENE = "intervene"
    BLOCK = "block"


@dataclass
class JudgeScore:
    """Per-rule binary result from a single judge call."""

    criterion: str
    score: int
    max_score: int = 1
    reasoning: str = ""
    evidence: list[str] = field(default_factory=list)
    confidence: float = 1.0
    corrective_actions: str | None = None

    @property
    def normalized(self) -> float:
        return float(self.score) if self.max_score else 0.0


@dataclass
class JudgeVerdict:
    """Turn-level verdict for a set of binary rules."""

    scores: list[JudgeScore]
    composite_score: float
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
    """Per-session state for judge evaluations."""

    session_id: str
    evaluation_history: list[JudgeVerdict] = field(default_factory=list)
    score_trend: list[float] = field(default_factory=list)
    violation_counts: dict[str, int] = field(default_factory=dict)
    turn_count: int = 0
    total_tokens_used: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    last_updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def record_verdict(self, verdict: JudgeVerdict) -> None:
        self.evaluation_history.append(verdict)
        self.score_trend.append(verdict.composite_score)
        self.total_tokens_used += verdict.token_usage
        self.last_updated_at = datetime.now(tz=timezone.utc)
        if verdict.action != VerdictAction.PASS:
            action_key = verdict.action.value
            self.violation_counts[action_key] = self.violation_counts.get(action_key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "total_tokens_used": self.total_tokens_used,
            "score_trend": self.score_trend,
            "violation_counts": self.violation_counts,
            "evaluation_count": len(self.evaluation_history),
            "created_at": self.created_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
        }
