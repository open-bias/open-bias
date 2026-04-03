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
class JudgeRuleResult:
    """Binary result for one rule from one judge."""

    rule: str
    passed: bool
    reasoning: str = ""
    evidence: list[str] = field(default_factory=list)
    confidence: float = 1.0
    corrective_actions: str | None = None
    judge_name: str = ""
    judge_model: str = ""
    latency_ms: float = 0.0
    token_usage: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "passed": self.passed,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "corrective_actions": self.corrective_actions,
            "judge_name": self.judge_name,
            "judge_model": self.judge_model,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "metadata": self.metadata,
        }


@dataclass
class AggregatedRuleResult:
    """Aggregated binary outcome for one rule across all judges."""

    rule: str
    passed: bool
    action: VerdictAction
    judge_results: list[JudgeRuleResult]
    summary: str = ""
    aggregation_mode: str = "majority"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "passed": self.passed,
            "action": self.action.value,
            "judge_results": [result.to_dict() for result in self.judge_results],
            "summary": self.summary,
            "aggregation_mode": self.aggregation_mode,
            "metadata": self.metadata,
        }


@dataclass
class JudgeVerdict:
    """Turn-level verdict built from aggregated per-rule results."""

    aggregated_results: list[AggregatedRuleResult]
    failed_rules: list[str]
    action: VerdictAction
    summary: str
    judge_models: list[str]
    latency_ms: float = 0.0
    token_usage: int = 0
    scope: EvaluationScope = EvaluationScope.TURN
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregated_results": [
                result.to_dict() for result in self.aggregated_results
            ],
            "failed_rules": self.failed_rules,
            "action": self.action.value,
            "summary": self.summary,
            "judge_models": self.judge_models,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "scope": self.scope.value,
            "metadata": self.metadata,
        }


@dataclass
class JudgeSessionContext:
    """Per-session state for judge evaluations."""

    session_id: str
    evaluation_history: list[JudgeVerdict] = field(default_factory=list)
    failed_rules_history: list[list[str]] = field(default_factory=list)
    violation_counts: dict[str, int] = field(default_factory=dict)
    turn_count: int = 0
    total_tokens_used: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    last_updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def record_verdict(self, verdict: JudgeVerdict) -> None:
        self.evaluation_history.append(verdict)
        self.failed_rules_history.append(list(verdict.failed_rules))
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
            "failed_rules_history": self.failed_rules_history,
            "violation_counts": self.violation_counts,
            "evaluation_count": len(self.evaluation_history),
            "created_at": self.created_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
        }
