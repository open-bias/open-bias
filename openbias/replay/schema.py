"""Schemas for replaying trace datasets against a policy engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ReplayAction = Literal["allow", "intervene", "block", "shadow", "error", "unknown"]


@dataclass(frozen=True)
class ReplayCaseOutcome:
    """Observed replay result for one trace case."""

    case_id: str
    expected_action: ReplayAction
    observed_action: ReplayAction
    matched: bool | None
    supported: bool
    violation_reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplaySummary:
    """Aggregate replay metrics for one trace dataset."""

    total_cases: int
    supported_cases: int
    unsupported_cases: int
    matched_cases: int
    mismatched_cases: int
    intervention_rate: float
    block_rate: float
    pass_through_rate: float
    shadow_rate: float
    expected_action_coverage: float
    per_rule_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayRunResult:
    """Replay outputs for one dataset."""

    dataset_name: str
    outcomes: list[ReplayCaseOutcome]
    failures: list[dict[str, str]]
    summary: ReplaySummary

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
