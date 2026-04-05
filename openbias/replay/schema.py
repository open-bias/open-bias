"""Schemas for replaying trace datasets against a policy engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ReplayBoundary = Literal["request", "response"]


@dataclass(frozen=True)
class ReplayCaseOutcome:
    """Observed replay result for one trace case."""

    case_id: str
    expected_detection: bool | None
    observed_detection: bool
    matched: bool | None
    boundary: ReplayBoundary
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
    detection_rate: float
    expected_detection_coverage: float
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
