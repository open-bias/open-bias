"""Schemas for baseline-vs-candidate policy comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ComparisonStatus = Literal["pass", "review", "fail"]


@dataclass(frozen=True)
class SuiteComparison:
    name: str
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    delta_exact_case_pass_rate: float
    delta_false_positive_rate: float


@dataclass(frozen=True)
class TraceComparison:
    name: str
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    delta_matched_rate: float
    delta_intervention_rate: float
    delta_block_rate: float
    delta_pass_through_rate: float


@dataclass(frozen=True)
class ComparisonGate:
    status: ComparisonStatus
    reason: str


@dataclass(frozen=True)
class PolicyComparisonResult:
    status: ComparisonStatus
    baseline_policy_path: str
    candidate_policy_path: str
    candidate_details: dict[str, Any] = field(default_factory=dict)
    suites: list[SuiteComparison] = field(default_factory=list)
    traces: list[TraceComparison] = field(default_factory=list)
    gates: list[ComparisonGate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
