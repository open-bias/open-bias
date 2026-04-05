"""Schemas for replay-based policy improvement runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ImproveStatus = Literal["pass", "review"]


@dataclass(frozen=True)
class VariantProvenance:
    baseline_policy_path: str
    instruction: str
    variant_id: str
    generated_policy_path: str


@dataclass(frozen=True)
class PolicyVariant:
    variant_id: str
    policy_path: str
    provenance: VariantProvenance


@dataclass(frozen=True)
class ImprovementAggregate:
    labeled_cases: int
    matched_cases: int
    mismatched_cases: int
    matched_rate: float
    detection_rate: float
    failures: int


@dataclass(frozen=True)
class ImprovementTraceRun:
    trace_path: str
    dataset_name: str
    summary: dict[str, Any]
    failures: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ImprovementVariantResult:
    variant: PolicyVariant
    aggregate: ImprovementAggregate
    traces: list[ImprovementTraceRun] = field(default_factory=list)


@dataclass(frozen=True)
class ImprovementResult:
    status: ImproveStatus
    boundary: str
    baseline_policy_path: str
    instruction: str
    variants: list[ImprovementVariantResult] = field(default_factory=list)
    winner_variant_id: str | None = None
    review_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
