"""Intervention payload and aggregation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


InterventionMode = Literal["sync", "async"]


@dataclass
class AggregatedInterventionInput:
    """Turn-level aggregated violation view consumed by the builder."""

    mode: InterventionMode
    source_violations: list[dict[str, Any]] = field(default_factory=list)
    merged_violation_summary: str = ""
    evaluators: list[str] = field(default_factory=list)


@dataclass
class InterventionPayload:
    """Deterministic instruction payload produced after aggregation."""

    sync_repair_instruction: str
    async_guidance: str | None
    cleanup_rules: list[str]
    source_violations: list[dict[str, Any]]
    merged_violation_summary: str
    mode: InterventionMode
