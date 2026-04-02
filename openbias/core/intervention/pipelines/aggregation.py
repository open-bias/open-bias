"""Violation fan-in stage for turn-level intervention decisions."""

from __future__ import annotations

from typing import Any

from .types import AggregatedInterventionInput, InterventionMode


class ViolationAggregationStage:
    """Merge per-evaluator intervention records into one turn-level summary."""

    def aggregate(
        self, *, records: list[dict[str, Any]], mode: InterventionMode
    ) -> AggregatedInterventionInput:
        deduped: dict[str, dict[str, Any]] = {}
        evaluators: list[str] = []

        for record in records:
            evaluator = str(record.get("evaluator") or "unknown")
            if evaluator not in evaluators:
                evaluators.append(evaluator)

            message = str(record.get("message") or "").strip()
            if not message:
                continue

            metadata = record.get("metadata") or {}
            violations = metadata.get("violations") or []
            violation_meta = violations[0] if violations else {}
            key = message.casefold()

            if key not in deduped:
                deduped[key] = {
                    "message": message,
                    "severity": violation_meta.get("severity", "error"),
                    "scope": violation_meta.get("scope", "turn"),
                    "engine": violation_meta.get("engine", evaluator),
                    "confidence": violation_meta.get("confidence"),
                    "sources": [evaluator],
                }
            elif evaluator not in deduped[key]["sources"]:
                deduped[key]["sources"].append(evaluator)

        source_violations = list(deduped.values())
        merged_violation_summary = "; ".join(
            violation["message"] for violation in source_violations
        )

        return AggregatedInterventionInput(
            mode=mode,
            source_violations=source_violations,
            merged_violation_summary=merged_violation_summary,
            evaluators=evaluators,
        )
