"""Replay trace datasets directly against one initialized policy engine."""

from __future__ import annotations

from collections import Counter
from typing import Any

from openbias.policy.protocols import EvaluationStatus, PolicyEngine
from openbias.replay.schema import (
    ReplayBoundary,
    ReplayCaseOutcome,
    ReplayRunResult,
    ReplaySummary,
)
from openbias.traces import TraceCase, TraceDataset


class ReplayRunner:
    """Run one initialized engine against one trace dataset."""

    def __init__(self, *, boundary: ReplayBoundary = "response"):
        self._boundary = boundary

    async def run(self, engine: PolicyEngine, dataset: TraceDataset) -> ReplayRunResult:
        outcomes: list[ReplayCaseOutcome] = []
        failures: list[dict[str, str]] = []

        for case in dataset.cases:
            try:
                expected_detection = self._expected_detection(case)
                observed_detection, reasons, notes = await self._replay_case(engine, case)
                matched = (
                    None
                    if expected_detection is None
                    else expected_detection == observed_detection
                )
                outcomes.append(
                    ReplayCaseOutcome(
                        case_id=case.id,
                        expected_detection=expected_detection,
                        observed_detection=observed_detection,
                        matched=matched,
                        boundary=self._boundary,
                        supported=True,
                        violation_reasons=reasons,
                        notes=notes,
                    )
                )
            except Exception as exc:
                failures.append({"case_id": case.id, "error": str(exc)})

        summary = self._summarize(outcomes)
        return ReplayRunResult(
            dataset_name=dataset.name,
            outcomes=outcomes,
            failures=failures,
            summary=summary,
        )

    async def _replay_case(
        self,
        engine: PolicyEngine,
        case: TraceCase,
    ) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
        request_messages = [dict(message) for message in case.messages[:-1]]
        assistant_message = dict(case.messages[-1])
        request_data = {
            "messages": request_messages,
            **({"model": case.metadata.model} if case.metadata.model else {}),
        }
        response_data: dict[str, Any] = {"content": assistant_message.get("content", "")}
        if case.tool_calls:
            response_data["tool_calls"] = list(case.tool_calls)

        session_id = f"replay:{case.session_id}:{case.id}:{self._boundary}"
        try:
            if self._boundary == "request":
                result = await engine.evaluate_request(
                    session_id=session_id,
                    request_data=request_data,
                )
            else:
                result = await engine.evaluate_response(
                    session_id=session_id,
                    response_data=response_data,
                    request_data=request_data,
                )

            reasons = tuple(
                violation.reason
                for violation in result.violations
                if isinstance(violation.reason, str) and violation.reason
            )
            observed_detection = result.status == EvaluationStatus.VIOLATION
            notes = (f"evaluated_{self._boundary}_boundary",)
            return observed_detection, reasons, notes
        finally:
            await engine.reset_session(session_id)

    @staticmethod
    def _expected_detection(case: TraceCase) -> bool | None:
        if not isinstance(case.labels, dict) or "violation" not in case.labels:
            return None
        value = case.labels["violation"]
        return value if isinstance(value, bool) else None

    def _summarize(self, outcomes: list[ReplayCaseOutcome]) -> ReplaySummary:
        supported = [outcome for outcome in outcomes if outcome.supported]
        supported_total = len(supported)
        rule_counter = Counter()
        matched_cases = 0
        mismatched_cases = 0
        expected_detection_cases = 0
        detected_cases = 0

        for outcome in supported:
            for reason in outcome.violation_reasons:
                rule_counter[reason] += 1
            if outcome.observed_detection:
                detected_cases += 1
            if outcome.matched is not None:
                expected_detection_cases += 1
                if outcome.matched:
                    matched_cases += 1
                else:
                    mismatched_cases += 1

        return ReplaySummary(
            total_cases=len(outcomes),
            supported_cases=supported_total,
            unsupported_cases=len(outcomes) - supported_total,
            matched_cases=matched_cases,
            mismatched_cases=mismatched_cases,
            detection_rate=(detected_cases / supported_total) if supported_total else 0.0,
            expected_detection_coverage=(
                expected_detection_cases / supported_total if supported_total else 0.0
            ),
            per_rule_counts=dict(rule_counter),
        )
