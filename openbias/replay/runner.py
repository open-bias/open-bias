"""Replay trace datasets directly against one initialized policy engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from openbias.policy.protocols import EvaluationResult, EvaluationStatus, PolicyEngine
from openbias.replay.schema import ReplayCaseOutcome, ReplayRunResult, ReplaySummary
from openbias.traces import TraceCase, TraceDataset


@dataclass(frozen=True)
class _ObservedReplay:
    action: str
    reasons: tuple[str, ...]
    supported: bool
    notes: tuple[str, ...] = ()


class ReplayRunner:
    """Run one initialized engine against one trace dataset."""

    def __init__(self, *, fail_action: str = "intervene"):
        self._fail_action = fail_action

    async def run(self, engine: PolicyEngine, dataset: TraceDataset) -> ReplayRunResult:
        outcomes: list[ReplayCaseOutcome] = []
        failures: list[dict[str, str]] = []

        for case in dataset.cases:
            try:
                observed = await self._replay_case(engine, case)
                expected_action = case.metadata.final_action
                matched = None if expected_action == "unknown" else expected_action == observed.action
                outcomes.append(
                    ReplayCaseOutcome(
                        case_id=case.id,
                        expected_action=expected_action,
                        observed_action=observed.action,
                        matched=matched,
                        supported=observed.supported,
                        violation_reasons=observed.reasons,
                        notes=observed.notes,
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

    async def _replay_case(self, engine: PolicyEngine, case: TraceCase) -> _ObservedReplay:
        request_messages = [dict(message) for message in case.messages[:-1]]
        assistant_message = dict(case.messages[-1])
        request_data = {
            "messages": request_messages,
            **({"model": case.metadata.model} if case.metadata.model else {}),
        }
        response_data: dict[str, Any] = {"content": assistant_message.get("content", "")}
        if case.tool_calls:
            response_data["tool_calls"] = list(case.tool_calls)

        session_id = f"replay:{case.session_id}:{case.id}"
        notes: list[str] = []
        try:
            request_eval = await engine.evaluate_request(
                session_id=session_id,
                request_data=request_data,
            )
            if request_eval.status == EvaluationStatus.VIOLATION:
                return _ObservedReplay(
                    action=self._action_from_eval(request_eval),
                    reasons=self._violation_reasons(request_eval),
                    supported=True,
                    notes=("request_violation",),
                )

            response_eval = await engine.evaluate_response(
                session_id=session_id,
                response_data=response_data,
                request_data=request_data,
            )
            if response_eval.status == EvaluationStatus.VIOLATION:
                notes.append("response_violation")
            return _ObservedReplay(
                action=self._action_from_eval(response_eval),
                reasons=self._violation_reasons(response_eval),
                supported=True,
                notes=tuple(notes),
            )
        finally:
            await engine.reset_session(session_id)

    def _action_from_eval(self, eval_result: EvaluationResult) -> str:
        if eval_result.status == EvaluationStatus.ALLOW:
            return "allow"
        if self._fail_action == "shadow":
            return "shadow"
        if self._fail_action == "block":
            return "block"
        return "intervene"

    @staticmethod
    def _violation_reasons(eval_result: EvaluationResult) -> tuple[str, ...]:
        return tuple(violation.reason for violation in eval_result.violations)

    def _summarize(self, outcomes: list[ReplayCaseOutcome]) -> ReplaySummary:
        supported = [outcome for outcome in outcomes if outcome.supported]
        supported_total = len(supported)
        action_counter = Counter(outcome.observed_action for outcome in supported)
        rule_counter = Counter()
        matched_cases = 0
        mismatched_cases = 0
        covered_cases = 0

        for outcome in supported:
            for reason in outcome.violation_reasons:
                rule_counter[reason] += 1
            if outcome.matched is not None:
                covered_cases += 1
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
            intervention_rate=(action_counter["intervene"] / supported_total) if supported_total else 0.0,
            block_rate=(action_counter["block"] / supported_total) if supported_total else 0.0,
            pass_through_rate=(action_counter["allow"] / supported_total) if supported_total else 0.0,
            shadow_rate=(action_counter["shadow"] / supported_total) if supported_total else 0.0,
            expected_action_coverage=(covered_cases / supported_total) if supported_total else 0.0,
            per_rule_counts=dict(sorted(rule_counter.items())),
        )
