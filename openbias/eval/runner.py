"""Minimal one-engine, one-suite eval runner."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from openbias.core.interceptor import Interceptor
from openbias.eval.schema import (
    EvalCase,
    EvalCaseOutcome,
    EvalOutcomeName,
    EvalLabels,
    EvalRunResult,
    EvalSuite,
    EvalSummary,
    EvalValidationError,
    build_turn_blueprint,
)
from openbias.policy.protocols import EvaluationResult, EvaluationStatus, PolicyEngine


@dataclass(frozen=True)
class _TurnEvaluation:
    index: int
    request_data: dict[str, Any]
    response_data: dict[str, Any] | None
    request_eval: EvaluationResult
    response_eval: EvaluationResult | None


@dataclass(frozen=True)
class _RepairVerification:
    fixed: bool
    intervention_applied: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class EvalRuntimeConfig:
    """Runtime enforcement knobs that affect offline eval behavior."""

    request_phase_enabled: bool = True
    response_phase_enabled: bool = True
    mode: Literal["sync", "async"] = "async"
    fail_action: Literal["intervene", "block", "shadow"] = "intervene"
    strategy: Literal["system_prompt_append", "user_message_inject"] = "user_message_inject"


class EvalRunner:
    """Run one initialized engine against one canonical eval suite."""

    def __init__(self, runtime: EvalRuntimeConfig | None = None):
        self._runtime = runtime or EvalRuntimeConfig()

    async def run(self, engine: PolicyEngine, suite: EvalSuite) -> EvalRunResult:
        outcomes: list[EvalCaseOutcome] = []
        failures: list[dict[str, str]] = []

        for case in suite.cases:
            try:
                direct_turns = await self._run_direct_case(engine, case)
                outcomes.append(await self._classify_case(engine, case, direct_turns))
            except Exception as exc:
                failures.append({"case_id": case.id, "error": str(exc)})

        summary = self._summarize(outcomes)
        return EvalRunResult(
            suite_name=suite.name,
            outcomes=outcomes,
            failures=failures,
            summary=summary,
        )

    async def _run_direct_case(
        self,
        engine: PolicyEngine,
        case: EvalCase,
    ) -> list[_TurnEvaluation]:
        turns = build_turn_blueprint(case.messages)
        session_id = self._session_id(case.id, "direct")
        evaluations: list[_TurnEvaluation] = []

        try:
            for turn in turns:
                request_data = self._request_data(turn.request_messages)
                if self._runtime.request_phase_enabled:
                    request_eval = await engine.evaluate_request(
                        session_id=session_id,
                        request_data=request_data,
                    )
                else:
                    request_eval = EvaluationResult(status=EvaluationStatus.ALLOW)
                response_data = self._response_data(turn.assistant_message)
                response_eval = None
                if response_data is not None and self._runtime.response_phase_enabled:
                    response_eval = await engine.evaluate_response(
                        session_id=session_id,
                        response_data=response_data,
                        request_data=request_data,
                    )
                evaluations.append(
                    _TurnEvaluation(
                        index=turn.index,
                        request_data=request_data,
                        response_data=response_data,
                        request_eval=request_eval,
                        response_eval=response_eval,
                    )
                )
        finally:
            await engine.reset_session(session_id)

        return evaluations

    async def _classify_case(
        self,
        engine: PolicyEngine,
        case: EvalCase,
        direct_turns: list[_TurnEvaluation],
    ) -> EvalCaseOutcome:
        observed_request_turns = tuple(
            turn.index for turn in direct_turns if turn.request_eval.status == EvaluationStatus.VIOLATION
        )
        observed_response_turns = tuple(
            turn.index
            for turn in direct_turns
            if turn.response_eval is not None
            and turn.response_eval.status == EvaluationStatus.VIOLATION
        )
        observed_pairs = {
            ("request", index) for index in observed_request_turns
        } | {
            ("response", index) for index in observed_response_turns
        }

        notes: list[str] = []
        false_positive = bool(observed_pairs) and not case.labels.violation
        detected = self._detected_expected_violation(case.labels, observed_pairs)
        fixed: bool | None = None
        if case.labels.repair_expected is not None:
            repair = await self._verify_repair(engine, case)
            fixed = repair.fixed
            notes.extend(repair.notes)

        outcome: EvalOutcomeName
        if not case.labels.violation:
            outcome = "false_positive" if false_positive else "correct_non_violation"
        elif not detected:
            outcome = "missed_violation"
        elif case.labels.repair_expected is None:
            outcome = "detected_violation"
        elif fixed:
            outcome = "detected_and_fixed"
        else:
            outcome = "detected_not_fixed"

        expected_outcome = self._expected_outcome(case.labels)
        return EvalCaseOutcome(
            case_id=case.id,
            outcome=outcome,
            expected_outcome=expected_outcome,
            passed=outcome == expected_outcome,
            detected=detected,
            false_positive=false_positive,
            fixed=fixed,
            detection_turns=tuple(index for _, index in sorted(observed_pairs, key=lambda item: (item[1], item[0]))),
            detection_boundaries=tuple(
                boundary for boundary, _ in sorted(observed_pairs, key=lambda item: (item[1], item[0]))
            ),
            notes=tuple(notes),
        )

    async def _verify_repair(self, engine: PolicyEngine, case: EvalCase) -> _RepairVerification:
        labels = case.labels
        if labels.repair_verified_at_turn is None:
            raise EvalValidationError(
                f"Case {case.id!r} is missing repair_verified_at_turn."
            )

        turns = build_turn_blueprint(case.messages)
        interceptor = Interceptor(
            pre_call_evaluators=[engine] if self._runtime.request_phase_enabled else [],
            post_call_evaluators=[engine] if self._runtime.response_phase_enabled else [],
            mode=self._runtime.mode,
            default_strategy=self._runtime.strategy,
            fail_action=self._runtime.fail_action,
        )
        session_id = self._session_id(case.id, "repair")
        notes: list[str] = []
        intervention_applied = False
        compliance_status = EvaluationStatus.ALLOW
        pending_repair_request: dict[str, Any] | None = None

        try:
            for turn in turns:
                request_data = self._request_data(turn.request_messages)
                pre_result = await interceptor.run_pre_call(
                    session_id=session_id,
                    request_data=request_data,
                    user_request_id=f"{case.id}:turn:{turn.index}:pre",
                )
                effective_request = copy.deepcopy(pre_result.modified_data or request_data)
                if pre_result.modified_data is not None:
                    intervention_applied = True
                    notes.append(f"intervention_applied_at_turn={turn.index}")

                if (
                    turn.index == labels.repair_verified_at_turn
                    and pre_result.modified_data is None
                    and pending_repair_request is not None
                ):
                    effective_request = copy.deepcopy(pending_repair_request)
                    notes.append(f"sync_replay_context_used_at_turn={turn.index}")

                if turn.index == labels.repair_verified_at_turn:
                    response_data = self._response_data(turn.assistant_message)
                    if response_data is None:
                        raise EvalValidationError(
                            f"Case {case.id!r} repair turn {turn.index} is missing an assistant response."
                        )
                    repair_eval = await engine.evaluate_response(
                        session_id=session_id,
                        response_data=response_data,
                        request_data=effective_request,
                    )
                    compliance_status = repair_eval.status
                    break

                response_data = self._response_data(turn.assistant_message)
                if response_data is not None:
                    post_result = await interceptor.run_post_call(
                        session_id=session_id,
                        request_data=effective_request,
                        response_data=response_data,
                        user_request_id=f"{case.id}:turn:{turn.index}:post",
                    )
                    if not post_result.allowed:
                        notes.append(f"response_blocked_at_turn={turn.index}")
                    if post_result.pending_intervention is not None:
                        intervention_applied = True
                        notes.append(f"sync_intervention_queued_at_turn={turn.index}")
                        pending_request = post_result.pending_intervention.get("request_data")
                        if isinstance(pending_request, dict):
                            pending_repair_request = copy.deepcopy(pending_request)
        finally:
            await engine.reset_session(session_id)
            await interceptor.shutdown()

        if not intervention_applied:
            notes.append("repair_turn_reached_without_intervention")
        if compliance_status != EvaluationStatus.ALLOW:
            notes.append("repair_turn_response_still_violates")

        return _RepairVerification(
            fixed=intervention_applied and compliance_status == EvaluationStatus.ALLOW,
            intervention_applied=intervention_applied,
            notes=tuple(notes),
        )

    def _summarize(self, outcomes: list[EvalCaseOutcome]) -> EvalSummary:
        true_positive = sum(1 for outcome in outcomes if outcome.detected and outcome.expected_outcome != "correct_non_violation")
        false_negative = sum(1 for outcome in outcomes if outcome.expected_outcome != "correct_non_violation" and not outcome.detected)
        false_positive = sum(1 for outcome in outcomes if outcome.false_positive)
        true_negative = sum(1 for outcome in outcomes if outcome.expected_outcome == "correct_non_violation" and not outcome.false_positive)

        repair_outcomes = [outcome for outcome in outcomes if outcome.fixed is not None]
        fix_success_count = sum(1 for outcome in repair_outcomes if outcome.fixed)
        fix_failure_count = sum(1 for outcome in repair_outcomes if not outcome.fixed)

        positive_total = true_positive + false_negative
        negative_total = true_negative + false_positive
        repair_total = fix_success_count + fix_failure_count

        return EvalSummary(
            true_positive=true_positive,
            false_negative=false_negative,
            false_positive=false_positive,
            true_negative=true_negative,
            detection_recall=(true_positive / positive_total) if positive_total else 0.0,
            false_positive_rate=(false_positive / negative_total) if negative_total else 0.0,
            fix_success_count=fix_success_count,
            fix_failure_count=fix_failure_count,
            fix_rate=(fix_success_count / repair_total) if repair_total else 0.0,
            exact_case_pass_rate=(
                sum(1 for outcome in outcomes if outcome.passed) / len(outcomes)
                if outcomes
                else 0.0
            ),
        )

    @staticmethod
    def _detected_expected_violation(
        labels: EvalLabels,
        observed_pairs: set[tuple[str, int]],
    ) -> bool:
        if not labels.violation or labels.detect_at_turn is None:
            return False
        if labels.detection_scope == "either":
            return (
                ("request", labels.detect_at_turn) in observed_pairs
                or ("response", labels.detect_at_turn) in observed_pairs
            )
        return (labels.detection_scope, labels.detect_at_turn) in observed_pairs

    @staticmethod
    def _expected_outcome(labels: EvalLabels) -> EvalOutcomeName:
        if not labels.violation:
            return "correct_non_violation"
        if labels.repair_expected is None:
            return "detected_violation"
        if labels.repair_expected:
            return "detected_and_fixed"
        return "detected_not_fixed"

    @staticmethod
    def _request_data(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"messages": copy.deepcopy(messages), "model": "eval-harness"}

    @staticmethod
    def _response_data(message: dict[str, Any] | None) -> dict[str, Any] | None:
        if message is None:
            return None
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": message.get("content"),
                        "tool_calls": copy.deepcopy(message.get("tool_calls")),
                    }
                }
            ],
            "model": "eval-harness",
        }

    @staticmethod
    def _session_id(case_id: str, phase: str) -> str:
        return f"eval-{phase}-{case_id}-{uuid4().hex[:8]}"
