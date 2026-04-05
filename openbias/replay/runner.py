"""Replay trace datasets directly against one initialized policy engine."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Any

from openbias.core.interceptor import Interceptor
from openbias.eval import EvalRuntimeConfig
from openbias.policy.protocols import PolicyEngine
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

    def __init__(self, *, runtime: EvalRuntimeConfig | None = None):
        self._runtime = runtime or EvalRuntimeConfig()

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
        interceptor = Interceptor(
            pre_call_evaluators=[engine] if self._runtime.request_phase_enabled else [],
            post_call_evaluators=[engine] if self._runtime.response_phase_enabled else [],
            mode=self._runtime.mode,
            default_strategy=self._runtime.strategy,
            fail_action=self._runtime.fail_action,
        )
        notes: list[str] = []
        try:
            pre_result = await interceptor.run_pre_call(
                session_id=session_id,
                request_data=request_data,
                user_request_id=f"{case.id}:pre",
            )

            action = self._action_from_interception(pre_result)
            reasons = list(self._violation_reasons(pre_result.internal_metadata))
            if action in {"block", "intervene", "shadow"} and not reasons:
                reasons.extend(
                    await self._request_violation_reasons(
                        engine=engine,
                        session_id=session_id,
                        request_data=request_data,
                    )
                )
            if action in {"block", "intervene", "shadow"}:
                notes.append("request_violation")
            if not pre_result.allowed:
                return _ObservedReplay(
                    action=action,
                    reasons=tuple(reasons),
                    supported=True,
                    notes=tuple(notes),
                )

            effective_request = copy.deepcopy(pre_result.modified_data or request_data)
            post_result = await interceptor.run_post_call(
                session_id=session_id,
                request_data=effective_request,
                response_data=response_data,
                user_request_id=f"{case.id}:post",
            )
            post_action = self._action_from_interception(post_result)
            post_reasons = list(self._violation_reasons(post_result.internal_metadata))
            if post_action in {"block", "intervene", "shadow"} and not post_reasons:
                post_reasons.extend(
                    await self._response_violation_reasons(
                        engine=engine,
                        session_id=session_id,
                        request_data=effective_request,
                        response_data=response_data,
                    )
                )
            if post_action in {"block", "intervene", "shadow"}:
                notes.append("response_violation")
            if self._runtime.mode == "async" and self._runtime.response_phase_enabled:
                async_action, async_reasons = await self._collect_async_post_action(
                    interceptor=interceptor,
                    session_id=session_id,
                )
                if async_action in {"block", "intervene", "shadow"}:
                    notes.append("async_response_violation")
                post_action = self._combine_actions(post_action, async_action)
                post_reasons = (*post_reasons, *async_reasons)

            return _ObservedReplay(
                action=self._combine_actions(action, post_action),
                reasons=tuple(dict.fromkeys((*reasons, *post_reasons))),
                supported=True,
                notes=tuple(notes),
            )
        finally:
            await engine.reset_session(session_id)

    @staticmethod
    def _action_from_interception(result: Any) -> str:
        if not result.allowed:
            return "block"
        if result.modified_data is not None or result.pending_intervention is not None:
            return "intervene"

        decisions = [
            item.get("decision")
            for item in result.internal_metadata.get("results", [])
            if isinstance(item, dict)
        ]
        if "shadow" in decisions:
            return "shadow"
        if "intervene" in decisions:
            return "intervene"
        if "block" in decisions:
            return "block"
        return "allow"

    async def _collect_async_post_action(
        self,
        *,
        interceptor: Interceptor,
        session_id: str,
    ) -> tuple[str, tuple[str, ...]]:
        # Replay needs to classify the traced request itself, so we eagerly
        # drain async post-call work instead of waiting for a future turn.
        await interceptor._await_pending_async(session_id)
        pending_results = interceptor._collect_completed_async(session_id)
        try:
            action = "allow"
            reasons: list[str] = []
            for pending in pending_results:
                mapped = interceptor._map_evaluation(pending.result, session_id)
                action = self._combine_actions(action, mapped.decision)
                reasons.extend(violation.reason for violation in pending.result.violations)
                if not pending.result.violations:
                    reasons.extend(self._violation_reasons(mapped.metadata))
            return action, tuple(dict.fromkeys(reasons))
        finally:
            interceptor._confirm_collected(session_id)

    @staticmethod
    def _combine_actions(current: str, observed: str) -> str:
        priority = {"block": 4, "intervene": 3, "shadow": 2, "allow": 1, "unknown": 0}
        return current if priority.get(current, 0) >= priority.get(observed, 0) else observed

    @staticmethod
    def _violation_reasons(metadata: dict[str, Any]) -> tuple[str, ...]:
        violations = metadata.get("violations", [])
        if not isinstance(violations, list):
            return ()
        reasons: list[str] = []
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            reason = violation.get("message") or violation.get("reason")
            if isinstance(reason, str) and reason:
                reasons.append(reason)
        return tuple(reasons)

    @staticmethod
    async def _request_violation_reasons(
        *,
        engine: PolicyEngine,
        session_id: str,
        request_data: dict[str, Any],
    ) -> tuple[str, ...]:
        inspect_session = f"{session_id}:inspect_request"
        try:
            result = await engine.evaluate_request(
                session_id=inspect_session,
                request_data=copy.deepcopy(request_data),
            )
            return tuple(violation.reason for violation in result.violations)
        finally:
            await engine.reset_session(inspect_session)

    @staticmethod
    async def _response_violation_reasons(
        *,
        engine: PolicyEngine,
        session_id: str,
        request_data: dict[str, Any],
        response_data: dict[str, Any],
    ) -> tuple[str, ...]:
        inspect_session = f"{session_id}:inspect_response"
        try:
            result = await engine.evaluate_response(
                session_id=inspect_session,
                response_data=copy.deepcopy(response_data),
                request_data=copy.deepcopy(request_data),
            )
            return tuple(violation.reason for violation in result.violations)
        finally:
            await engine.reset_session(inspect_session)

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
