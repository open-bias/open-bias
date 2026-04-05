from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from openbias.eval import EvalRuntimeConfig
from openbias.policy.protocols import EvaluationResult, EvaluationStatus, ViolationRecord
from openbias.replay import ReplayRunner
from openbias.traces import TraceCase, TraceDataset, TraceMetadata


@dataclass
class FakeReplayEngine:
    request_rule: str | None = None
    response_rule: str | None = None
    sessions: dict[str, list[str]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def engine_type(self) -> str:
        return "fake"

    async def initialize(self, config: dict[str, Any]) -> None:
        return None

    async def evaluate_request(
        self,
        session_id: str,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        del context
        self.sessions.setdefault(session_id, [])
        joined = " ".join(
            message.get("content", "")
            for message in request_data.get("messages", [])
            if isinstance(message, dict)
        )
        if self.request_rule and self.request_rule in joined:
            return EvaluationResult(
                status=EvaluationStatus.VIOLATION,
                violations=[ViolationRecord(reason=self.request_rule)],
            )
        return EvaluationResult(status=EvaluationStatus.ALLOW)

    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        del session_id, request_data, context
        joined = response_data.get("content", "") if isinstance(response_data, dict) else str(response_data)
        if self.response_rule and self.response_rule in joined:
            return EvaluationResult(
                status=EvaluationStatus.VIOLATION,
                violations=[ViolationRecord(reason=self.response_rule)],
            )
        return EvaluationResult(status=EvaluationStatus.ALLOW)

    async def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        return {"session_id": session_id}

    async def reset_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    async def shutdown(self) -> None:
        self.sessions.clear()


def _dataset(*, final_action: str, user: str, assistant: str) -> TraceDataset:
    return TraceDataset(
        name="trace",
        cases=[
            TraceCase(
                id=f"{final_action}-case",
                session_id="sess-1",
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                metadata=TraceMetadata(final_action=final_action),
            )
        ],
    )


@pytest.mark.asyncio
async def test_replay_runner_respects_pre_call_only_runtime():
    dataset = _dataset(final_action="block", user="unsafe request", assistant="refund complete")
    engine = FakeReplayEngine(request_rule="unsafe", response_rule="refund")
    runner = ReplayRunner(
        runtime=EvalRuntimeConfig(
            request_phase_enabled=True,
            response_phase_enabled=False,
            mode="sync",
            fail_action="block",
        )
    )

    result = await runner.run(engine, dataset)

    assert result.outcomes[0].observed_action == "block"
    assert result.outcomes[0].violation_reasons == ("unsafe",)
    assert result.outcomes[0].notes == ("request_violation",)


@pytest.mark.asyncio
async def test_replay_runner_respects_post_call_only_runtime():
    dataset = _dataset(final_action="intervene", user="unsafe request", assistant="refund complete")
    engine = FakeReplayEngine(request_rule="unsafe", response_rule="refund")
    runner = ReplayRunner(
        runtime=EvalRuntimeConfig(
            request_phase_enabled=False,
            response_phase_enabled=True,
            mode="sync",
            fail_action="intervene",
        )
    )

    result = await runner.run(engine, dataset)

    assert result.outcomes[0].observed_action == "intervene"
    assert result.outcomes[0].violation_reasons == ("refund",)
    assert result.outcomes[0].notes == ("response_violation",)


@pytest.mark.asyncio
async def test_replay_runner_sync_intervene_uses_pending_replay_signal():
    dataset = _dataset(final_action="intervene", user="hello", assistant="refund complete")
    engine = FakeReplayEngine(response_rule="refund")
    runner = ReplayRunner(
        runtime=EvalRuntimeConfig(
            request_phase_enabled=False,
            response_phase_enabled=True,
            mode="sync",
            fail_action="intervene",
        )
    )

    result = await runner.run(engine, dataset)

    assert result.outcomes[0].observed_action == "intervene"
    assert result.outcomes[0].matched is True
    assert result.summary.intervention_rate == 1.0


@pytest.mark.asyncio
async def test_replay_runner_async_intervene_drains_deferred_post_call_result():
    dataset = _dataset(final_action="intervene", user="hello", assistant="refund complete")
    engine = FakeReplayEngine(response_rule="refund")
    runner = ReplayRunner(
        runtime=EvalRuntimeConfig(
            request_phase_enabled=False,
            response_phase_enabled=True,
            mode="async",
            fail_action="intervene",
        )
    )

    result = await runner.run(engine, dataset)

    assert result.outcomes[0].observed_action == "intervene"
    assert result.outcomes[0].violation_reasons == ("refund",)
    assert result.outcomes[0].notes == ("async_response_violation",)


@pytest.mark.asyncio
async def test_replay_runner_sync_block_reports_block_action():
    dataset = _dataset(final_action="block", user="unsafe request", assistant="blocked")
    engine = FakeReplayEngine(request_rule="unsafe")
    runner = ReplayRunner(
        runtime=EvalRuntimeConfig(
            request_phase_enabled=True,
            response_phase_enabled=False,
            mode="sync",
            fail_action="block",
        )
    )

    result = await runner.run(engine, dataset)

    assert result.outcomes[0].observed_action == "block"
    assert result.summary.block_rate == 1.0


@pytest.mark.asyncio
async def test_replay_runner_preserves_shadow_mode_via_interceptor_mapping():
    dataset = _dataset(final_action="shadow", user="unsafe request", assistant="ok")
    engine = FakeReplayEngine(request_rule="unsafe")
    runner = ReplayRunner(
        runtime=EvalRuntimeConfig(
            request_phase_enabled=True,
            response_phase_enabled=False,
            mode="sync",
            fail_action="shadow",
        )
    )

    result = await runner.run(engine, dataset)

    assert result.outcomes[0].observed_action == "shadow"
    assert result.summary.shadow_rate == 1.0
