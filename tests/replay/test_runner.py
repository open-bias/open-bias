from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

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

    async def evaluate_request(self, session_id: str, request_data: dict[str, Any], context: dict[str, Any] | None = None) -> EvaluationResult:
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

    async def evaluate_response(self, session_id: str, response_data: Any, request_data: dict[str, Any], context: dict[str, Any] | None = None) -> EvaluationResult:
        del request_data, context
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


@pytest.mark.asyncio
async def test_replay_runner_tracks_allow_and_intervene_actions():
    dataset = TraceDataset(
        name="smoke",
        cases=[
            TraceCase(
                id="allow-case",
                session_id="sess-1",
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
                metadata=TraceMetadata(final_action="allow"),
            ),
            TraceCase(
                id="intervene-case",
                session_id="sess-2",
                messages=[
                    {"role": "user", "content": "refund me"},
                    {"role": "assistant", "content": "refund complete"},
                ],
                metadata=TraceMetadata(final_action="intervene"),
            ),
        ],
    )
    engine = FakeReplayEngine(response_rule="refund")
    result = await ReplayRunner(fail_action="intervene").run(engine, dataset)

    assert result.summary.total_cases == 2
    assert result.summary.intervention_rate == 0.5
    assert result.summary.pass_through_rate == 0.5
    assert result.summary.matched_cases == 2
    assert result.summary.per_rule_counts == {"refund": 1}


@pytest.mark.asyncio
async def test_replay_runner_maps_request_violations_to_block():
    dataset = TraceDataset(
        name="blocks",
        cases=[
            TraceCase(
                id="request-block",
                session_id="sess-1",
                messages=[
                    {"role": "user", "content": "unsafe request"},
                    {"role": "assistant", "content": "blocked"},
                ],
                metadata=TraceMetadata(final_action="block"),
            )
        ],
    )
    engine = FakeReplayEngine(request_rule="unsafe")
    result = await ReplayRunner(fail_action="block").run(engine, dataset)

    assert result.outcomes[0].observed_action == "block"
    assert result.summary.block_rate == 1.0
