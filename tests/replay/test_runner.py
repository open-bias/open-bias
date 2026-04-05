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
        joined = (
            response_data.get("content", "")
            if isinstance(response_data, dict)
            else str(response_data)
        )
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


def _dataset(
    *,
    user: str,
    assistant: str,
    labels: dict[str, Any] | None = None,
) -> TraceDataset:
    return TraceDataset(
        name="trace",
        cases=[
            TraceCase(
                id="trace-case",
                session_id="sess-1",
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                metadata=TraceMetadata(final_action="unknown"),
                labels=labels,
            )
        ],
    )


@pytest.mark.asyncio
async def test_replay_runner_defaults_to_response_boundary_detection():
    dataset = _dataset(user="hello", assistant="refund complete")
    engine = FakeReplayEngine(response_rule="refund")

    result = await ReplayRunner().run(engine, dataset)

    assert result.outcomes[0].boundary == "response"
    assert result.outcomes[0].observed_detection is True
    assert result.outcomes[0].violation_reasons == ("refund",)
    assert result.summary.detection_rate == 1.0


@pytest.mark.asyncio
async def test_replay_runner_request_boundary_detects_request_violation():
    dataset = _dataset(user="unsafe request", assistant="safe response")
    engine = FakeReplayEngine(request_rule="unsafe")

    result = await ReplayRunner(boundary="request").run(engine, dataset)

    assert result.outcomes[0].boundary == "request"
    assert result.outcomes[0].observed_detection is True
    assert result.outcomes[0].violation_reasons == ("unsafe",)


@pytest.mark.asyncio
async def test_request_boundary_ignores_response_only_violations():
    dataset = _dataset(user="hello", assistant="refund complete")
    engine = FakeReplayEngine(response_rule="refund")

    result = await ReplayRunner(boundary="request").run(engine, dataset)

    assert result.outcomes[0].observed_detection is False
    assert result.outcomes[0].violation_reasons == ()
    assert result.summary.detection_rate == 0.0


@pytest.mark.asyncio
async def test_response_boundary_ignores_request_only_violations():
    dataset = _dataset(user="unsafe request", assistant="all good")
    engine = FakeReplayEngine(request_rule="unsafe")

    result = await ReplayRunner(boundary="response").run(engine, dataset)

    assert result.outcomes[0].observed_detection is False
    assert result.outcomes[0].violation_reasons == ()
    assert result.summary.detection_rate == 0.0


@pytest.mark.asyncio
async def test_trace_labels_drive_expected_detection_and_matching():
    dataset = _dataset(
        user="unsafe request",
        assistant="all good",
        labels={"violation": True},
    )
    engine = FakeReplayEngine(request_rule="unsafe")

    result = await ReplayRunner(boundary="request").run(engine, dataset)

    assert result.outcomes[0].expected_detection is True
    assert result.outcomes[0].matched is True
    assert result.summary.matched_cases == 1
    assert result.summary.expected_detection_coverage == 1.0


@pytest.mark.asyncio
async def test_missing_trace_labels_leave_expected_detection_unknown():
    dataset = _dataset(user="hello", assistant="all good")
    engine = FakeReplayEngine()

    result = await ReplayRunner(boundary="response").run(engine, dataset)

    assert result.outcomes[0].expected_detection is None
    assert result.outcomes[0].matched is None
    assert result.summary.expected_detection_coverage == 0.0


@pytest.mark.asyncio
async def test_replay_summary_reports_detection_metrics():
    dataset = TraceDataset(
        name="trace",
        cases=[
            TraceCase(
                id="detects",
                session_id="sess-1",
                messages=[
                    {"role": "user", "content": "unsafe request"},
                    {"role": "assistant", "content": "ok"},
                ],
                metadata=TraceMetadata(final_action="unknown"),
                labels={"violation": True},
            ),
            TraceCase(
                id="misses",
                session_id="sess-2",
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "ok"},
                ],
                metadata=TraceMetadata(final_action="unknown"),
                labels={"violation": False},
            ),
            TraceCase(
                id="unlabeled",
                session_id="sess-3",
                messages=[
                    {"role": "user", "content": "unsafe request"},
                    {"role": "assistant", "content": "ok"},
                ],
                metadata=TraceMetadata(final_action="unknown"),
            ),
        ],
    )
    engine = FakeReplayEngine(request_rule="unsafe")

    result = await ReplayRunner(boundary="request").run(engine, dataset)

    assert result.summary.total_cases == 3
    assert result.summary.supported_cases == 3
    assert result.summary.matched_cases == 2
    assert result.summary.mismatched_cases == 0
    assert result.summary.detection_rate == pytest.approx(2 / 3)
    assert result.summary.expected_detection_coverage == pytest.approx(2 / 3)
