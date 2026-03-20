"""Shared test helpers for eval tests."""

from __future__ import annotations

from opensentinel.eval.runner import EvalResult, TurnResult
from opensentinel.policy.protocols import Decision, EngineResult


def make_turn(
    idx: int,
    decision: Decision = Decision.ALLOW,
    violations: list | None = None,
) -> TurnResult:
    meta = {"violations": violations or []}
    return TurnResult(
        turn_index=idx,
        request_data={"messages": [], "model": "test"},
        response_data={},
        request_eval=EngineResult(decision=Decision.ALLOW),
        response_eval=EngineResult(decision=decision, metadata=meta),
    )


def make_result(
    turns: list[TurnResult] | None = None,
    scenario_path: str = "test.json",
    engine_type: str = "fsm",
    error: str | None = None,
) -> EvalResult:
    return EvalResult(
        scenario_path=scenario_path,
        session_id="sess-1",
        turns=turns or [],
        engine_type=engine_type,
        error=error,
    )
