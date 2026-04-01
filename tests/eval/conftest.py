"""Shared test helpers for eval tests."""

from __future__ import annotations

from openbias.eval.runner import EvalResult, TurnResult
from openbias.policy.protocols import EvaluationResult, EvaluationStatus, ViolationRecord


def make_turn(
    idx: int,
    status: EvaluationStatus = EvaluationStatus.ALLOW,
    violations: list | None = None,
    request_status: EvaluationStatus = EvaluationStatus.ALLOW,
    request_violations: list | None = None,
) -> TurnResult:
    meta = {"violations": violations or []}
    request_meta = {"violations": request_violations or []}

    def _make_eval(st: EvaluationStatus, m: dict) -> EvaluationResult:
        if st == EvaluationStatus.VIOLATION:
            return EvaluationResult(
                status=st,
                violations=[ViolationRecord(rule_id="test", rule_name="test", reason="violation", engine="test")],
                metadata=m,
            )
        return EvaluationResult(status=st, metadata=m)

    return TurnResult(
        turn_index=idx,
        request_data={"messages": [], "model": "test"},
        response_data={},
        request_eval=_make_eval(request_status, request_meta),
        response_eval=_make_eval(status, meta),
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
