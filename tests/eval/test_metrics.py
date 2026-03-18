"""Unit tests for opensentinel.eval.metrics."""

from __future__ import annotations

import pytest

from opensentinel.eval.metrics import EvalMetrics, compute_metrics
from opensentinel.eval.runner import EvalResult, TurnResult
from opensentinel.policy.protocols import Decision, EngineResult


def _allow_result(**kw) -> EngineResult:
    return EngineResult(decision=Decision.ALLOW, **kw)


def _block_result(**kw) -> EngineResult:
    return EngineResult(decision=Decision.BLOCK, **kw)


def _intervene_result(**kw) -> EngineResult:
    return EngineResult(decision=Decision.INTERVENE, **kw)


def _make_turn(
    idx: int,
    response_decision: Decision = Decision.ALLOW,
    violations: list | None = None,
) -> TurnResult:
    meta = {"violations": violations or []}
    return TurnResult(
        turn_index=idx,
        request_data={"messages": [], "model": "test"},
        response_data={},
        request_eval=_allow_result(),
        response_eval=EngineResult(decision=response_decision, metadata=meta),
    )


def _make_result(
    turns: list[TurnResult],
    engine_type: str = "fsm",
    error: str | None = None,
) -> EvalResult:
    return EvalResult(
        scenario_path="test.json",
        session_id="sess-1",
        turns=turns,
        engine_type=engine_type,
        error=error,
    )


class TestEvalMetrics:
    def test_defaults(self):
        m = EvalMetrics()
        assert m.total_turns == 0
        assert m.decisions == {}
        assert m.violation_count == 0
        assert m.intervention_count == 0


class TestComputeMetrics:
    def test_empty_results(self):
        metrics = compute_metrics([])
        assert metrics.total_turns == 0
        assert metrics.decisions == {}
        assert metrics.violation_count == 0
        assert metrics.intervention_count == 0

    def test_single_allow_turn(self):
        turn = _make_turn(0, Decision.ALLOW)
        result = _make_result([turn])
        metrics = compute_metrics([result])

        assert metrics.total_turns == 1
        assert metrics.decisions == {"allow": 1}
        assert metrics.violation_count == 0
        assert metrics.intervention_count == 0

    def test_block_increments_intervention_count(self):
        turn = _make_turn(0, Decision.BLOCK)
        metrics = compute_metrics([_make_result([turn])])

        assert metrics.intervention_count == 1
        assert metrics.decisions == {"block": 1}

    def test_intervene_increments_intervention_count(self):
        turn = _make_turn(0, Decision.INTERVENE)
        metrics = compute_metrics([_make_result([turn])])

        assert metrics.intervention_count == 1
        assert metrics.decisions == {"intervene": 1}

    def test_violations_counted(self):
        violations = [{"name": "v1"}, {"name": "v2"}]
        turn = _make_turn(0, Decision.INTERVENE, violations=violations)
        metrics = compute_metrics([_make_result([turn])])

        assert metrics.violation_count == 2

    def test_multiple_results_aggregate(self):
        r1 = _make_result([_make_turn(0, Decision.ALLOW), _make_turn(1, Decision.BLOCK)])
        r2 = _make_result([_make_turn(0, Decision.INTERVENE)])
        metrics = compute_metrics([r1, r2])

        assert metrics.total_turns == 3
        assert metrics.decisions["allow"] == 1
        assert metrics.decisions["block"] == 1
        assert metrics.decisions["intervene"] == 1
        assert metrics.intervention_count == 2

    def test_violations_across_turns(self):
        t1 = _make_turn(0, Decision.ALLOW, violations=[{"name": "v1"}])
        t2 = _make_turn(1, Decision.INTERVENE, violations=[{"name": "v2"}, {"name": "v3"}])
        metrics = compute_metrics([_make_result([t1, t2])])

        assert metrics.violation_count == 3

    def test_multiple_scenarios_same_decision(self):
        r1 = _make_result([_make_turn(0, Decision.ALLOW)])
        r2 = _make_result([_make_turn(0, Decision.ALLOW)])
        metrics = compute_metrics([r1, r2])

        assert metrics.total_turns == 2
        assert metrics.decisions["allow"] == 2
        assert metrics.intervention_count == 0
