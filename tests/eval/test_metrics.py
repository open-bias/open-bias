"""Unit tests for openbias.eval.metrics."""

from __future__ import annotations

from openbias.eval.metrics import EvalMetrics, compute_metrics
from openbias.eval.runner import TurnResult
from openbias.policy.protocols import EvaluationResult, EvaluationStatus, ViolationRecord
from tests.eval.conftest import make_result, make_turn


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
        turn = make_turn(0, EvaluationStatus.ALLOW)
        result = make_result([turn])
        metrics = compute_metrics([result])

        assert metrics.total_turns == 1
        # Both request_eval (ALLOW) and response_eval (ALLOW) counted
        assert metrics.decisions == {"allow": 2}
        assert metrics.violation_count == 0
        assert metrics.intervention_count == 0

    def test_block_increments_intervention_count(self):
        turn = make_turn(0, EvaluationStatus.VIOLATION)
        metrics = compute_metrics([make_result([turn])])

        assert metrics.intervention_count == 1
        assert metrics.decisions == {"allow": 1, "violation": 1}

    def test_intervene_increments_intervention_count(self):
        turn = make_turn(0, EvaluationStatus.VIOLATION)
        metrics = compute_metrics([make_result([turn])])

        assert metrics.intervention_count == 1
        assert metrics.decisions == {"allow": 1, "violation": 1}

    def test_violations_counted(self):
        violations = [{"name": "v1"}, {"name": "v2"}]
        turn = make_turn(0, EvaluationStatus.VIOLATION, violations=violations)
        metrics = compute_metrics([make_result([turn])])

        assert metrics.violation_count == 2

    def test_multiple_results_aggregate(self):
        r1 = make_result([make_turn(0, EvaluationStatus.ALLOW), make_turn(1, EvaluationStatus.VIOLATION)])
        r2 = make_result([make_turn(0, EvaluationStatus.VIOLATION)])
        metrics = compute_metrics([r1, r2])

        assert metrics.total_turns == 3
        # request_eval is ALLOW for all turns, so 3 extra ALLOWs
        assert metrics.decisions["allow"] == 4
        assert metrics.decisions["violation"] == 2
        assert metrics.intervention_count == 2

    def test_violations_across_turns(self):
        t1 = make_turn(0, EvaluationStatus.ALLOW, violations=[{"name": "v1"}])
        t2 = make_turn(1, EvaluationStatus.VIOLATION, violations=[{"name": "v2"}, {"name": "v3"}])
        metrics = compute_metrics([make_result([t1, t2])])

        assert metrics.violation_count == 3

    def test_request_eval_block_counted(self):
        turn = TurnResult(
            turn_index=0,
            request_data={"messages": [], "model": "test"},
            response_data={},
            request_eval=EvaluationResult(
                status=EvaluationStatus.VIOLATION,
                violations=[ViolationRecord(rule_id="test", rule_name="test", reason="blocked_input", engine="test")],
                metadata={"violations": [{"name": "blocked_input"}]},
            ),
            response_eval=EvaluationResult(status=EvaluationStatus.ALLOW),
        )
        metrics = compute_metrics([make_result([turn])])

        assert metrics.decisions == {"violation": 1, "allow": 1}
        assert metrics.intervention_count == 1
        assert metrics.violation_count == 1

    def test_multiple_scenarios_same_decision(self):
        r1 = make_result([make_turn(0, EvaluationStatus.ALLOW)])
        r2 = make_result([make_turn(0, EvaluationStatus.ALLOW)])
        metrics = compute_metrics([r1, r2])

        assert metrics.total_turns == 2
        assert metrics.decisions["allow"] == 4
        assert metrics.intervention_count == 0
