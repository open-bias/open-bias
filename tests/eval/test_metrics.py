"""Unit tests for opensentinel.eval.metrics."""

from __future__ import annotations

from opensentinel.eval.metrics import EvalMetrics, compute_metrics
from opensentinel.eval.runner import TurnResult
from opensentinel.policy.protocols import Decision, EngineResult
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
        turn = make_turn(0, Decision.ALLOW)
        result = make_result([turn])
        metrics = compute_metrics([result])

        assert metrics.total_turns == 1
        # Both request_eval (ALLOW) and response_eval (ALLOW) counted
        assert metrics.decisions == {"allow": 2}
        assert metrics.violation_count == 0
        assert metrics.intervention_count == 0

    def test_block_increments_intervention_count(self):
        turn = make_turn(0, Decision.BLOCK)
        metrics = compute_metrics([make_result([turn])])

        assert metrics.intervention_count == 1
        assert metrics.decisions == {"allow": 1, "block": 1}

    def test_intervene_increments_intervention_count(self):
        turn = make_turn(0, Decision.INTERVENE)
        metrics = compute_metrics([make_result([turn])])

        assert metrics.intervention_count == 1
        assert metrics.decisions == {"allow": 1, "intervene": 1}

    def test_violations_counted(self):
        violations = [{"name": "v1"}, {"name": "v2"}]
        turn = make_turn(0, Decision.INTERVENE, violations=violations)
        metrics = compute_metrics([make_result([turn])])

        assert metrics.violation_count == 2

    def test_multiple_results_aggregate(self):
        r1 = make_result([make_turn(0, Decision.ALLOW), make_turn(1, Decision.BLOCK)])
        r2 = make_result([make_turn(0, Decision.INTERVENE)])
        metrics = compute_metrics([r1, r2])

        assert metrics.total_turns == 3
        # request_eval is ALLOW for all turns, so 3 extra ALLOWs
        assert metrics.decisions["allow"] == 4
        assert metrics.decisions["block"] == 1
        assert metrics.decisions["intervene"] == 1
        assert metrics.intervention_count == 2

    def test_violations_across_turns(self):
        t1 = make_turn(0, Decision.ALLOW, violations=[{"name": "v1"}])
        t2 = make_turn(1, Decision.INTERVENE, violations=[{"name": "v2"}, {"name": "v3"}])
        metrics = compute_metrics([make_result([t1, t2])])

        assert metrics.violation_count == 3

    def test_request_eval_block_counted(self):
        turn = TurnResult(
            turn_index=0,
            request_data={"messages": [], "model": "test"},
            response_data={},
            request_eval=EngineResult(
                decision=Decision.BLOCK,
                metadata={"violations": [{"name": "blocked_input"}]},
            ),
            response_eval=EngineResult(decision=Decision.ALLOW),
        )
        metrics = compute_metrics([make_result([turn])])

        assert metrics.decisions == {"block": 1, "allow": 1}
        assert metrics.intervention_count == 1
        assert metrics.violation_count == 1

    def test_multiple_scenarios_same_decision(self):
        r1 = make_result([make_turn(0, Decision.ALLOW)])
        r2 = make_result([make_turn(0, Decision.ALLOW)])
        metrics = compute_metrics([r1, r2])

        assert metrics.total_turns == 2
        assert metrics.decisions["allow"] == 4
        assert metrics.intervention_count == 0
