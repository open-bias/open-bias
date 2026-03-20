"""Unit tests for opensentinel.eval.reporter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from opensentinel.eval.reporter import export_json, print_report
from opensentinel.eval.runner import EvalResult, TurnResult
from opensentinel.policy.protocols import Decision, EngineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turn(
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


def _make_result(
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


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------

class TestExportJson:
    def test_empty(self):
        data = export_json([])
        assert data["summary"]["total_scenarios"] == 0
        assert data["summary"]["total_turns"] == 0
        assert data["scenarios"] == []

    def test_single_allow_turn(self):
        turn = _make_turn(0, Decision.ALLOW)
        result = _make_result(turns=[turn])
        data = export_json([result])

        assert data["summary"]["total_scenarios"] == 1
        assert data["summary"]["total_turns"] == 1
        assert data["summary"]["violation_count"] == 0
        assert data["summary"]["intervention_count"] == 0

        scenario = data["scenarios"][0]
        assert scenario["scenario_path"] == "test.json"
        assert scenario["engine_type"] == "fsm"
        assert scenario["error"] is None
        assert len(scenario["turns"]) == 1

        t = scenario["turns"][0]
        assert t["turn_index"] == 0
        assert t["response_decision"] == "allow"
        assert t["intervention_needed"] is False

    def test_block_turn_marks_intervention_needed(self):
        turn = _make_turn(0, Decision.BLOCK)
        data = export_json([_make_result(turns=[turn])])
        assert data["scenarios"][0]["turns"][0]["intervention_needed"] is True

    def test_intervene_turn_marks_intervention_needed(self):
        turn = _make_turn(0, Decision.INTERVENE)
        data = export_json([_make_result(turns=[turn])])
        assert data["scenarios"][0]["turns"][0]["intervention_needed"] is True

    def test_violations_serialized_as_dicts(self):
        violations = [{"name": "v1", "severity": "high", "message": "bad"}]
        turn = _make_turn(0, Decision.INTERVENE, violations=violations)
        data = export_json([_make_result(turns=[turn])])
        t = data["scenarios"][0]["turns"][0]
        assert len(t["violations"]) == 1
        assert t["violations"][0]["name"] == "v1"

    def test_error_scenario_included(self):
        result = _make_result(error="something broke")
        data = export_json([result])
        assert data["scenarios"][0]["error"] == "something broke"

    def test_multiple_scenarios_summary(self):
        r1 = _make_result(turns=[_make_turn(0, Decision.ALLOW)])
        r2 = _make_result(turns=[_make_turn(0, Decision.BLOCK)])
        data = export_json([r1, r2])
        assert data["summary"]["total_scenarios"] == 2
        assert data["summary"]["total_turns"] == 2
        assert data["summary"]["intervention_count"] == 1


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------

_CLI_UI = "opensentinel.cli_ui"


class TestPrintReport:
    def test_no_results_runs_without_error(self):
        """print_report should not raise even with no results."""
        with patch(f"{_CLI_UI}.config_panel"), \
             patch(f"{_CLI_UI}.make_table"), \
             patch(f"{_CLI_UI}.console"), \
             patch(f"{_CLI_UI}.success") as mock_success, \
             patch(f"{_CLI_UI}.warning"):
            print_report([])
            mock_success.assert_called_once()

    def test_violations_trigger_warning(self):
        violations = [{"name": "v1"}]
        turn = _make_turn(0, Decision.INTERVENE, violations=violations)
        result = _make_result(turns=[turn])

        with patch(f"{_CLI_UI}.config_panel"), \
             patch(f"{_CLI_UI}.make_table"), \
             patch(f"{_CLI_UI}.console"), \
             patch(f"{_CLI_UI}.success") as mock_success, \
             patch(f"{_CLI_UI}.warning") as mock_warning:
            print_report([result])
            mock_warning.assert_called()
            mock_success.assert_not_called()

    def test_error_triggers_warning(self):
        result = _make_result(error="boom")

        with patch(f"{_CLI_UI}.config_panel"), \
             patch(f"{_CLI_UI}.make_table"), \
             patch(f"{_CLI_UI}.console"), \
             patch(f"{_CLI_UI}.success"), \
             patch(f"{_CLI_UI}.warning") as mock_warning:
            print_report([result])
            mock_warning.assert_called()

    def test_verbose_mode_prints_per_turn(self):
        turn = _make_turn(0, Decision.ALLOW)
        result = _make_result(turns=[turn])

        with patch(f"{_CLI_UI}.config_panel"), \
             patch(f"{_CLI_UI}.make_table"), \
             patch(f"{_CLI_UI}.console") as mock_console, \
             patch(f"{_CLI_UI}.success"), \
             patch(f"{_CLI_UI}.warning"):
            print_report([result], verbose=True)
            # console.print should be called with per-turn info
            assert mock_console.print.call_count > 0

    def test_clean_run_calls_success(self):
        turn = _make_turn(0, Decision.ALLOW)
        result = _make_result(turns=[turn])

        with patch(f"{_CLI_UI}.config_panel"), \
             patch(f"{_CLI_UI}.make_table"), \
             patch(f"{_CLI_UI}.console"), \
             patch(f"{_CLI_UI}.success") as mock_success, \
             patch(f"{_CLI_UI}.warning"):
            print_report([result])
            mock_success.assert_called_once()
