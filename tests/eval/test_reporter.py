"""Unit tests for opensentinel.eval.reporter."""

from __future__ import annotations

from unittest.mock import patch

from opensentinel.eval.reporter import export_json, print_report
from opensentinel.policy.protocols import Decision
from tests.eval.conftest import make_result, make_turn

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
        turn = make_turn(0, Decision.ALLOW)
        result = make_result(turns=[turn])
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
        turn = make_turn(0, Decision.BLOCK)
        data = export_json([make_result(turns=[turn])])
        assert data["scenarios"][0]["turns"][0]["intervention_needed"] is True

    def test_intervene_turn_marks_intervention_needed(self):
        turn = make_turn(0, Decision.INTERVENE)
        data = export_json([make_result(turns=[turn])])
        assert data["scenarios"][0]["turns"][0]["intervention_needed"] is True

    def test_violations_serialized_as_dicts(self):
        violations = [{"name": "v1", "severity": "high", "message": "bad"}]
        turn = make_turn(0, Decision.INTERVENE, violations=violations)
        data = export_json([make_result(turns=[turn])])
        t = data["scenarios"][0]["turns"][0]
        assert len(t["violations"]) == 1
        assert t["violations"][0]["name"] == "v1"

    def test_error_scenario_included(self):
        result = make_result(error="something broke")
        data = export_json([result])
        assert data["scenarios"][0]["error"] == "something broke"

    def test_multiple_scenarios_summary(self):
        r1 = make_result(turns=[make_turn(0, Decision.ALLOW)])
        r2 = make_result(turns=[make_turn(0, Decision.BLOCK)])
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
        turn = make_turn(0, Decision.INTERVENE, violations=violations)
        result = make_result(turns=[turn])

        with patch(f"{_CLI_UI}.config_panel"), \
             patch(f"{_CLI_UI}.make_table"), \
             patch(f"{_CLI_UI}.console"), \
             patch(f"{_CLI_UI}.success") as mock_success, \
             patch(f"{_CLI_UI}.warning") as mock_warning:
            print_report([result])
            mock_warning.assert_called()
            mock_success.assert_not_called()

    def test_error_triggers_warning(self):
        result = make_result(error="boom")

        with patch(f"{_CLI_UI}.config_panel"), \
             patch(f"{_CLI_UI}.make_table"), \
             patch(f"{_CLI_UI}.console"), \
             patch(f"{_CLI_UI}.success"), \
             patch(f"{_CLI_UI}.warning") as mock_warning:
            print_report([result])
            mock_warning.assert_called()

    def test_verbose_mode_prints_per_turn(self):
        turn = make_turn(0, Decision.ALLOW)
        result = make_result(turns=[turn])

        with patch(f"{_CLI_UI}.config_panel"), \
             patch(f"{_CLI_UI}.make_table"), \
             patch(f"{_CLI_UI}.console") as mock_console, \
             patch(f"{_CLI_UI}.success"), \
             patch(f"{_CLI_UI}.warning"):
            print_report([result], verbose=True)
            # console.print should be called with per-turn info
            assert mock_console.print.call_count > 0

    def test_clean_run_calls_success(self):
        turn = make_turn(0, Decision.ALLOW)
        result = make_result(turns=[turn])

        with patch(f"{_CLI_UI}.config_panel"), \
             patch(f"{_CLI_UI}.make_table"), \
             patch(f"{_CLI_UI}.console"), \
             patch(f"{_CLI_UI}.success") as mock_success, \
             patch(f"{_CLI_UI}.warning"):
            print_report([result])
            mock_success.assert_called_once()

    def test_request_eval_block_shows_as_violation(self):
        """Pre-call block (request_eval=BLOCK) must not show as pass."""
        request_violations = [{"name": "input_rail", "message": "blocked input"}]
        turn = make_turn(
            0,
            decision=Decision.ALLOW,
            request_decision=Decision.BLOCK,
            request_violations=request_violations,
        )
        result = make_result(turns=[turn])

        with patch(f"{_CLI_UI}.config_panel"), \
             patch(f"{_CLI_UI}.make_table") as mock_table, \
             patch(f"{_CLI_UI}.console"), \
             patch(f"{_CLI_UI}.success") as mock_success, \
             patch(f"{_CLI_UI}.warning") as mock_warning:
            print_report([result])
            mock_warning.assert_called()
            mock_success.assert_not_called()
            # Verify the scenario row shows 1 violation, not 0
            scenario_table_call = mock_table.call_args_list[-1]
            rows = scenario_table_call[0][2]
            assert rows[0][2] == "1"  # violations column

    def test_request_eval_violations_added_to_response_violations(self):
        """Violations from both request_eval and response_eval are summed."""
        request_violations = [{"name": "input_rail"}]
        response_violations = [{"name": "output_policy"}]
        turn = make_turn(
            0,
            decision=Decision.INTERVENE,
            violations=response_violations,
            request_decision=Decision.BLOCK,
            request_violations=request_violations,
        )
        result = make_result(turns=[turn])

        with patch(f"{_CLI_UI}.config_panel"), \
             patch(f"{_CLI_UI}.make_table") as mock_table, \
             patch(f"{_CLI_UI}.console"), \
             patch(f"{_CLI_UI}.success"), \
             patch(f"{_CLI_UI}.warning"):
            print_report([result])
            scenario_table_call = mock_table.call_args_list[-1]
            rows = scenario_table_call[0][2]
            assert rows[0][2] == "2"  # 1 request + 1 response violation
