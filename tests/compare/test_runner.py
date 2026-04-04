from pathlib import Path

from openbias.compare import build_comparison_result
from openbias.compare.schema import SuiteComparison, TraceComparison


def test_build_comparison_result_fails_on_guard_false_positive_regression():
    result = build_comparison_result(
        baseline_policy_path=Path("rules.md"),
        candidate_policy_path=Path("rules.candidate.md"),
        candidate_details={"provider": "file"},
        suites=[
            SuiteComparison(
                name="false_positive_guards",
                baseline={"exact_case_pass_rate": 1.0, "false_positive_rate": 0.0},
                candidate={"exact_case_pass_rate": 1.0, "false_positive_rate": 0.2},
                delta_exact_case_pass_rate=0.0,
                delta_false_positive_rate=0.2,
            )
        ],
        traces=[],
        trace_regression_budget=0.05,
    )

    assert result.status == "fail"
    assert any("false positives" in gate.reason for gate in result.gates)


def test_build_comparison_result_passes_when_candidate_improves():
    result = build_comparison_result(
        baseline_policy_path=Path("rules.md"),
        candidate_policy_path=Path("rules.candidate.md"),
        candidate_details={"provider": "file"},
        suites=[
            SuiteComparison(
                name="safe",
                baseline={"exact_case_pass_rate": 0.8, "false_positive_rate": 0.1},
                candidate={"exact_case_pass_rate": 0.9, "false_positive_rate": 0.05},
                delta_exact_case_pass_rate=0.1,
                delta_false_positive_rate=-0.05,
            )
        ],
        traces=[
            TraceComparison(
                name="prod-traces",
                baseline={"supported_cases": 10, "matched_cases": 8, "intervention_rate": 0.2, "block_rate": 0.0, "pass_through_rate": 0.8},
                candidate={"supported_cases": 10, "matched_cases": 9, "intervention_rate": 0.3, "block_rate": 0.0, "pass_through_rate": 0.7},
                delta_matched_rate=0.1,
                delta_intervention_rate=0.1,
                delta_block_rate=0.0,
                delta_pass_through_rate=-0.1,
            )
        ],
        trace_regression_budget=0.05,
    )

    assert result.status == "pass"
