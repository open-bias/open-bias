from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openbias.candidates import CandidatePolicyBundle
from openbias.compare import build_comparison_result
from openbias.compare.runner import compare_policy_runs
from openbias.compare.schema import SuiteComparison, TraceComparison
from openbias.policy.protocols import EvaluationResult, EvaluationStatus, ViolationRecord
from openbias.traces import TraceCase, TraceDataset, TraceMetadata


@dataclass
class FakeReplayEngine:
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

    async def evaluate_request(self, session_id: str, request_data: dict[str, Any], context=None) -> EvaluationResult:
        del session_id, request_data, context
        return EvaluationResult(status=EvaluationStatus.ALLOW)

    async def evaluate_response(self, session_id: str, response_data: Any, request_data: dict[str, Any], context=None) -> EvaluationResult:
        del session_id, request_data, context
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


def _dataset(*, final_action: str, user: str, assistant: str) -> TraceDataset:
    return TraceDataset(
        name="trace",
        cases=[
            TraceCase(
                id=f"{final_action}-case",
                session_id="sess-1",
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                metadata=TraceMetadata(final_action=final_action),
            )
        ],
    )


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


@pytest.mark.asyncio
async def test_compare_policy_runs_uses_runtime_aware_replay_for_trace_deltas(tmp_path, monkeypatch):
    config_path = tmp_path / "openbias.yaml"
    config_path.write_text("evaluators: []\n", encoding="utf-8")
    (tmp_path / "rules.md").write_text("# baseline\n", encoding="utf-8")
    candidate_path = tmp_path / "rules.candidate.md"
    candidate_path.write_text("# candidate\n", encoding="utf-8")

    async def _fake_build_engine_for_policy(*, settings, config_path, rules_path):
        del settings, config_path
        if Path(rules_path).name == "rules.md":
            return FakeReplayEngine(response_rule="refund")
        return FakeReplayEngine()

    dataset = _dataset(final_action="intervene", user="hello", assistant="refund complete")
    monkeypatch.setattr("openbias.compare.runner.build_engine_for_policy", _fake_build_engine_for_policy)
    monkeypatch.setattr("openbias.compare.runner.load_native_suites", lambda _path: [])
    monkeypatch.setattr("openbias.compare.runner.load_trace_dataset", lambda _path: dataset)

    candidate_bundle = CandidatePolicyBundle(
        name="candidate",
        policy_path=str(candidate_path),
        policy_text="# candidate\n",
        provider="test",
    )
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")

    post_call_settings = SimpleNamespace(
        evaluators=[SimpleNamespace(phase="post_call")],
        mode="sync",
        fail_action="intervene",
        strategy="user_message_inject",
    )
    pre_call_settings = SimpleNamespace(
        evaluators=[SimpleNamespace(phase="pre_call")],
        mode="sync",
        fail_action="intervene",
        strategy="user_message_inject",
    )

    post_call_result = await compare_policy_runs(
        settings=post_call_settings,
        config_path=config_path,
        candidate_bundle=candidate_bundle,
        trace_paths=(trace_path,),
    )
    pre_call_result = await compare_policy_runs(
        settings=pre_call_settings,
        config_path=config_path,
        candidate_bundle=candidate_bundle,
        trace_paths=(trace_path,),
    )

    assert post_call_result.traces[0].delta_matched_rate == -1.0
    assert pre_call_result.traces[0].delta_matched_rate == 0.0
