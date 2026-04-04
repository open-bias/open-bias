"""Policy comparison using repo-owned eval suites and replayable traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openbias.candidates import (
    CandidatePolicyBundle,
    FileCandidateProvider,
    PolicyCandidateProvider,
)
from openbias.compare.schema import (
    ComparisonGate,
    PolicyComparisonResult,
    SuiteComparison,
    TraceComparison,
)
from openbias.config.settings import Settings
from openbias.eval import EvalRunner, load_native_suites
from openbias.policy.compiler.runtime import compile_runtime_config_for_evaluator
from openbias.policy.registry import PolicyEngineRegistry
from openbias.replay import ReplayRunner
from openbias.traces import load_trace_dataset


async def build_engine_for_policy(
    *,
    settings: Settings,
    config_path: Path | None,
    rules_path: Path,
) -> Any:
    if not settings.evaluators:
        raise ValueError("Policy comparison requires at least one configured evaluator.")

    evaluator = settings.evaluators[0]
    evaluator_config = dict(evaluator.config)
    settings.inject_default_model(
        evaluator.type, evaluator_config, settings.proxy.default_model
    )
    base_dir = (config_path.parent if config_path else Path.cwd()).resolve()
    compiled = await compile_runtime_config_for_evaluator(
        evaluator_name=evaluator.name,
        evaluator_type=evaluator.type,
        evaluator_config=evaluator_config,
        default_model=settings.proxy.default_model,
        base_dir=base_dir,
        rules_path=rules_path,
    )
    return await PolicyEngineRegistry.create_and_initialize(evaluator.type, compiled)


async def compare_policy_runs(
    *,
    settings: Settings,
    config_path: Path | None,
    candidate_policy_path: Path | None = None,
    candidate_bundle: CandidatePolicyBundle | None = None,
    candidate_provider: PolicyCandidateProvider | None = None,
    trace_paths: tuple[Path, ...] = (),
    trace_regression_budget: float = 0.05,
) -> PolicyComparisonResult:
    base_dir = (config_path.parent if config_path else Path.cwd()).resolve()
    baseline_policy_path = base_dir / "rules.md"
    if not baseline_policy_path.is_file():
        raise ValueError(f"Baseline rules.md not found at {baseline_policy_path}")

    if candidate_bundle is None:
        provider = candidate_provider or FileCandidateProvider()
        candidate_bundle = provider.generate(
            baseline_policy_path=baseline_policy_path,
            candidate_policy_path=candidate_policy_path,
        )
    candidate_path = Path(candidate_bundle.policy_path)

    baseline_engine = await build_engine_for_policy(
        settings=settings,
        config_path=config_path,
        rules_path=baseline_policy_path,
    )
    candidate_engine = await build_engine_for_policy(
        settings=settings,
        config_path=config_path,
        rules_path=candidate_path,
    )

    try:
        suite_results: list[SuiteComparison] = []
        eval_runner = EvalRunner()
        for suite in load_native_suites(base_dir / "evals" / "suites"):
            baseline_run = await eval_runner.run(baseline_engine, suite)
            candidate_run = await eval_runner.run(candidate_engine, suite)
            suite_results.append(
                SuiteComparison(
                    name=suite.name,
                    baseline=baseline_run.summary.__dict__,
                    candidate=candidate_run.summary.__dict__,
                    delta_exact_case_pass_rate=(
                        candidate_run.summary.exact_case_pass_rate
                        - baseline_run.summary.exact_case_pass_rate
                    ),
                    delta_false_positive_rate=(
                        candidate_run.summary.false_positive_rate
                        - baseline_run.summary.false_positive_rate
                    ),
                )
            )

        replay_runner = ReplayRunner(fail_action=settings.fail_action)
        trace_results: list[TraceComparison] = []
        for path in trace_paths:
            dataset = load_trace_dataset(path)
            baseline_run = await replay_runner.run(baseline_engine, dataset)
            candidate_run = await replay_runner.run(candidate_engine, dataset)
            baseline_match_rate = _matched_rate(baseline_run.summary)
            candidate_match_rate = _matched_rate(candidate_run.summary)
            trace_results.append(
                TraceComparison(
                    name=dataset.name,
                    baseline=baseline_run.summary.__dict__,
                    candidate=candidate_run.summary.__dict__,
                    delta_matched_rate=candidate_match_rate - baseline_match_rate,
                    delta_intervention_rate=(
                        candidate_run.summary.intervention_rate
                        - baseline_run.summary.intervention_rate
                    ),
                    delta_block_rate=(
                        candidate_run.summary.block_rate
                        - baseline_run.summary.block_rate
                    ),
                    delta_pass_through_rate=(
                        candidate_run.summary.pass_through_rate
                        - baseline_run.summary.pass_through_rate
                    ),
                )
            )

        return build_comparison_result(
            baseline_policy_path=baseline_policy_path,
            candidate_policy_path=candidate_path,
            candidate_details=candidate_bundle.provenance_dict(),
            suites=suite_results,
            traces=trace_results,
            trace_regression_budget=trace_regression_budget,
        )
    finally:
        await baseline_engine.shutdown()
        await candidate_engine.shutdown()


def build_comparison_result(
    *,
    baseline_policy_path: Path,
    candidate_policy_path: Path,
    candidate_details: dict[str, Any] | None = None,
    suites: list[SuiteComparison],
    traces: list[TraceComparison],
    trace_regression_budget: float,
) -> PolicyComparisonResult:
    gates: list[ComparisonGate] = []
    improved = False

    for suite in suites:
        if "false_positive" in suite.name and suite.delta_false_positive_rate > 0:
            gates.append(
                ComparisonGate(
                    status="fail",
                    reason=(
                        f"Candidate increased false positives for guard suite "
                        f"{suite.name} by {suite.delta_false_positive_rate:.2%}."
                    ),
                )
            )
        if suite.delta_exact_case_pass_rate < 0:
            gates.append(
                ComparisonGate(
                    status="fail",
                    reason=(
                        f"Candidate reduced exact-case pass rate for suite "
                        f"{suite.name} by {abs(suite.delta_exact_case_pass_rate):.2%}."
                    ),
                )
            )
        if suite.delta_exact_case_pass_rate > 0 or suite.delta_false_positive_rate < 0:
            improved = True

    for trace in traces:
        if trace.delta_matched_rate < -trace_regression_budget:
            gates.append(
                ComparisonGate(
                    status="fail",
                    reason=(
                        f"Candidate exceeded the trace regression budget on {trace.name}: "
                        f"matched-rate delta {trace.delta_matched_rate:.2%}."
                    ),
                )
            )
        if trace.delta_matched_rate > 0:
            improved = True

    status = "fail" if any(gate.status == "fail" for gate in gates) else ("pass" if improved else "review")
    if status == "review":
        gates.append(
            ComparisonGate(
                status="review",
                reason="No regression gates failed; human review is still required before applying the candidate policy.",
            )
        )
    if status == "pass":
        gates.append(
            ComparisonGate(
                status="pass",
                reason="Candidate cleared default gates and improved at least one tracked metric.",
            )
        )

    return PolicyComparisonResult(
        status=status,
        baseline_policy_path=str(baseline_policy_path),
        candidate_policy_path=str(candidate_policy_path),
        candidate_details=dict(candidate_details or {}),
        suites=suites,
        traces=traces,
        gates=gates,
    )


def render_comparison_markdown(result: PolicyComparisonResult) -> str:
    """Render a compact Markdown comparison report."""

    lines = [
        "# Policy Comparison",
        "",
        f"- Status: `{result.status}`",
        f"- Baseline: `{result.baseline_policy_path}`",
        f"- Candidate: `{result.candidate_policy_path}`",
    ]
    if result.candidate_details:
        lines.append(
            f"- Candidate provider: `{result.candidate_details.get('provider', 'unknown')}`"
        )
    lines.extend(["", "## Gates"])
    for gate in result.gates:
        lines.append(f"- `{gate.status}` {gate.reason}")

    if result.suites:
        lines.extend(["", "## Eval Suites"])
        for suite in result.suites:
            lines.append(
                "- "
                f"`{suite.name}` exact-pass delta `{suite.delta_exact_case_pass_rate:+.2%}`, "
                f"false-positive delta `{suite.delta_false_positive_rate:+.2%}`"
            )

    if result.traces:
        lines.extend(["", "## Trace Replay"])
        for trace in result.traces:
            lines.append(
                "- "
                f"`{trace.name}` matched delta `{trace.delta_matched_rate:+.2%}`, "
                f"intervene delta `{trace.delta_intervention_rate:+.2%}`, "
                f"block delta `{trace.delta_block_rate:+.2%}`"
            )

    lines.append("")
    return "\n".join(lines)


def _matched_rate(summary: dict[str, Any]) -> float:
    supported_cases = int(summary.get("supported_cases", 0))
    if supported_cases == 0:
        return 0.0
    return int(summary.get("matched_cases", 0)) / supported_cases
