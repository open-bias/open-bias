from __future__ import annotations

from openbias.policy.engines.judge.models import (
    AggregatedRuleResult,
    JudgeRuleResult,
    JudgeSessionContext,
    JudgeVerdict,
    VerdictAction,
)


def test_record_verdict_updates_session_counters():
    session = JudgeSessionContext(session_id="s1")
    verdict = JudgeVerdict(
        aggregated_results=[
            AggregatedRuleResult(
                rule="Rule A",
                passed=False,
                action=VerdictAction.INTERVENE,
                judge_results=[
                    JudgeRuleResult(
                        rule="Rule A",
                        passed=False,
                        reasoning="failed",
                        judge_name="primary",
                        judge_model="gpt-4o-mini",
                    )
                ],
                summary="failed",
            )
        ],
        failed_rules=["Rule A"],
        action=VerdictAction.INTERVENE,
        summary="failed",
        judge_models=["gpt-4o-mini"],
        token_usage=12,
    )

    session.record_verdict(verdict)

    assert len(session.evaluation_history) == 1
    assert session.failed_rules_history == [["Rule A"]]
    assert session.total_tokens_used == 12
    assert session.violation_counts["intervene"] == 1
