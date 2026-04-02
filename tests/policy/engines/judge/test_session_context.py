from __future__ import annotations

from openbias.policy.engines.judge.models import JudgeScore, JudgeSessionContext, JudgeVerdict, VerdictAction


def test_record_verdict_updates_session_counters():
    session = JudgeSessionContext(session_id="s1")
    verdict = JudgeVerdict(
        scores=[JudgeScore(criterion="Rule A", score=0, reasoning="failed")],
        composite_score=0.0,
        action=VerdictAction.INTERVENE,
        summary="failed",
        judge_model="gpt-4o-mini",
        token_usage=12,
        metadata={"criterion_failures": ["Rule A"]},
    )

    session.record_verdict(verdict)

    assert len(session.evaluation_history) == 1
    assert session.score_trend == [0.0]
    assert session.total_tokens_used == 12
    assert session.violation_counts["intervene"] == 1
