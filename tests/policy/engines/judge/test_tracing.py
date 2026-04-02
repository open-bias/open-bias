from __future__ import annotations

import pytest

from openbias.policy.engines.judge.engine import JudgePolicyEngine
from openbias.policy.engines.judge.models import JudgeScore, JudgeVerdict, VerdictAction


class _TracerSpy:
    def __init__(self):
        self.calls = []

    def log_judge_evaluation(self, **kwargs):
        self.calls.append(kwargs)


@pytest.mark.asyncio
async def test_trace_verdict_uses_compiled_rules_rubric_name():
    engine = JudgePolicyEngine()
    await engine.initialize(
        {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "_compiled_rules": ["Rule A"],
            "_rules_source": "rules.md",
        }
    )

    tracer = _TracerSpy()
    engine.set_tracer(tracer)

    async def _mock_eval(*args, **kwargs):
        return JudgeVerdict(
            scores=[JudgeScore(criterion="Rule A", score=1, reasoning="ok")],
            composite_score=1.0,
            action=VerdictAction.PASS,
            summary="ok",
            judge_model="gpt-4o-mini",
        )

    engine._evaluator.evaluate_turn = _mock_eval

    await engine.evaluate_response(
        session_id="s1",
        response_data={"content": "safe"},
        request_data={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert tracer.calls
    assert tracer.calls[0]["rubric_name"] == "rules.md"
