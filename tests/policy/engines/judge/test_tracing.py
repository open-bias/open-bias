from __future__ import annotations

import pytest

from openbias.policy.engines.judge.engine import JudgePolicyEngine
from openbias.policy.engines.judge.models import JudgeRuleResult


class _TracerSpy:
    def __init__(self):
        self.calls = []

    def log_judge_evaluation(self, **kwargs):
        self.calls.append(kwargs)


@pytest.mark.asyncio
async def test_trace_verdict_uses_compiled_rules_source_name():
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
        return JudgeRuleResult(
            rule=kwargs["rule"],
            passed=True,
            reasoning="ok",
            judge_name=kwargs["model_name"],
            judge_model="gpt-4o-mini",
        )

    engine._evaluator.evaluate_rule = _mock_eval

    await engine.evaluate_response(
        session_id="s1",
        response_data={"content": "safe"},
        request_data={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert tracer.calls
    trace_call = tracer.calls[0]
    assert trace_call["rules_source"] == "rules.md"
    assert trace_call["evaluator_name"] == "judge"
    assert "rule_results" in trace_call
    assert "failed_rules" in trace_call
    assert "participating_judges" in trace_call
    assert "composite_score" not in trace_call
    assert not any("rubric" in key for key in trace_call)
