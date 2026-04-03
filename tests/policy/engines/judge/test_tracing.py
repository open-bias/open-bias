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
async def test_trace_verdict_emits_rule_scoped_payload():
    engine = JudgePolicyEngine()
    await engine.initialize(
        {
            "models": [
                {"name": "judge-a", "model": "gpt-4o-mini"},
                {"name": "judge-b", "model": "gpt-4.1-mini"},
            ],
            "_compiled_rules": ["Rule A"],
            "_rules_source": "rules.md",
        }
    )

    tracer = _TracerSpy()
    engine.set_tracer(tracer)

    async def _mock_eval(*args, **kwargs):
        return JudgeRuleResult(
            rule=kwargs["rule"],
            passed=kwargs["model_name"] == "judge-b",
            reasoning="ok",
            judge_name=kwargs["model_name"],
            judge_model=kwargs["model_name"],
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
    assert trace_call["scope"] == "turn"
    assert trace_call["action"] == "intervene"
    assert trace_call["failed_rules"] == ["Rule A"]
    assert trace_call["participating_judges"] == ["judge-a", "judge-b"]
    assert trace_call["metadata"] == {"aggregation_mode": "majority"}
    assert trace_call["rule_results"] == [
        {
            "rule": "Rule A",
            "passed": False,
            "action": "intervene",
            "summary": "Rule failed: Rule A (failing judges: judge-a)",
            "aggregation_mode": "majority",
            "judge_results": [
                {
                    "rule": "Rule A",
                    "passed": False,
                    "reasoning": "ok",
                    "evidence": [],
                    "confidence": 1.0,
                    "corrective_actions": None,
                    "judge_name": "judge-a",
                    "judge_model": "judge-a",
                    "latency_ms": 0.0,
                    "token_usage": 0,
                    "metadata": {},
                },
                {
                    "rule": "Rule A",
                    "passed": True,
                    "reasoning": "ok",
                    "evidence": [],
                    "confidence": 1.0,
                    "corrective_actions": None,
                    "judge_name": "judge-b",
                    "judge_model": "judge-b",
                    "latency_ms": 0.0,
                    "token_usage": 0,
                    "metadata": {},
                },
            ],
        }
    ]
