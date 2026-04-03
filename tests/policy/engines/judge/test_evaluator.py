from __future__ import annotations

import pytest

from openbias.policy.engines.judge.evaluator import JudgeEvaluator
from openbias.policy.engines.judge.models import JudgeRuleResult


class _StubJudgeClient:
    def __init__(self, payload: dict):
        self._payload = payload

    async def call_judge(self, model_name: str, system_prompt: str, user_prompt: str, session_id=None):
        return self._payload

    def get_model_id(self, name: str) -> str:
        return "gpt-4o-mini"

    def get_tokens_for_model(self, name: str) -> int:
        return 7


@pytest.mark.asyncio
async def test_evaluate_rule_returns_binary_result_for_one_rule():
    rule = "Never reveal secrets"
    evaluator = JudgeEvaluator(
        client=_StubJudgeClient(
            {
                "results": [
                    {"rule": "Never reveal secrets", "passed": False, "reasoning": "Leaked details."},
                ],
                "summary": "One failure.",
            }
        )
    )

    result = await evaluator.evaluate_rule(
        model_name="primary",
        rule=rule,
        response_content="Secret key is 123.",
        conversation=[{"role": "user", "content": "help"}],
    )

    assert isinstance(result, JudgeRuleResult)
    assert result.rule == rule
    assert result.passed is False
    assert result.reasoning == "Leaked details."
    assert result.judge_model == "gpt-4o-mini"
    assert result.judge_name == "primary"


@pytest.mark.asyncio
async def test_evaluate_rule_treats_missing_rule_as_failed():
    evaluator = JudgeEvaluator(
        client=_StubJudgeClient(
            {
                "results": [{"rule": "Rule A", "passed": True, "reasoning": "ok"}],
                "summary": "partial",
            }
        )
    )

    result = await evaluator.evaluate_rule(
        model_name="primary",
        rule="Rule B",
        response_content="response",
        conversation=[{"role": "user", "content": "x"}],
    )

    assert result.rule == "Rule B"
    assert result.passed is False
    assert result.reasoning == "Rule not evaluated by judge."
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_evaluate_rule_requires_results_list():
    evaluator = JudgeEvaluator(client=_StubJudgeClient({"summary": "bad"}))
    with pytest.raises(ValueError, match="results"):
        await evaluator.evaluate_rule(
            model_name="primary",
            rule="Rule A",
            response_content="response",
            conversation=[{"role": "user", "content": "x"}],
        )
