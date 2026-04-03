from __future__ import annotations

import pytest

from openbias.policy.engines.judge.evaluator import JudgeEvaluator
from openbias.policy.engines.judge.models import JudgeRuleResult


class _StubJudgeClient:
    def __init__(self, payload: dict):
        self._payload = payload
        self.calls = []

    async def call_judge(self, model_name: str, system_prompt: str, user_prompt: str, session_id=None):
        self.calls.append(
            {
                "model_name": model_name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "session_id": session_id,
            }
        )
        return self._payload

    def get_model_id(self, name: str) -> str:
        return "gpt-4o-mini"

    def get_tokens_for_model(self, name: str) -> int:
        return 7


@pytest.mark.asyncio
async def test_evaluate_rule_returns_binary_result_for_one_rule():
    rule = "Never reveal secrets"
    client = _StubJudgeClient(
        {
            "rule": "Never reveal secrets",
            "passed": False,
            "reasoning": "Leaked details.",
            "summary": "Rule failed.",
        }
    )
    evaluator = JudgeEvaluator(
        client=client
    )

    result = await evaluator.evaluate_rule(
        model_name="primary",
        rule=rule,
        content_to_evaluate="Secret key is 123.",
        conversation=[{"role": "user", "content": "help"}],
        target_role="assistant",
    )

    assert isinstance(result, JudgeRuleResult)
    assert result.rule == rule
    assert result.passed is False
    assert result.reasoning == "Leaked details."
    assert result.judge_model == "gpt-4o-mini"
    assert result.judge_name == "primary"
    assert client.calls
    assert "latest assistant response" in client.calls[0]["system_prompt"]
    assert "Rule:" in client.calls[0]["system_prompt"]
    assert "- Rule 1:" not in client.calls[0]["system_prompt"]
    assert "Latest assistant response to evaluate" in client.calls[0]["user_prompt"]


@pytest.mark.asyncio
async def test_evaluate_rule_treats_missing_rule_as_failed():
    evaluator = JudgeEvaluator(
        client=_StubJudgeClient(
            {
                "rule": "Rule A",
                "passed": True,
                "reasoning": "ok",
            }
        )
    )

    result = await evaluator.evaluate_rule(
        model_name="primary",
        rule="Rule B",
        content_to_evaluate="response",
        conversation=[{"role": "user", "content": "x"}],
        target_role="assistant",
    )

    assert result.rule == "Rule B"
    assert result.passed is False
    assert result.reasoning == "Rule not evaluated by judge."
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_evaluate_rule_requires_object_payload_with_rule_verdict():
    evaluator = JudgeEvaluator(client=_StubJudgeClient({"summary": "bad"}))
    with pytest.raises(ValueError, match="rule"):
        await evaluator.evaluate_rule(
            model_name="primary",
            rule="Rule A",
            content_to_evaluate="response",
            conversation=[{"role": "user", "content": "x"}],
            target_role="assistant",
        )


@pytest.mark.asyncio
async def test_evaluate_rule_uses_user_message_prompt_for_pre_call_checks():
    client = _StubJudgeClient(
        {
            "rule": "Do not request credentials",
            "passed": True,
            "reasoning": "ok",
        }
    )
    evaluator = JudgeEvaluator(client=client)

    await evaluator.evaluate_rule(
        model_name="primary",
        rule="Do not request credentials",
        content_to_evaluate="Please share your password.",
        conversation=[{"role": "user", "content": "Please share your password."}],
        target_role="user",
    )

    assert client.calls
    assert "latest user message" in client.calls[0]["system_prompt"]
    assert "Latest user message to evaluate" in client.calls[0]["user_prompt"]


@pytest.mark.asyncio
async def test_evaluate_rule_uses_target_neutral_tool_call_labeling():
    client = _StubJudgeClient(
        {
            "rule": "Do not request credentials",
            "passed": True,
            "reasoning": "ok",
        }
    )
    evaluator = JudgeEvaluator(client=client)

    await evaluator.evaluate_rule(
        model_name="primary",
        rule="Do not request credentials",
        content_to_evaluate="Please share your password.",
        conversation=[{"role": "user", "content": "Please share your password."}],
        target_role="user",
        tool_calls=[{"function_name": "lookup_user", "arguments": "{\"id\": 1}"}],
    )

    assert client.calls
    assert "Tool calls in the evaluated content:" in client.calls[0]["user_prompt"]
    assert "Tool calls in this response:" not in client.calls[0]["user_prompt"]
