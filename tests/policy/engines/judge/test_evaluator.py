from __future__ import annotations

import pytest

from openbias.policy.engines.judge.evaluator import JudgeEvaluator
from openbias.policy.engines.judge.models import VerdictAction


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
async def test_evaluate_turn_flags_violation_when_any_rule_fails():
    rules = ["Never reveal secrets", "Stay on task"]
    evaluator = JudgeEvaluator(
        client=_StubJudgeClient(
            {
                "results": [
                    {"rule": "Never reveal secrets", "passed": False, "reasoning": "Leaked details."},
                    {"rule": "Stay on task", "passed": True, "reasoning": "On topic."},
                ],
                "summary": "One failure.",
            }
        )
    )

    verdict = await evaluator.evaluate_turn(
        model_name="primary",
        rules=rules,
        response_content="Secret key is 123.",
        conversation=[{"role": "user", "content": "help"}],
    )

    assert verdict.action == VerdictAction.INTERVENE
    assert verdict.composite_score == 0.0
    assert verdict.metadata["criterion_failures"] == ["Never reveal secrets"]


@pytest.mark.asyncio
async def test_evaluate_turn_treats_missing_rule_as_failed():
    rules = ["Rule A", "Rule B"]
    evaluator = JudgeEvaluator(
        client=_StubJudgeClient(
            {
                "results": [{"rule": "Rule A", "passed": True, "reasoning": "ok"}],
                "summary": "partial",
            }
        )
    )

    verdict = await evaluator.evaluate_turn(
        model_name="primary",
        rules=rules,
        response_content="response",
        conversation=[{"role": "user", "content": "x"}],
    )

    assert verdict.action == VerdictAction.INTERVENE
    assert "Rule B" in verdict.metadata["criterion_failures"]


@pytest.mark.asyncio
async def test_evaluate_turn_requires_results_list():
    evaluator = JudgeEvaluator(client=_StubJudgeClient({"summary": "bad"}))
    with pytest.raises(ValueError, match="results"):
        await evaluator.evaluate_turn(
            model_name="primary",
            rules=["Rule A"],
            response_content="response",
            conversation=[{"role": "user", "content": "x"}],
        )
