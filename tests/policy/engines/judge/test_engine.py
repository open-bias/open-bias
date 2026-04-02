from __future__ import annotations

import pytest

from openbias.policy.engines.judge.engine import JudgePolicyEngine
from openbias.policy.engines.judge.models import JudgeScore, JudgeVerdict, VerdictAction
from openbias.policy.protocols import EvaluationStatus


def _config(rules: list[str]) -> dict:
    return {
        "models": [{"name": "primary", "model": "gpt-4o-mini"}],
        "_compiled_rules": rules,
        "_rules_source": "rules.md",
    }


def _verdict(action: VerdictAction, failed: list[str] | None = None) -> JudgeVerdict:
    failed = failed or []
    scores = [
        JudgeScore(
            criterion=rule,
            score=0 if rule in failed else 1,
            reasoning="test",
        )
        for rule in ["Never reveal secrets", "Stay on task"]
    ]
    return JudgeVerdict(
        scores=scores,
        composite_score=0.0 if failed else 1.0,
        action=action,
        summary="test",
        judge_model="gpt-4o-mini",
        metadata={"criterion_failures": failed} if failed else {},
    )


@pytest.mark.asyncio
async def test_initialize_loads_rules_from_compiled_rules():
    engine = JudgePolicyEngine()
    await engine.initialize(_config(["Never reveal secrets", "Stay on task"]))

    assert engine._rules == ["Never reveal secrets", "Stay on task"]
    assert engine._rules_source == "rules.md"


@pytest.mark.asyncio
async def test_initialize_requires_compiled_rules_when_rules_file_missing():
    engine = JudgePolicyEngine()
    with pytest.raises(ValueError, match="compiled rules"):
        await engine.initialize({"models": [{"name": "primary", "model": "gpt-4o-mini"}]})


def test_validate_config_requires_compiled_rules_and_model():
    errors = JudgePolicyEngine.validate_config({})
    assert any("No model configured" in e for e in errors)
    assert any("requires compiled rules" in e for e in errors)


@pytest.mark.asyncio
async def test_evaluate_response_maps_failed_rule_to_violation():
    engine = JudgePolicyEngine()
    await engine.initialize(_config(["Never reveal secrets", "Stay on task"]))

    async def _mock_eval(*args, **kwargs):
        return _verdict(VerdictAction.INTERVENE, failed=["Never reveal secrets"])

    engine._evaluator.evaluate_turn = _mock_eval

    result = await engine.evaluate_response(
        session_id="s1",
        response_data={"content": "secret=123"},
        request_data={"messages": [{"role": "user", "content": "help"}]},
    )

    assert result.status == EvaluationStatus.VIOLATION
    assert result.violations
    assert result.violations[0].severity == "intervene"
