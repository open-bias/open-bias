from __future__ import annotations

import pytest

from openbias.policy.engines.judge.engine import JudgePolicyEngine
from openbias.policy.engines.judge.models import (
    AggregatedRuleResult,
    JudgeRuleResult,
    JudgeVerdict,
    VerdictAction,
)
from openbias.policy.protocols import EvaluationStatus


def _config(rules: list[str], models: list[dict] | None = None) -> dict:
    return {
        "models": models or [{"name": "primary", "model": "gpt-4o-mini"}],
        "_compiled_rules": rules,
        "_rules_source": "rules.md",
    }


def _verdict(action: VerdictAction, failed: list[str] | None = None) -> JudgeVerdict:
    failed = failed or []
    aggregated_results = [
        AggregatedRuleResult(
            rule=rule,
            passed=rule not in failed,
            action=VerdictAction.PASS if rule not in failed else action,
            judge_results=[
                JudgeRuleResult(
                    rule=rule,
                    passed=rule not in failed,
                    reasoning="test",
                    judge_name="primary",
                    judge_model="gpt-4o-mini",
                )
            ],
            summary="test",
        )
        for rule in ["Never reveal secrets", "Stay on task"]
    ]
    return JudgeVerdict(
        aggregated_results=aggregated_results,
        failed_rules=failed,
        action=action,
        summary="test",
        judge_models=["gpt-4o-mini"],
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
    with pytest.raises(ValueError, match="_compiled_rules"):
        await engine.initialize({"models": [{"name": "primary", "model": "gpt-4o-mini"}]})


def test_validate_config_requires_compiled_rules_and_model():
    errors = JudgePolicyEngine.validate_config({})
    assert any("No model configured" in e for e in errors)
    assert any("_compiled_rules" in e for e in errors)


@pytest.mark.asyncio
async def test_initialize_rejects_rules_file_fallback():
    engine = JudgePolicyEngine()
    with pytest.raises(ValueError, match="no longer accepts `rules_file`"):
        await engine.initialize(
            {
                "models": [{"name": "primary", "model": "gpt-4o-mini"}],
                "_compiled_rules": ["Stay safe"],
                "rules_file": "./rules.md",
            }
        )


def test_validate_config_rejects_rules_file_fallback():
    errors = JudgePolicyEngine.validate_config(
        {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "_compiled_rules": ["Stay safe"],
            "rules_file": "./rules.md",
        }
    )
    assert any("no longer accepts `rules_file`" in e for e in errors)


def test_validate_config_rejects_empty_compiled_rules():
    errors = JudgePolicyEngine.validate_config(
        {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "_compiled_rules": [],
        }
    )
    assert any("_compiled_rules" in e for e in errors)


@pytest.mark.asyncio
async def test_initialize_rejects_unknown_aggregation_mode():
    engine = JudgePolicyEngine()
    with pytest.raises(ValueError, match="aggregation_mode"):
        await engine.initialize(
            {
                "models": [{"name": "primary", "model": "gpt-4o-mini"}],
                "_compiled_rules": ["Stay safe"],
                "aggregation_mode": "sometimes",
            }
        )


@pytest.mark.asyncio
async def test_evaluate_response_maps_failed_rule_to_violation():
    engine = JudgePolicyEngine()
    await engine.initialize(_config(["Never reveal secrets", "Stay on task"]))

    async def _mock_eval(*args, **kwargs):
        rule = kwargs["rule"]
        return JudgeRuleResult(
            rule=rule,
            passed=rule != "Never reveal secrets",
            reasoning="test",
            judge_name=kwargs["model_name"],
            judge_model="gpt-4o-mini",
        )

    engine._evaluator.evaluate_rule = _mock_eval

    result = await engine.evaluate_response(
        session_id="s1",
        response_data={"content": "secret=123"},
        request_data={"messages": [{"role": "user", "content": "help"}]},
    )

    assert result.status == EvaluationStatus.VIOLATION
    assert result.violations
    assert result.violations[0].severity == "intervene"
    assert result.violations[0].extra["rule"] == "Never reveal secrets"


@pytest.mark.asyncio
async def test_evaluate_response_reports_rules_source_in_metadata():
    engine = JudgePolicyEngine()
    await engine.initialize(_config(["Never reveal secrets", "Stay on task"]))

    async def _mock_eval(*args, **kwargs):
        return JudgeRuleResult(
            rule=kwargs["rule"],
            passed=True,
            reasoning="ok",
            judge_name=kwargs["model_name"],
            judge_model="gpt-4o-mini",
        )

    engine._evaluator.evaluate_rule = _mock_eval

    result = await engine.evaluate_response(
        session_id="s1",
        response_data={"content": "all good"},
        request_data={"messages": [{"role": "user", "content": "help"}]},
    )

    verdict = result.metadata["judge"]["verdict"]
    assert verdict["rules_source"] == "rules.md"
    assert verdict["failed_rules"] == []
    assert "rubric_name" not in verdict


@pytest.mark.asyncio
async def test_evaluate_response_aggregates_multi_judge_results_per_rule():
    engine = JudgePolicyEngine()
    await engine.initialize(
        _config(
            ["Never reveal secrets", "Stay on task"],
            models=[
                {"name": "judge-a", "model": "gpt-4o-mini"},
                {"name": "judge-b", "model": "gpt-4.1-mini"},
                {"name": "judge-c", "model": "gpt-4.1-nano"},
            ],
        )
    )

    async def _mock_eval(*args, **kwargs):
        outcomes = {
            ("judge-a", "Never reveal secrets"): False,
            ("judge-b", "Never reveal secrets"): True,
            ("judge-c", "Never reveal secrets"): False,
            ("judge-a", "Stay on task"): True,
            ("judge-b", "Stay on task"): True,
            ("judge-c", "Stay on task"): False,
        }
        model_name = kwargs["model_name"]
        rule = kwargs["rule"]
        return JudgeRuleResult(
            rule=rule,
            passed=outcomes[(model_name, rule)],
            reasoning="test",
            judge_name=model_name,
            judge_model=model_name,
        )

    engine._evaluator.evaluate_rule = _mock_eval

    result = await engine.evaluate_response(
        session_id="s1",
        response_data={"content": "response"},
        request_data={"messages": [{"role": "user", "content": "help"}]},
    )

    assert result.status == EvaluationStatus.VIOLATION
    assert [violation.extra["rule"] for violation in result.violations] == [
        "Never reveal secrets"
    ]
    assert result.metadata["judge"]["verdict"]["failed_rules"] == [
        "Never reveal secrets"
    ]


@pytest.mark.asyncio
async def test_evaluate_response_fails_closed_when_all_judges_error_for_rule():
    engine = JudgePolicyEngine()
    await engine.initialize(
        _config(
            ["Never reveal secrets"],
            models=[
                {"name": "judge-a", "model": "gpt-4o-mini"},
                {"name": "judge-b", "model": "gpt-4.1-mini"},
            ],
        )
    )

    async def _mock_eval(*args, **kwargs):
        raise RuntimeError(f"judge unavailable: {kwargs['model_name']}")

    engine._evaluator.evaluate_rule = _mock_eval

    result = await engine.evaluate_response(
        session_id="s1",
        response_data={"content": "response"},
        request_data={"messages": [{"role": "user", "content": "help"}]},
    )

    assert result.status == EvaluationStatus.VIOLATION
    assert [violation.extra["rule"] for violation in result.violations] == [
        "Never reveal secrets"
    ]
    assert result.violations[0].extra["evaluation_error"] == "all_judges_failed"
