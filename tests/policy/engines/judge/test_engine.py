from __future__ import annotations

import asyncio

import pytest

from openbias.policy.engines.judge.engine import JudgePolicyEngine
from openbias.policy.engines.judge.models import JudgeRuleResult
from openbias.policy.protocols import EvaluationStatus


def _config(rules: list[str], models: list[dict] | None = None) -> dict:
    return {
        "models": models or [{"name": "primary", "model": "gpt-4o-mini"}],
        "_compiled_rules": rules,
        "_rules_source": "rules.md",
    }


def _rule_result(
    rule: str,
    passed: bool,
    *,
    model_name: str,
    judge_model: str | None = None,
    reasoning: str = "test",
    confidence: float = 1.0,
    evidence: list[str] | None = None,
    corrective_actions: str | None = None,
) -> JudgeRuleResult:
    return JudgeRuleResult(
        rule=rule,
        passed=passed,
        reasoning=reasoning,
        evidence=evidence or [],
        confidence=confidence,
        corrective_actions=corrective_actions,
        judge_name=model_name,
        judge_model=judge_model or model_name,
    )


@pytest.mark.asyncio
async def test_initialize_loads_rules_from_compiled_rules():
    engine = JudgePolicyEngine()
    await engine.initialize(_config(["Never reveal secrets", "Stay on task"]))

    assert engine._rules == ["Never reveal secrets", "Stay on task"]
    assert engine._rules_source == "rules.md"
    assert engine._aggregation_mode == "majority"


@pytest.mark.asyncio
async def test_initialize_requires_non_empty_compiled_rules():
    engine = JudgePolicyEngine()

    with pytest.raises(ValueError, match="_compiled_rules"):
        await engine.initialize({"models": [{"name": "primary", "model": "gpt-4o-mini"}]})


def test_validate_config_requires_compiled_rules_and_model():
    errors = JudgePolicyEngine.validate_config({})

    assert any("No model configured" in error for error in errors)
    assert any("_compiled_rules" in error for error in errors)


def test_validate_config_rejects_empty_compiled_rules():
    errors = JudgePolicyEngine.validate_config(
        {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "_compiled_rules": [],
        }
    )

    assert any("_compiled_rules" in error for error in errors)


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
        return _rule_result(
            rule,
            passed=rule != "Never reveal secrets",
            model_name=kwargs["model_name"],
            judge_model="gpt-4o-mini",
            confidence=0.8,
            evidence=["secret=123"] if rule == "Never reveal secrets" else None,
            corrective_actions=(
                "Do not disclose secrets."
                if rule == "Never reveal secrets"
                else None
            ),
        )

    engine._evaluator.evaluate_rule = _mock_eval

    result = await engine.evaluate_response(
        session_id="s1",
        response_data={"content": "secret=123"},
        request_data={"messages": [{"role": "user", "content": "help"}]},
    )

    assert result.status == EvaluationStatus.VIOLATION
    assert len(result.violations) == 1

    violation = result.violations[0]
    assert violation.severity == "intervene"
    assert violation.confidence == pytest.approx(0.8)
    assert violation.extra["rule"] == "Never reveal secrets"
    assert violation.extra["aggregation_mode"] == "majority"
    assert violation.extra["summary"] == (
        "Rule failed: Never reveal secrets (failing judges: primary)"
    )
    assert violation.extra["judge_results"] == [
        {
            "rule": "Never reveal secrets",
            "passed": False,
            "reasoning": "test",
            "evidence": ["secret=123"],
            "confidence": 0.8,
            "corrective_actions": "Do not disclose secrets.",
            "judge_name": "primary",
            "judge_model": "gpt-4o-mini",
            "latency_ms": 0.0,
            "token_usage": 0,
            "metadata": {},
        }
    ]


@pytest.mark.asyncio
async def test_evaluate_response_reports_cleaned_verdict_metadata():
    engine = JudgePolicyEngine()
    await engine.initialize(_config(["Never reveal secrets", "Stay on task"]))

    async def _mock_eval(*args, **kwargs):
        return _rule_result(
            kwargs["rule"],
            passed=True,
            model_name=kwargs["model_name"],
            judge_model="gpt-4o-mini",
            reasoning="ok",
        )

    engine._evaluator.evaluate_rule = _mock_eval

    result = await engine.evaluate_response(
        session_id="s1",
        response_data={"content": "all good"},
        request_data={"messages": [{"role": "user", "content": "help"}]},
    )

    verdict = result.metadata["judge"]["verdict"]
    assert verdict == {
        "aggregated_results": [
            {
                "rule": "Never reveal secrets",
                "passed": True,
                "action": "pass",
                "judge_results": [
                    {
                        "rule": "Never reveal secrets",
                        "passed": True,
                        "reasoning": "ok",
                        "evidence": [],
                        "confidence": 1.0,
                        "corrective_actions": None,
                        "judge_name": "primary",
                        "judge_model": "gpt-4o-mini",
                        "latency_ms": 0.0,
                        "token_usage": 0,
                        "metadata": {},
                    }
                ],
                "summary": (
                    "Rule passed after 1 judge evaluation(s): Never reveal secrets"
                ),
                "aggregation_mode": "majority",
                "metadata": {
                    "passed_count": 1,
                    "failed_count": 0,
                    "participating_judges": ["primary"],
                    "failing_judges": [],
                },
            },
            {
                "rule": "Stay on task",
                "passed": True,
                "action": "pass",
                "judge_results": [
                    {
                        "rule": "Stay on task",
                        "passed": True,
                        "reasoning": "ok",
                        "evidence": [],
                        "confidence": 1.0,
                        "corrective_actions": None,
                        "judge_name": "primary",
                        "judge_model": "gpt-4o-mini",
                        "latency_ms": 0.0,
                        "token_usage": 0,
                        "metadata": {},
                    }
                ],
                "summary": "Rule passed after 1 judge evaluation(s): Stay on task",
                "aggregation_mode": "majority",
                "metadata": {
                    "passed_count": 1,
                    "failed_count": 0,
                    "participating_judges": ["primary"],
                    "failing_judges": [],
                },
            },
        ],
        "failed_rules": [],
        "action": "pass",
        "summary": "All compiled rules passed.",
        "latency_ms": 0.0,
        "token_usage": 0,
        "scope": "turn",
        "metadata": {"aggregation_mode": "majority"},
        "rules_source": "rules.md",
        "participating_judges": ["primary"],
    }


@pytest.mark.asyncio
async def test_evaluate_response_aggregates_multi_judge_results_per_rule_by_majority():
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
        return _rule_result(
            rule,
            passed=outcomes[(model_name, rule)],
            model_name=model_name,
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

    verdict = result.metadata["judge"]["verdict"]
    assert verdict["failed_rules"] == ["Never reveal secrets"]
    assert verdict["participating_judges"] == ["judge-a", "judge-b", "judge-c"]
    assert verdict["aggregated_results"][0]["metadata"] == {
        "passed_count": 1,
        "failed_count": 2,
        "participating_judges": ["judge-a", "judge-b", "judge-c"],
        "failing_judges": ["judge-a", "judge-c"],
    }
    assert verdict["aggregated_results"][1]["metadata"] == {
        "passed_count": 2,
        "failed_count": 1,
        "participating_judges": ["judge-a", "judge-b", "judge-c"],
        "failing_judges": ["judge-c"],
    }


@pytest.mark.asyncio
async def test_evaluate_response_emits_multiple_violations_for_multiple_failed_rules():
    engine = JudgePolicyEngine()
    await engine.initialize(_config(["Never reveal secrets", "Stay on task"]))

    async def _mock_eval(*args, **kwargs):
        return _rule_result(
            kwargs["rule"],
            passed=False,
            model_name=kwargs["model_name"],
            judge_model="gpt-4o-mini",
            reasoning=f"Failed {kwargs['rule']}",
        )

    engine._evaluator.evaluate_rule = _mock_eval

    result = await engine.evaluate_response(
        session_id="s1",
        response_data={"content": "response"},
        request_data={"messages": [{"role": "user", "content": "help"}]},
    )

    assert result.status == EvaluationStatus.VIOLATION
    assert [violation.extra["rule"] for violation in result.violations] == [
        "Never reveal secrets",
        "Stay on task",
    ]
    assert result.metadata["judge"]["verdict"]["failed_rules"] == [
        "Never reveal secrets",
        "Stay on task",
    ]


@pytest.mark.asyncio
async def test_evaluate_response_processes_rules_sequentially_and_judges_in_parallel():
    engine = JudgePolicyEngine()
    await engine.initialize(
        _config(
            ["Rule A", "Rule B"],
            models=[
                {"name": "judge-a", "model": "gpt-4o-mini"},
                {"name": "judge-b", "model": "gpt-4.1-mini"},
            ],
        )
    )

    started: list[tuple[str, str]] = []
    release_rule_a = asyncio.Event()
    release_rule_b = asyncio.Event()

    async def _mock_eval(*args, **kwargs):
        rule = kwargs["rule"]
        model_name = kwargs["model_name"]
        started.append((rule, model_name))
        await asyncio.sleep(0)

        started_for_rule = [name for current_rule, name in started if current_rule == rule]
        if rule == "Rule A":
            if len(started_for_rule) == 2:
                assert not any(current_rule == "Rule B" for current_rule, _ in started)
                release_rule_a.set()
            await release_rule_a.wait()
        else:
            assert len([1 for current_rule, _ in started if current_rule == "Rule A"]) == 2
            release_rule_b.set()
            await release_rule_b.wait()

        return _rule_result(rule, passed=True, model_name=model_name, judge_model=model_name)

    engine._evaluator.evaluate_rule = _mock_eval

    result = await engine.evaluate_response(
        session_id="s1",
        response_data={"content": "response"},
        request_data={"messages": [{"role": "user", "content": "help"}]},
    )

    assert result.status == EvaluationStatus.ALLOW
    assert len(started) == 4
    assert all(rule == "Rule A" for rule, _ in started[:2])
    assert all(rule == "Rule B" for rule, _ in started[2:])
    assert {judge for _, judge in started[:2]} == {"judge-a", "judge-b"}
    assert {judge for _, judge in started[2:]} == {"judge-a", "judge-b"}


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
    assert result.violations[0].extra["judge_errors"] == [
        "judge-a: judge unavailable: judge-a",
        "judge-b: judge unavailable: judge-b",
    ]
    assert result.violations[0].extra["judge_results"] == []
