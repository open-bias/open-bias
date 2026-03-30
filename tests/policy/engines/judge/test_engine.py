"""
Tests for JudgePolicyEngine.
"""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openbias.core.utils import extract_response_content
from openbias.policy.engines.judge import JudgePolicyEngine
from openbias.policy.engines.judge.models import (
    JudgeVerdict,
    VerdictAction,
    EvaluationScope,
    JudgeScore,
)
from openbias.policy.protocols import Decision
from openbias.policy.registry import PolicyEngineRegistry


@pytest.fixture
def engine():
    """Create an uninitialized engine."""
    return JudgePolicyEngine()


@pytest.fixture
def judge_config():
    """Minimal judge engine configuration."""
    return {
        "models": [
            {"name": "primary", "model": "gpt-4o-mini"},
        ],
    }


@pytest.fixture
def full_config():
    """Full judge engine configuration."""
    return {
        "models": [
            {"name": "primary", "model": "gpt-4o-mini", "temperature": 0.0},
        ],
        "default_rubric": "agent_behavior",
        "max_intervention_attempts": 3,
    }


@pytest.fixture
def sample_request():
    return {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        "model": "gpt-4o",
    }


@pytest.fixture
def sample_response():
    return {
        "choices": [
            {"message": {"content": "Hello! How can I help you today?"}},
        ],
    }


def _passing_judge_response():
    return {
        "scores": [
            {"criterion": "instruction_following", "score": 5, "reasoning": "Good", "evidence": [], "confidence": 0.9},
            {"criterion": "tool_use_safety", "score": 5, "reasoning": "Safe", "evidence": [], "confidence": 0.9},
            {"criterion": "no_hallucination", "score": 5, "reasoning": "Grounded", "evidence": [], "confidence": 0.9},
            {"criterion": "task_completion", "score": 4, "reasoning": "Progress", "evidence": [], "confidence": 0.8},
        ],
        "summary": "Good response overall.",
    }


def _failing_judge_response():
    return {
        "scores": [
            {"criterion": "instruction_following", "score": 1, "reasoning": "Ignored", "evidence": [], "confidence": 0.9},
            {"criterion": "tool_use_safety", "score": 1, "reasoning": "Dangerous", "evidence": [], "confidence": 0.9},
            {"criterion": "no_hallucination", "score": 1, "reasoning": "Hallucinated", "evidence": [], "confidence": 0.9},
            {"criterion": "task_completion", "score": 1, "reasoning": "No progress", "evidence": [], "confidence": 0.9},
        ],
        "summary": "Very poor response.",
    }


class TestRegistration:
    def test_engine_registered(self):
        engine = PolicyEngineRegistry.create("judge")
        assert isinstance(engine, JudgePolicyEngine)

    def test_engine_type(self, engine):
        assert engine.engine_type == "judge"


class TestInitialization:
    async def test_initialize_minimal(self, engine, judge_config):
        await engine.initialize(judge_config)
        assert engine._initialized
        assert engine.name == "judge:agent_behavior"

    async def test_initialize_full_config(self, engine, full_config):
        await engine.initialize(full_config)
        assert engine._initialized
        assert engine._max_intervention_attempts == 3

    async def test_initialize_raises_without_models(self, engine):
        """Test that engine raises ValueError when no models provided."""
        with pytest.raises(ValueError, match="Judge engine requires a model"):
            await engine.initialize({})

    async def test_initialize_raises_when_default_rubric_not_found(self, engine):
        """Test that engine raises ValueError when default rubric doesn't exist."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "default_rubric": "nonexistent_rubric",
        }
        with pytest.raises(ValueError, match="Default rubric 'nonexistent_rubric' not found"):
            await engine.initialize(config)


class TestEvaluateRequest:
    async def test_raises_when_uninitialized(self, engine, sample_request):
        with pytest.raises(RuntimeError, match="not initialized"):
            await engine.evaluate_request("s1", sample_request)

    async def test_always_evaluates_with_default_rubric(self, engine, sample_request):
        """evaluate_request always runs the default rubric (no pre_call_enabled guard)."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
        }
        await engine.initialize(config)
        engine._client.call_judge = AsyncMock(return_value=_passing_judge_response())

        result = await engine.evaluate_request("s1", sample_request)

        assert result.decision == Decision.ALLOW
        engine._client.call_judge.assert_called_once()

    async def test_failing_request(self, engine, sample_request):
        """Failing rubric evaluation results in INTERVENE."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
        }
        await engine.initialize(config)
        engine._client.call_judge = AsyncMock(return_value=_failing_judge_response())

        result = await engine.evaluate_request("s1", sample_request)

        assert result.decision == Decision.INTERVENE
        violations = result.metadata.get("violations", [])
        assert len(violations) > 0

    async def test_empty_user_message_allows(self, engine):
        """No user message in request → ALLOW without calling judge."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
        }
        await engine.initialize(config)

        result = await engine.evaluate_request("s1", {"messages": []})
        assert result.decision == Decision.ALLOW

    async def test_judge_error_failopen(self, engine, sample_request):
        """Judge LLM error → fail-open to ALLOW."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
        }
        await engine.initialize(config)
        engine._client.call_judge = AsyncMock(side_effect=Exception("LLM error"))

        result = await engine.evaluate_request("s1", sample_request)
        assert result.decision == Decision.ALLOW


class TestEvaluateResponse:
    async def test_raises_when_uninitialized(self, engine, sample_request, sample_response):
        with pytest.raises(RuntimeError, match="not initialized"):
            await engine.evaluate_response("s1", sample_response, sample_request)

    async def test_passing_response(self, engine, judge_config, sample_request, sample_response):
        await engine.initialize(judge_config)
        engine._client.call_judge = AsyncMock(return_value=_passing_judge_response())

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result.decision == Decision.ALLOW
        assert len(result.metadata.get("violations", [])) == 0

    async def test_failing_response(self, engine, judge_config, sample_request, sample_response):
        await engine.initialize(judge_config)
        engine._client.call_judge = AsyncMock(return_value=_failing_judge_response())

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result.decision == Decision.INTERVENE
        violations = result.metadata.get("violations", [])
        assert len(violations) > 0
        for v in violations:
            assert "name" in v
            assert "message" in v
            assert "severity" in v

    async def test_judge_metadata_in_result(self, engine, judge_config, sample_request, sample_response):
        await engine.initialize(judge_config)
        engine._client.call_judge = AsyncMock(return_value=_passing_judge_response())

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        assert "judge" in result.metadata
        assert "verdicts" in result.metadata["judge"]

    async def test_session_turn_starts_at_one(self, engine, judge_config, sample_request, sample_response):
        """session_turn in metadata should be 1 on the first evaluation, not 0."""
        await engine.initialize(judge_config)
        engine._client.call_judge = AsyncMock(return_value=_passing_judge_response())

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result.metadata["judge"]["session_turn"] == 1

    async def test_session_turn_increments(self, engine, judge_config, sample_request, sample_response):
        """session_turn should increment with each evaluation."""
        await engine.initialize(judge_config)
        engine._client.call_judge = AsyncMock(return_value=_passing_judge_response())

        result1 = await engine.evaluate_response("s1", sample_response, sample_request)
        result2 = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result1.metadata["judge"]["session_turn"] == 1
        assert result2.metadata["judge"]["session_turn"] == 2

    async def test_llm_error_failopen(self, engine, judge_config, sample_request, sample_response):
        """Engine should fail-open if judge LLM call raises."""
        await engine.initialize(judge_config)
        engine._client.call_judge = AsyncMock(side_effect=Exception("LLM error"))

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result.decision == Decision.ALLOW

    async def test_string_response_data(self, engine, judge_config, sample_request):
        """Should handle string response_data."""
        await engine.initialize(judge_config)
        engine._client.call_judge = AsyncMock(return_value=_passing_judge_response())

        result = await engine.evaluate_response("s1", "Hello!", sample_request)
        assert result.decision == Decision.ALLOW


class TestSessionManagement:
    async def test_get_session_state(self, engine, judge_config):
        await engine.initialize(judge_config)
        engine._get_or_create_session("s1")

        state = await engine.get_session_state("s1")
        assert state is not None
        assert state["session_id"] == "s1"

    async def test_get_session_state_nonexistent(self, engine, judge_config):
        await engine.initialize(judge_config)
        state = await engine.get_session_state("nonexistent")
        assert state is None

    async def test_reset_session(self, engine, judge_config):
        await engine.initialize(judge_config)
        engine._get_or_create_session("s1")
        assert "s1" in engine._sessions

        await engine.reset_session("s1")
        assert "s1" not in engine._sessions

    async def test_reset_nonexistent_session(self, engine, judge_config):
        """Resetting a nonexistent session should not raise."""
        await engine.initialize(judge_config)
        await engine.reset_session("nonexistent")

    async def test_shutdown(self, engine, judge_config):
        await engine.initialize(judge_config)
        engine._get_or_create_session("s1")
        await engine.shutdown()
        assert len(engine._sessions) == 0

    async def test_shutdown_resets_initialized_flag(self, engine, judge_config):
        await engine.initialize(judge_config)
        assert engine._initialized
        await engine.shutdown()
        assert not engine._initialized

    async def test_evaluate_response_raises_after_shutdown(
        self, engine, judge_config, sample_request, sample_response
    ):
        await engine.initialize(judge_config)
        await engine.shutdown()
        with pytest.raises(RuntimeError, match="not initialized"):
            await engine.evaluate_response("s1", sample_response, sample_request)


class TestResponseExtraction:
    def test_extract_openai_format(self, engine):
        data = {"choices": [{"message": {"content": "Hello"}}]}
        assert extract_response_content(data) == "Hello"

    def test_extract_string(self, engine):
        assert extract_response_content("Hello") == "Hello"

    def test_extract_dict_content(self, engine):
        assert extract_response_content({"content": "Hello"}) == "Hello"

    def test_extract_fallback(self, engine):
        assert extract_response_content(42) == "42"


class TestConfigShorthands:
    """Tests for config shorthands: `rubric` and `policies`."""

    async def test_rubric_shorthand(self, engine):
        """rubric: 'name' sets default_rubric."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "rubric": "agent_behavior",
        }
        await engine.initialize(config)
        assert engine._default_rubric == "agent_behavior"

    async def test_policies_shorthand(self, engine):
        """policies: [list] treated as inline_policy."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "policies": ["Be safe", "Be helpful"],
        }
        await engine.initialize(config)
        assert engine._default_rubric == "inline_policy"
        rubric = engine._registry.get("inline_policy")
        assert rubric is not None

    async def test_explicit_overrides_shorthand(self, engine):
        """Explicit default_rubric takes precedence over rubric shorthand."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "rubric": "safety",
            "default_rubric": "agent_behavior",
        }
        await engine.initialize(config)
        assert engine._default_rubric == "agent_behavior"

    async def test_max_intervention_attempts_from_config(self):
        """max_intervention_attempts is read from config."""
        engine = JudgePolicyEngine()
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "max_intervention_attempts": 5,
        }
        await engine.initialize(config)
        assert engine._max_intervention_attempts == 5


class TestPerRuleCriteria:
    """Tests for per-rule criteria in inline policies (Step 3)."""

    async def test_one_rule_violated_cites_specific_rule(self, engine, sample_request, sample_response):
        """3 rules defined, 1 violated → result cites the specific rule."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": [
                "Never provide financial advice",
                "Be professional",
                "Do not share personal opinions",
            ],
        }
        await engine.initialize(config)

        rubric = engine._registry.get("inline_policy")
        criteria_names = [c.name for c in rubric.criteria]

        # Simulate: first rule fails, others pass
        judge_response = {
            "scores": [
                {"criterion": criteria_names[0], "score": 0, "reasoning": "Gave stock tips", "evidence": [], "confidence": 0.9},
                {"criterion": criteria_names[1], "score": 1, "reasoning": "Professional tone", "evidence": [], "confidence": 0.9},
                {"criterion": criteria_names[2], "score": 1, "reasoning": "No opinions", "evidence": [], "confidence": 0.9},
            ],
            "summary": "Rule violation detected.",
        }
        engine._client.call_judge = AsyncMock(return_value=judge_response)

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result.decision == Decision.INTERVENE
        assert "Gave stock tips" in result.message
        assert "Please adjust your response accordingly." in result.message

    async def test_all_rules_pass(self, engine, sample_request, sample_response):
        """All rules pass → ALLOW."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": ["Be helpful", "Be safe"],
        }
        await engine.initialize(config)

        rubric = engine._registry.get("inline_policy")
        criteria_names = [c.name for c in rubric.criteria]

        judge_response = {
            "scores": [
                {"criterion": criteria_names[0], "score": 1, "reasoning": "Helpful", "evidence": [], "confidence": 0.9},
                {"criterion": criteria_names[1], "score": 1, "reasoning": "Safe", "evidence": [], "confidence": 0.9},
            ],
            "summary": "All good.",
        }
        engine._client.call_judge = AsyncMock(return_value=judge_response)

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result.decision == Decision.ALLOW

    async def test_multiple_rules_violated_all_listed(self, engine, sample_request, sample_response):
        """Multiple rules violated → all are listed in the intervention message."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": ["No financial advice", "Be professional", "No personal opinions"],
        }
        await engine.initialize(config)

        rubric = engine._registry.get("inline_policy")
        criteria_names = [c.name for c in rubric.criteria]

        judge_response = {
            "scores": [
                {"criterion": criteria_names[0], "score": 0, "reasoning": "Gave investment tips", "evidence": [], "confidence": 0.9},
                {"criterion": criteria_names[1], "score": 1, "reasoning": "OK tone", "evidence": [], "confidence": 0.9},
                {"criterion": criteria_names[2], "score": 0, "reasoning": "Shared personal view", "evidence": [], "confidence": 0.9},
            ],
            "summary": "Multiple violations.",
        }
        engine._client.call_judge = AsyncMock(return_value=judge_response)

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result.decision == Decision.INTERVENE
        # Both failed criteria reasoning should be cited
        assert "Gave investment tips" in result.message
        assert "Shared personal view" in result.message


class TestTargetedInterventionMessages:
    """Tests for natural-language intervention messages."""

    def test_message_uses_corrective_actions_as_primary(self, engine):
        """When corrective_actions is present, it leads the message."""
        verdict = JudgeVerdict(
            scores=[
                JudgeScore(
                    criterion="tool_use_safety",
                    score=0,
                    max_score=1,
                    reasoning="The agent called delete_database() which is unauthorized.",
                    evidence=["delete_database(table='users')"],
                    confidence=0.95,
                    corrective_actions="Ask for explicit user confirmation before destructive operations.",
                ),
                JudgeScore(
                    criterion="instruction_following",
                    score=1,
                    max_score=1,
                    reasoning="Good",
                    confidence=0.9,
                ),
            ],
            composite_score=0.3,
            action=VerdictAction.INTERVENE,
            summary="Dangerous tool use detected",
            judge_model="gpt-4o-mini",
            metadata={"criterion_failures": ["tool_use_safety"]},
        )

        message = engine._build_violation_message(verdict)
        # Natural language — no machine labels
        assert "POLICY VIOLATION" not in message
        assert "FAILED" not in message
        # Corrective action leads
        assert "Ask for explicit user confirmation" in message
        # Evidence inlined as quotes
        assert "delete_database(table='users')" in message
        # Closing guidance
        assert "Please adjust your response accordingly." in message

    def test_multiple_failures_all_included(self, engine):
        """Multiple failures should all be included as separate paragraphs."""
        verdict = JudgeVerdict(
            scores=[
                JudgeScore(
                    criterion="tool_use_safety",
                    score=0,
                    max_score=1,
                    reasoning="Called dangerous API.",
                    evidence=["delete_all()"],
                    confidence=0.95,
                    corrective_actions="Do not call destructive APIs without confirmation.",
                ),
                JudgeScore(
                    criterion="instruction_following",
                    score=0,
                    max_score=1,
                    reasoning="Ignored task constraints.",
                    evidence=["off-topic response"],
                    confidence=0.9,
                    corrective_actions="Stay on the assigned topic.",
                ),
            ],
            composite_score=0.1,
            action=VerdictAction.BLOCK,
            summary="Multiple violations",
            judge_model="gpt-4o-mini",
            metadata={"criterion_failures": ["tool_use_safety", "instruction_following"]},
        )

        message = engine._build_violation_message(verdict)
        assert "Do not call destructive APIs" in message
        assert "Stay on the assigned topic" in message
        # Separated by paragraph breaks
        assert "\n\n" in message
        # No machine labels
        assert "POLICY VIOLATION" not in message
        assert "FAILED" not in message

    def test_fallback_to_reasoning_without_corrective_actions(self, engine):
        """Without corrective_actions, uses reasoning + evidence."""
        verdict = JudgeVerdict(
            scores=[
                JudgeScore(
                    criterion="tool_use_safety",
                    score=0,
                    max_score=1,
                    reasoning="Called dangerous API.",
                    evidence=["drop_table()"],
                    confidence=0.95,
                ),
            ],
            composite_score=0.2,
            action=VerdictAction.INTERVENE,
            summary="Violation detected",
            judge_model="gpt-4o-mini",
            metadata={"criterion_failures": ["tool_use_safety"]},
        )

        message = engine._build_violation_message(verdict)
        assert "Called dangerous API." in message
        assert '"drop_table()"' in message
        assert "POLICY VIOLATION" not in message
        assert "FAILED" not in message

    def test_no_criterion_failures_falls_back_to_summary(self, engine):
        """When no criterion failures exist, fall back to verdict summary."""
        verdict = JudgeVerdict(
            scores=[
                JudgeScore(
                    criterion="quality",
                    score=2,
                    max_score=5,
                    reasoning="Low quality",
                    confidence=0.8,
                ),
            ],
            composite_score=0.25,
            action=VerdictAction.INTERVENE,
            summary="Minor issues with response quality.",
            judge_model="gpt-4o-mini",
            metadata={},
        )

        message = engine._build_violation_message(verdict)
        assert "Minor issues with response quality." in message
        # Non-directive summaries get actionable guidance appended
        assert "Please review and adjust" in message

    def test_no_machine_slugs_in_output(self, engine):
        """Output must not contain criterion slugs like rule_1_*."""
        verdict = JudgeVerdict(
            scores=[
                JudgeScore(
                    criterion="rule_1_no_financial_advice",
                    score=0,
                    max_score=1,
                    reasoning="The response provides stock recs.",
                    evidence=["I recommend buying AAPL"],
                    confidence=0.9,
                    corrective_actions="Avoid providing financial advice. Suggest consulting a financial advisor.",
                ),
            ],
            composite_score=0.0,
            action=VerdictAction.INTERVENE,
            summary="Financial advice violation",
            judge_model="gpt-4o-mini",
            metadata={"criterion_failures": ["rule_1_no_financial_advice"]},
        )

        message = engine._build_violation_message(verdict)
        assert "rule_1_no_financial_advice" not in message
        assert "POLICY VIOLATION" not in message
        assert "FAILED" not in message
        assert "Avoid providing financial advice" in message


class TestInlinePolicy:
    async def test_initialize_with_inline_rules(self, engine):
        """Engine should load inline rules and set default rubric."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": [
                "No financial advice",
                "Be professional",
            ],
        }
        await engine.initialize(config)
        assert engine._initialized
        assert engine._default_rubric == "inline_policy"

        # Verify rubric is registered
        rubric = engine._registry.get("inline_policy")
        assert rubric is not None
        assert "No financial advice" in rubric.prompt_overrides["additional_instructions"]

    async def test_initialize_with_inline_dict_rules_raises(self, engine):
        """Dict-format inline policy should raise ValueError."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": {
                "rules": ["Never lie", "Stay on topic"],
            },
        }
        with pytest.raises(ValueError, match="Dict-format inline policy is no longer supported"):
            await engine.initialize(config)

    async def test_initialize_with_inline_rubrics_dict_raises(self, engine):
        """Dict-format inline policy with rubrics should raise ValueError."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": {
                "rubrics": [{
                    "name": "my_custom",
                    "description": "Test rubric",
                    "criteria": [{
                        "name": "tone",
                        "description": "Professional tone",
                        "scale": "binary",
                    }],
                }],
            },
        }
        with pytest.raises(ValueError, match="Dict-format inline policy is no longer supported"):
            await engine.initialize(config)

    async def test_inline_policy_does_not_break_custom_rubrics_path(self, engine):
        """custom_rubrics_path and inline_policy should coexist."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": ["Be kind"],
        }
        await engine.initialize(config)
        # Should have the inline_policy rubric as default
        assert engine._default_rubric == "inline_policy"
        # But built-in rubrics should still be available
        assert engine._registry.get("agent_behavior") is not None


class TestInterventionEscalation:
    """Tests for intervention tracking and escalation (Step 6)."""

    async def test_first_violation_intervene_second_same_violation_escalated(
        self, engine, sample_request, sample_response
    ):
        """First violation → INTERVENE, second same criterion → BLOCK via escalation."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": ["Never delete user data"],
        }
        await engine.initialize(config)

        rubric = engine._registry.get("inline_policy")
        criteria_names = [c.name for c in rubric.criteria]

        fail_response = {
            "scores": [
                {
                    "criterion": criteria_names[0],
                    "score": 0,
                    "reasoning": "Deleted user records",
                    "evidence": [],
                    "confidence": 0.9,
                },
            ],
            "summary": "Policy violation.",
        }
        engine._client.call_judge = AsyncMock(return_value=fail_response)

        result1 = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result1.decision == Decision.INTERVENE
        assert result1.metadata.get("escalated") is not True

        # Second violation on same criterion → should escalate INTERVENE → BLOCK
        result2 = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result2.decision == Decision.BLOCK
        assert result2.metadata.get("escalated") is True
        assert "repeat" in result2.metadata.get("escalation_reason", "").lower()
        assert "ESCALATED" in result2.message

    async def test_different_criteria_violations_no_cross_escalation(
        self, engine, sample_request, sample_response
    ):
        """Different criteria violations don't cross-escalate."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": ["No financial advice", "Be professional"],
        }
        await engine.initialize(config)

        rubric = engine._registry.get("inline_policy")
        criteria_names = [c.name for c in rubric.criteria]

        # First violation: criterion 0 fails
        fail_response_1 = {
            "scores": [
                {"criterion": criteria_names[0], "score": 0, "reasoning": "Gave tips", "evidence": [], "confidence": 0.9},
                {"criterion": criteria_names[1], "score": 1, "reasoning": "OK", "evidence": [], "confidence": 0.9},
            ],
            "summary": "Violation.",
        }
        engine._client.call_judge = AsyncMock(return_value=fail_response_1)
        result1 = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result1.decision == Decision.INTERVENE

        # Second violation: different criterion (1) fails, criterion 0 passes
        fail_response_2 = {
            "scores": [
                {"criterion": criteria_names[0], "score": 1, "reasoning": "OK now", "evidence": [], "confidence": 0.9},
                {"criterion": criteria_names[1], "score": 0, "reasoning": "Rude", "evidence": [], "confidence": 0.9},
            ],
            "summary": "Different violation.",
        }
        engine._client.call_judge = AsyncMock(return_value=fail_response_2)
        result2 = await engine.evaluate_response("s1", sample_response, sample_request)
        # Should NOT escalate — different criterion
        assert result2.metadata.get("escalated") is not True

    async def test_intervention_count_cap_triggers_escalation(
        self, engine, sample_request, sample_response
    ):
        """Total intervention count exceeding cap (3) triggers escalation to BLOCK."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": [
                "Rule A",
                "Rule B",
                "Rule C",
                "Rule D",
                "Rule E",
            ],
        }
        await engine.initialize(config)

        rubric = engine._registry.get("inline_policy")
        criteria_names = [c.name for c in rubric.criteria]

        # Simulate 4 different criterion violations (each unique, no repeat escalation)
        for i in range(4):
            scores = []
            for j, name in enumerate(criteria_names):
                scores.append({
                    "criterion": name,
                    "score": 0 if j == i else 1,
                    "reasoning": f"{'Fail' if j == i else 'Pass'}",
                    "evidence": [],
                    "confidence": 0.9,
                })
            response = {"scores": scores, "summary": f"Violation {i+1}."}
            engine._client.call_judge = AsyncMock(return_value=response)
            result = await engine.evaluate_response("s1", sample_response, sample_request)

        # 4th violation should have triggered the count cap (>3) → BLOCK
        assert result.decision == Decision.BLOCK
        assert result.metadata.get("escalated") is True
        assert "intervention_count_exceeded" in result.metadata.get("escalation_reason", "")

    def test_session_context_tracks_intervention_criteria(self):
        """JudgeSessionContext correctly tracks intervention criteria.

        record_verdict tracks per-criterion counts and last_intervention_criteria,
        but does NOT increment intervention_count or turn_count — those are
        managed by the engine's evaluate_response() to avoid double-counting
        when multiple verdicts are recorded per evaluation.
        """
        from openbias.policy.engines.judge.models import JudgeSessionContext
        session = JudgeSessionContext(session_id="test")

        # Record a verdict with criterion failures
        verdict = JudgeVerdict(
            scores=[
                JudgeScore(criterion="safety", score=0, max_score=1, reasoning="Bad"),
            ],
            composite_score=0.0,
            action=VerdictAction.INTERVENE,
            summary="Unsafe",
            judge_model="test",
            metadata={"criterion_failures": ["safety"]},
        )
        session.record_verdict(verdict)

        # intervention_count is NOT incremented by record_verdict
        assert session.intervention_count == 0
        assert session.last_intervention_criteria == ["safety"]
        assert session.criterion_intervention_counts == {"safety": 1}

        # Record a second intervention on the same criterion
        session.record_verdict(verdict)
        assert session.intervention_count == 0
        assert session.criterion_intervention_counts == {"safety": 2}

    def test_session_context_no_tracking_on_pass(self):
        """Passing verdicts should not affect intervention tracking."""
        from openbias.policy.engines.judge.models import JudgeSessionContext
        session = JudgeSessionContext(session_id="test")

        verdict = JudgeVerdict(
            scores=[
                JudgeScore(criterion="safety", score=1, max_score=1, reasoning="Good"),
            ],
            composite_score=1.0,
            action=VerdictAction.PASS,
            summary="OK",
            judge_model="test",
            metadata={},
        )
        session.record_verdict(verdict)

        assert session.intervention_count == 0
        assert session.last_intervention_criteria == []
        assert session.criterion_intervention_counts == {}

    def test_intervene_without_criterion_failures_no_count(self):
        """record_verdict does not increment intervention_count (engine does)."""
        from openbias.policy.engines.judge.models import JudgeSessionContext
        session = JudgeSessionContext(session_id="test")

        verdict = JudgeVerdict(
            scores=[
                JudgeScore(criterion="quality", score=2, max_score=5, reasoning="Low"),
            ],
            composite_score=0.3,
            action=VerdictAction.INTERVENE,
            summary="Below threshold",
            judge_model="test",
            metadata={},  # No criterion_failures
        )
        session.record_verdict(verdict)

        assert session.intervention_count == 0
        assert session.criterion_intervention_counts == {}

    def test_block_without_criterion_failures_no_count(self):
        """record_verdict does not increment intervention_count (engine does)."""
        from openbias.policy.engines.judge.models import JudgeSessionContext
        session = JudgeSessionContext(session_id="test")

        verdict = JudgeVerdict(
            scores=[
                JudgeScore(criterion="quality", score=1, max_score=5, reasoning="Very low"),
            ],
            composite_score=0.1,
            action=VerdictAction.BLOCK,
            summary="Blocked",
            judge_model="test",
            metadata={},  # No criterion_failures
        )
        session.record_verdict(verdict)

        assert session.intervention_count == 0
        assert session.criterion_intervention_counts == {}

    def test_escalation_cap_works_with_composite_only_verdicts(self):
        """Intervention count cap triggers on composite-only verdicts
        (no criterion_failures) via direct session model testing.
        """
        from openbias.policy.engines.judge.models import JudgeSessionContext
        session = JudgeSessionContext(session_id="test")

        # Simulate 4 evaluations each with an INTERVENE verdict.
        # intervention_count is managed by the engine, so we set it directly
        # to mimic 4 prior evaluations that each had a failure.
        session.intervention_count = 4
        # No per-criterion tracking since no failures
        assert session.criterion_intervention_counts == {}

        # Escalation check: pending_count (4+1=5) > 3 should trigger
        engine = JudgePolicyEngine()
        next_verdict = JudgeVerdict(
            scores=[
                JudgeScore(criterion="quality", score=2, max_score=5, reasoning="Low"),
            ],
            composite_score=0.3,
            action=VerdictAction.INTERVENE,
            summary="Still low",
            judge_model="test",
            metadata={},
        )
        result = engine._check_escalation(next_verdict, session)
        assert result["should_escalate"] is True
        assert "intervention_count_exceeded" in result["reason"]

    def test_no_escalation_metadata_when_already_block(self):
        """Escalation metadata is NOT added when decision is already BLOCK.

        Scenario: conversation verdict is BLOCK (worst), turn verdict is
        INTERVENE with a repeat criterion violation. Escalation conditions are
        met, but the decision was already BLOCK — no actual upgrade occurred,
        so escalation metadata would be misleading.
        """
        from openbias.policy.engines.judge.models import JudgeSessionContext

        session = JudgeSessionContext(session_id="esc-nonworst")
        # Record that "no_pii" was flagged in a prior intervention
        session.last_intervention_criteria = {"no_pii"}
        session.intervention_count = 1

        engine = JudgePolicyEngine()

        # Conversation verdict: BLOCK (worst) — no criterion_failures
        conv_verdict = JudgeVerdict(
            scores=[
                JudgeScore(criterion="tone", score=0, max_score=5, reasoning="Hostile"),
            ],
            composite_score=0.1,
            action=VerdictAction.BLOCK,
            summary="Severe tone violation",
            judge_model="test",
            scope=EvaluationScope.CONVERSATION,
            metadata={},
        )

        # Turn verdict: INTERVENE with repeat criterion "no_pii"
        turn_verdict = JudgeVerdict(
            scores=[
                JudgeScore(criterion="no_pii", score=1, max_score=5, reasoning="PII leaked"),
            ],
            composite_score=0.3,
            action=VerdictAction.INTERVENE,
            summary="PII violation",
            judge_model="test",
            scope=EvaluationScope.TURN,
            metadata={"criterion_failures": ["no_pii"]},
        )

        result = engine._build_result([conv_verdict, turn_verdict], session)

        # Decision is BLOCK (from worst verdict), no escalation upgrade happened
        assert result.decision == Decision.BLOCK
        assert result.metadata.get("escalated") is not True
        assert "escalation_reason" not in result.metadata

    def test_escalation_from_non_worst_verdict_upgrades_decision(self):
        """Escalation from a non-worst INTERVENE verdict upgrades to BLOCK.

        Scenario: two INTERVENE verdicts, one has repeat criterion. The
        escalation should upgrade the decision from INTERVENE to BLOCK and
        include escalation metadata.
        """
        from openbias.policy.engines.judge.models import JudgeSessionContext

        session = JudgeSessionContext(session_id="esc-upgrade")
        session.last_intervention_criteria = {"no_pii"}
        session.intervention_count = 1

        engine = JudgePolicyEngine()

        # Conversation verdict: INTERVENE (worst by tie)
        conv_verdict = JudgeVerdict(
            scores=[
                JudgeScore(criterion="tone", score=2, max_score=5, reasoning="Rude"),
            ],
            composite_score=0.4,
            action=VerdictAction.INTERVENE,
            summary="Tone issue",
            judge_model="test",
            scope=EvaluationScope.CONVERSATION,
            metadata={},
        )

        # Turn verdict: INTERVENE with repeat criterion "no_pii"
        turn_verdict = JudgeVerdict(
            scores=[
                JudgeScore(criterion="no_pii", score=1, max_score=5, reasoning="PII leaked"),
            ],
            composite_score=0.3,
            action=VerdictAction.INTERVENE,
            summary="PII violation",
            judge_model="test",
            scope=EvaluationScope.TURN,
            metadata={"criterion_failures": ["no_pii"]},
        )

        result = engine._build_result([conv_verdict, turn_verdict], session)

        # Decision upgraded from INTERVENE to BLOCK via escalation
        assert result.decision == Decision.BLOCK
        assert result.metadata.get("escalated") is True
        assert "repeat" in result.metadata.get("escalation_reason", "")

    def test_no_escalation_when_all_verdicts_pass(self):
        """No escalation when all verdicts are PASS."""
        from openbias.policy.engines.judge.models import JudgeSessionContext

        session = JudgeSessionContext(session_id="all-pass")
        engine = JudgePolicyEngine()

        pass_verdict = JudgeVerdict(
            scores=[
                JudgeScore(criterion="safety", score=5, max_score=5, reasoning="Fine"),
            ],
            composite_score=0.9,
            action=VerdictAction.PASS,
            summary="All good",
            judge_model="test",
            metadata={},
        )

        result = engine._build_result([pass_verdict], session)
        assert result.decision == Decision.ALLOW
        assert result.metadata.get("escalated") is not True

    def test_configurable_max_intervention_attempts(self):
        """Escalation cap uses max_intervention_attempts from config, not hardcoded 3."""
        from openbias.policy.engines.judge.models import JudgeSessionContext

        engine = JudgePolicyEngine()
        engine._max_intervention_attempts = 5  # Higher than default 3

        session = JudgeSessionContext(session_id="custom-cap")
        session.intervention_count = 4  # Would trigger with default cap of 3

        verdict = JudgeVerdict(
            scores=[
                JudgeScore(criterion="quality", score=2, max_score=5, reasoning="Low"),
            ],
            composite_score=0.3,
            action=VerdictAction.INTERVENE,
            summary="Low quality",
            judge_model="test",
            metadata={},
        )

        # pending_count = 4+1 = 5, not > 5, so should NOT escalate
        result = engine._check_escalation(verdict, session)
        assert result["should_escalate"] is False

        # Now set count to 5 → pending = 6 > 5 → should escalate
        session.intervention_count = 5
        result = engine._check_escalation(verdict, session)
        assert result["should_escalate"] is True
        assert "intervention_count_exceeded" in result["reason"]

    async def test_intervention_cap_uses_config_value_end_to_end(
        self, engine, sample_request, sample_response
    ):
        """End-to-end: max_intervention_attempts=2 escalates after 2 interventions."""
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": ["Rule A", "Rule B", "Rule C"],
            "max_intervention_attempts": 2,
        }
        await engine.initialize(config)

        rubric = engine._registry.get("inline_policy")
        criteria_names = [c.name for c in rubric.criteria]

        # 3 different criterion violations (each unique, no repeat escalation)
        for i in range(3):
            scores = []
            for j, name in enumerate(criteria_names):
                scores.append({
                    "criterion": name,
                    "score": 0 if j == i else 1,
                    "reasoning": f"{'Fail' if j == i else 'Pass'}",
                    "evidence": [],
                    "confidence": 0.9,
                })
            response = {"scores": scores, "summary": f"Violation {i+1}."}
            engine._client.call_judge = AsyncMock(return_value=response)
            result = await engine.evaluate_response("s1", sample_response, sample_request)

        # 3rd violation should have triggered the count cap (>2) → BLOCK
        assert result.decision == Decision.BLOCK
        assert result.metadata.get("escalated") is True
        assert "intervention_count_exceeded" in result.metadata.get("escalation_reason", "")


class TestJudgeSessionEviction:
    """Tests for judge engine session TTL and LRU eviction."""

    def test_session_evicted_after_ttl(self, engine):
        """Sessions older than TTL are evicted on next _get_or_create_session."""
        engine._sessions._ttl = 1  # 1 second

        # Create a session and backdate its timestamp
        engine._get_or_create_session("old")
        engine._sessions._timestamps["old"] = time.monotonic() - 2

        # Creating a new session triggers eviction
        engine._get_or_create_session("new")

        assert "old" not in engine._sessions
        assert "new" in engine._sessions

    def test_max_sessions_eviction(self, engine):
        """When max_sessions is exceeded, oldest sessions are evicted."""
        engine._sessions._max_sessions = 2

        engine._get_or_create_session("s1")
        engine._get_or_create_session("s2")
        engine._get_or_create_session("s3")

        assert len(engine._sessions) <= 2
        assert "s3" in engine._sessions

    async def test_reset_session_clears_timestamp(self, engine):
        """reset_session removes the session from the store."""
        engine._get_or_create_session("s1")
        assert "s1" in engine._sessions

        await engine.reset_session("s1")

        assert "s1" not in engine._sessions

    async def test_shutdown_clears_timestamps(self, engine):
        """shutdown clears all sessions."""
        engine._get_or_create_session("s1")
        engine._get_or_create_session("s2")

        await engine.shutdown()

        assert len(engine._sessions) == 0

    async def test_initialize_with_session_config(self):
        """Session TTL and max can be configured via initialize()."""
        engine = JudgePolicyEngine()
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "session_ttl": 300,
            "max_sessions": 500,
        }
        await engine.initialize(config)

        assert engine._sessions._ttl == 300
        assert engine._sessions._max_sessions == 500


class TestMissingCriterionFalsePositive:
    """Tests for fix 1C: synthetic fills should not trigger false positives."""

    async def test_missing_criterion_not_false_positive(self):
        """When the judge omits a binary criterion, the synthetic fill should not
        cause a criterion_failure (false positive). The composite may still
        drop below threshold — the key point is no per-criterion false positive.
        """
        engine = JudgePolicyEngine()
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": ["No financial advice", "Be professional"],
        }
        await engine.initialize(config)

        rubric = engine._registry.get("inline_policy")
        criteria_names = [c.name for c in rubric.criteria]

        # Judge only returns score for the first criterion; second is omitted
        judge_response = {
            "scores": [
                {
                    "criterion": criteria_names[0],
                    "score": 1,
                    "reasoning": "No financial advice given",
                    "evidence": [],
                    "confidence": 0.9,
                },
                # criteria_names[1] intentionally omitted — evaluator fills it
            ],
            "summary": "Good response.",
        }
        engine._client.call_judge = AsyncMock(return_value=judge_response)

        sample_request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o",
        }
        sample_response = {"choices": [{"message": {"content": "Hi there"}}]}

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        # Composite drops below threshold so decision is BLOCK, but the key
        # invariant is that no per-criterion false positive is reported.
        # No criterion_failures should be reported — the synthetic fill with
        # confidence=0.0 must be skipped by _check_criterion_failures
        verdicts = result.metadata.get("judge", {}).get("verdicts", [])
        for v in verdicts:
            assert len(v.get("criterion_failures", [])) == 0, (
                f"Synthetic fill caused false criterion failure: {v}"
            )


class TestRubricIsolation:
    """Tests for fix 2C: engine instances must not share rubric registries."""

    async def test_two_engines_dont_share_rubrics(self):
        """Inline policy registered on one engine must not appear on another."""
        engine_a = JudgePolicyEngine()
        engine_b = JudgePolicyEngine()

        await engine_a.initialize({
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": ["Never share secrets"],
        })
        await engine_b.initialize({
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
        })

        # Engine A has inline_policy rubric
        assert engine_a._registry.get("inline_policy") is not None
        # Engine B should NOT have it
        assert engine_b._registry.get("inline_policy") is None

    async def test_inline_policy_doesnt_corrupt_builtins(self):
        """Registering an inline_policy should not modify built-in rubrics globally."""
        from openbias.policy.engines.judge.rubrics import RubricRegistry

        engine = JudgePolicyEngine()
        await engine.initialize({
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": ["Custom rule"],
        })

        # A fresh registry should not have the inline_policy
        fresh = RubricRegistry()
        assert fresh.get("inline_policy") is None
        # But should still have built-ins
        assert fresh.get("agent_behavior") is not None


class TestInterventionCountBehavior:
    """Tests for intervention_count increment semantics."""

    async def test_block_verdict_does_not_increment_intervention_count(
        self, engine, judge_config, sample_request, sample_response
    ):
        """BLOCK decisions must not increment intervention_count.

        intervention_count is used by _check_escalation to upgrade future
        INTERVENE decisions to BLOCK. If BLOCK verdicts also inflate the
        counter, sessions that already received a block will prematurely
        escalate subsequent INTERVENE decisions.
        """
        from openbias.policy.protocols import EngineResult
        await engine.initialize(judge_config)
        engine._client.call_judge = AsyncMock(return_value=_failing_judge_response())

        # Patch _build_result to always return BLOCK
        original_build = engine._build_result
        engine._build_result = lambda verdicts, session, rubric_name="unknown": EngineResult(
            decision=Decision.BLOCK,
            message="blocked",
            metadata={"judge": {"verdicts": [], "session_turn": 0}, "violations": []},
        )

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result.decision == Decision.BLOCK

        session = engine._sessions.get("s1")
        assert session is not None
        assert session.intervention_count == 0

    async def test_intervene_verdict_increments_intervention_count(
        self, engine, judge_config, sample_request, sample_response
    ):
        """INTERVENE decisions must increment intervention_count."""
        from openbias.policy.protocols import EngineResult
        await engine.initialize(judge_config)
        engine._client.call_judge = AsyncMock(return_value=_failing_judge_response())

        engine._build_result = lambda verdicts, session, rubric_name="unknown": EngineResult(
            decision=Decision.INTERVENE,
            message="intervene",
            metadata={"judge": {"verdicts": [], "session_turn": 0}, "violations": []},
        )

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result.decision == Decision.INTERVENE

        session = engine._sessions.get("s1")
        assert session is not None
        assert session.intervention_count == 1

    async def test_allow_verdict_does_not_increment_intervention_count(
        self, engine, judge_config, sample_request, sample_response
    ):
        """ALLOW decisions must not increment intervention_count."""
        await engine.initialize(judge_config)
        engine._client.call_judge = AsyncMock(return_value=_passing_judge_response())

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result.decision == Decision.ALLOW

        session = engine._sessions.get("s1")
        assert session is not None
        assert session.intervention_count == 0


class TestValidateConfig:
    """Tests for JudgePolicyEngine.validate_config() classmethod."""

    def test_valid_config_no_errors(self):
        errors = JudgePolicyEngine.validate_config({
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
        })
        assert errors == []

    def test_valid_config_with_inline_policy(self):
        errors = JudgePolicyEngine.validate_config({
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": ["Be professional", "No PII"],
        })
        assert errors == []

    def test_no_model_returns_error(self):
        errors = JudgePolicyEngine.validate_config({})
        assert any("No model configured" in e for e in errors)

    def test_empty_model_field_returns_error(self):
        errors = JudgePolicyEngine.validate_config({
            "models": [{"name": "primary"}],
        })
        assert any("missing 'model' field" in e for e in errors)

    def test_nonexistent_default_rubric(self):
        errors = JudgePolicyEngine.validate_config({
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "default_rubric": "does_not_exist",
        })
        assert any("does_not_exist" in e for e in errors)

    def test_validate_config_with_rubric_shorthand(self):
        errors = JudgePolicyEngine.validate_config({
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "rubric": "agent_behavior",
        })
        assert errors == []

    def test_validate_config_with_policies_shorthand(self):
        errors = JudgePolicyEngine.validate_config({
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "policies": ["Be safe"],
        })
        assert errors == []

    def test_invalid_inline_policy_type(self):
        errors = JudgePolicyEngine.validate_config({
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "inline_policy": {"rules": ["bad format"]},
        })
        assert any("Invalid inline policy" in e for e in errors)

    def test_multiple_errors_collected(self):
        errors = JudgePolicyEngine.validate_config({
            "default_rubric": "nonexistent",
        })
        # Should have at least: no model + missing rubric
        assert len(errors) >= 2

