"""
Tests for session context in judge prompts.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openbias.policy.engines.judge.models import (
    JudgeSessionContext,
    JudgeVerdict,
    JudgeScore,
    VerdictAction,
    EvaluationScope,
)
from openbias.policy.engines.judge.prompts import format_session_context_block
from openbias.policy.engines.judge import JudgePolicyEngine


class TestFormatSessionContextBlock:
    """Tests for format_session_context_block()."""

    def test_empty_session_returns_empty(self):
        session = JudgeSessionContext(session_id="s1")
        assert format_session_context_block(session) == ""

    def test_none_returns_empty(self):
        assert format_session_context_block(None) == ""

    def test_single_pass_verdict(self):
        session = JudgeSessionContext(session_id="s1")
        session.record_verdict(JudgeVerdict(
            scores=[JudgeScore(criterion="safety", score=1, max_score=1, reasoning="ok")],
            composite_score=1.0,
            action=VerdictAction.PASS,
            summary="Good",
            judge_model="test",
        ))

        result = format_session_context_block(session)
        assert "Turn 1: No violations" in result
        assert "Score trend: 1.0" in result
        assert "Active intervention count: 0" in result

    def test_mixed_verdicts(self):
        session = JudgeSessionContext(session_id="s1")

        # Turn 1: PASS
        session.record_verdict(JudgeVerdict(
            scores=[JudgeScore(criterion="safety", score=1, max_score=1, reasoning="ok")],
            composite_score=0.8,
            action=VerdictAction.PASS,
            summary="Good",
            judge_model="test",
        ))

        # Turn 2: INTERVENE
        session.record_verdict(JudgeVerdict(
            scores=[JudgeScore(criterion="safety", score=0, max_score=1, reasoning="bad")],
            composite_score=0.3,
            action=VerdictAction.INTERVENE,
            summary="Agent used delete without permission",
            judge_model="test",
            metadata={"criterion_failures": ["tool_use_safety"]},
        ))
        session.intervention_count += 1  # engine responsibility

        # Turn 3: INTERVENE (minor)
        session.record_verdict(JudgeVerdict(
            scores=[JudgeScore(criterion="safety", score=1, max_score=1, reasoning="better")],
            composite_score=0.5,
            action=VerdictAction.INTERVENE,
            summary="Minor issue",
            judge_model="test",
            metadata={"criterion_failures": ["tone"]},
        ))
        session.intervention_count += 1  # engine responsibility

        result = format_session_context_block(session)
        assert "Turn 1: No violations" in result
        assert "Turn 2: INTERVENE" in result
        assert "tool_use_safety" in result
        assert 'Intervention applied: "Agent used delete without permission"' in result
        assert "Turn 3: INTERVENE" in result
        assert "Score trend: 0.8 → 0.3 → 0.5" in result
        assert "Active intervention count: 2" in result

    def test_caps_at_10_turns(self):
        session = JudgeSessionContext(session_id="s1")
        for i in range(15):
            session.record_verdict(JudgeVerdict(
                scores=[JudgeScore(criterion="c", score=1, max_score=1, reasoning="ok")],
                composite_score=0.9,
                action=VerdictAction.PASS,
                summary="Good",
                judge_model="test",
            ))

        result = format_session_context_block(session)
        # Should only show turns 6-15 (last 10)
        assert "Turn 6:" in result
        assert "Turn 15:" in result
        assert "Turn 5:" not in result
        # Score trend also capped to last 10
        scores = result.split("Score trend: ")[1].split("\n")[0]
        assert scores.count("→") == 9  # 10 values, 9 arrows


class TestSessionContextInPrompt:
    """Integration: session context flows into judge prompt."""

    async def test_second_evaluation_includes_session_history(self):
        """Run two evaluations on same session, verify second prompt has history."""
        engine = JudgePolicyEngine()
        config = {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "conversation_rubric": None,
        }
        await engine.initialize(config)

        captured_prompts: list[str] = []

        async def capture_judge_call(
            model_name: str, system_prompt: str, user_prompt: str, **kwargs: object
        ) -> dict:
            captured_prompts.append(system_prompt)
            return {
                "scores": [
                    {"criterion": "instruction_following", "score": 2, "reasoning": "ok",
                     "evidence": [], "confidence": 0.9},
                    {"criterion": "tool_use_safety", "score": 5, "reasoning": "ok",
                     "evidence": [], "confidence": 0.9},
                    {"criterion": "no_hallucination", "score": 5, "reasoning": "ok",
                     "evidence": [], "confidence": 0.9},
                    {"criterion": "task_completion", "score": 4, "reasoning": "ok",
                     "evidence": [], "confidence": 0.8},
                ],
                "summary": "Minor issues.",
            }

        engine._client.call_judge = capture_judge_call

        request = {"messages": [{"role": "user", "content": "Hello"}]}
        response = {"choices": [{"message": {"content": "Hi!"}}]}

        # First evaluation — no session history yet
        await engine.evaluate_response("s1", response, request)
        assert len(captured_prompts) == 1
        # First prompt should not have session history
        assert "Prior evaluation history" not in captured_prompts[0]

        # Second evaluation — should include history from first
        await engine.evaluate_response("s1", response, request)
        assert len(captured_prompts) == 2
        assert "Prior evaluation history" in captured_prompts[1]
        assert "Turn 1:" in captured_prompts[1]
