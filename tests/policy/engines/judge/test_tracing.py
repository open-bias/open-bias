"""
Tests for judge engine OTEL tracing integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from opensentinel.policy.engines.judge.engine import JudgePolicyEngine
from opensentinel.policy.protocols import Decision


@pytest.fixture
def engine():
    return JudgePolicyEngine()


@pytest.fixture
def judge_config():
    return {
        "models": [{"name": "primary", "model": "gpt-4o-mini"}],
    }


@pytest.fixture
def mock_tracer():
    tracer = MagicMock()
    tracer.log_judge_evaluation = MagicMock()
    return tracer


@pytest.fixture
def sample_request():
    return {
        "messages": [
            {"role": "user", "content": "Hello"},
        ],
    }


@pytest.fixture
def sample_response():
    return {
        "choices": [
            {"message": {"content": "Hi there!"}},
        ],
    }


def _passing_response():
    return {
        "scores": [
            {"criterion": "instruction_following", "score": 5, "reasoning": "Good", "evidence": [], "confidence": 0.9},
            {"criterion": "tool_use_safety", "score": 5, "reasoning": "Safe", "evidence": [], "confidence": 0.9},
            {"criterion": "no_hallucination", "score": 5, "reasoning": "OK", "evidence": [], "confidence": 0.9},
            {"criterion": "task_completion", "score": 5, "reasoning": "Done", "evidence": [], "confidence": 0.9},
        ],
        "summary": "Good response",
    }


class TestTracerIntegration:
    def test_set_tracer(self, engine, mock_tracer):
        engine.set_tracer(mock_tracer)
        assert engine._tracer is mock_tracer

    async def test_trace_verdict_called(
        self, engine, judge_config, mock_tracer, sample_request, sample_response,
    ):
        """Tracer should be called after evaluation."""
        await engine.initialize(judge_config)
        engine.set_tracer(mock_tracer)
        engine._client.call_judge = AsyncMock(return_value=_passing_response())

        await engine.evaluate_response("s1", sample_response, sample_request)

        mock_tracer.log_judge_evaluation.assert_called_once()
        call_kwargs = mock_tracer.log_judge_evaluation.call_args[1]
        assert call_kwargs["session_id"] == "s1"
        assert call_kwargs["rubric_name"] == "agent_behavior"
        assert call_kwargs["scope"] == "turn"
        assert call_kwargs["action"] == "pass"
        assert call_kwargs["judge_model"] == "gpt-4o-mini"
        assert isinstance(call_kwargs["scores"], list)
        assert len(call_kwargs["scores"]) == 4

    async def test_no_tracer_no_error(
        self, engine, judge_config, sample_request, sample_response,
    ):
        """Without a tracer, evaluation should still work fine."""
        await engine.initialize(judge_config)
        engine._client.call_judge = AsyncMock(return_value=_passing_response())

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result.decision == Decision.ALLOW

    async def test_tracer_error_does_not_break_eval(
        self, engine, judge_config, mock_tracer, sample_request, sample_response,
    ):
        """Tracer errors should not break evaluation."""
        await engine.initialize(judge_config)
        engine.set_tracer(mock_tracer)
        engine._client.call_judge = AsyncMock(return_value=_passing_response())
        mock_tracer.log_judge_evaluation.side_effect = Exception("Trace failed")

        result = await engine.evaluate_response("s1", sample_response, sample_request)
        assert result.decision == Decision.ALLOW

