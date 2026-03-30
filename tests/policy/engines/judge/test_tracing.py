"""
Tests for judge engine OTEL tracing integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openbias.policy.engines.judge.engine import JudgePolicyEngine
from openbias.policy.engines.judge.models import VerdictAction
from openbias.policy.protocols import Decision


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

    async def test_parent_span_forwarded(
        self, engine, judge_config, mock_tracer, sample_request, sample_response,
    ):
        """When _parent_span is in context, it is forwarded to log_judge_evaluation."""
        await engine.initialize(judge_config)
        engine.set_tracer(mock_tracer)
        engine._client.call_judge = AsyncMock(return_value=_passing_response())

        mock_span = MagicMock()
        await engine.evaluate_response(
            "s1", sample_response, sample_request, context={"_parent_span": mock_span}
        )

        mock_tracer.log_judge_evaluation.assert_called_once()
        call_kwargs = mock_tracer.log_judge_evaluation.call_args[1]
        assert call_kwargs["parent_span"] is mock_span

    @pytest.mark.parametrize("context", [None, {}, {"unrelated_key": "value"}])
    async def test_parent_span_none_by_default(
        self, engine, judge_config, mock_tracer, sample_request, sample_response, context,
    ):
        """When no _parent_span is in context, parent_span=None is passed to log_judge_evaluation."""
        await engine.initialize(judge_config)
        engine.set_tracer(mock_tracer)
        engine._client.call_judge = AsyncMock(return_value=_passing_response())

        await engine.evaluate_response("s1", sample_response, sample_request, context=context)

        mock_tracer.log_judge_evaluation.assert_called_once()
        call_kwargs = mock_tracer.log_judge_evaluation.call_args[1]
        assert call_kwargs["parent_span"] is None


class TestTraceVerdictFromEvaluateRequest:
    """Test that evaluate_request calls _trace_verdict with correct args."""

    async def test_evaluate_request_traces_verdict(
        self, engine, judge_config, sample_request,
    ):
        """evaluate_request calls _trace_verdict with session_id, verdict, rubric name, and parent_span."""
        await engine.initialize(judge_config)

        # Create a mock verdict
        mock_verdict = MagicMock()
        mock_verdict.action = VerdictAction.PASS
        mock_verdict.scores = []
        mock_verdict.composite_score = 5.0
        mock_verdict.scope = MagicMock()
        mock_verdict.scope.value = "turn"
        mock_verdict.judge_model = "gpt-4o-mini"

        engine._evaluator.evaluate_turn = AsyncMock(return_value=mock_verdict)

        mock_parent_span = MagicMock()
        context = {"_parent_span": mock_parent_span}

        with patch.object(engine, "_trace_verdict") as mock_trace:
            with patch.object(engine, "_build_result", return_value=MagicMock(decision=Decision.ALLOW)):
                await engine.evaluate_request("s1", sample_request, context=context)

            mock_trace.assert_called_once_with(
                "s1",
                mock_verdict,
                "agent_behavior",  # default rubric name
                parent_span=mock_parent_span,
            )

