"""
Tests for PolicyEngineChecker adapter.

Verifies that the adapter correctly delegates to the engine's
evaluate_request/evaluate_response based on phase.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openbias.core.interceptor import CheckerMode, CheckPhase
from openbias.core.interceptor.adapters import PolicyEngineChecker
from openbias.policy.protocols import Decision, EngineResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_engine(name: str = "test_engine") -> MagicMock:
    """Create a mock PolicyEngine with a given name."""
    engine = MagicMock()
    engine.name = name
    engine.evaluate_request = AsyncMock()
    engine.evaluate_response = AsyncMock()
    return engine


# ===========================================================================
# Phase routing tests
# ===========================================================================


class TestPhaseRouting:

    async def test_pre_call_calls_evaluate_request(self):
        """PRE_CALL phase calls engine.evaluate_request."""
        engine = _mock_engine()
        engine.evaluate_request.return_value = EngineResult(decision=Decision.ALLOW)

        checker = PolicyEngineChecker(engine=engine, phase=CheckPhase.PRE_CALL)
        request_data = {"messages": [{"role": "user", "content": "hi"}]}
        await checker.evaluate(
            session_id="sess-1",
            request_data=request_data,
            context={"user_request_id": "req-1"},
        )

        engine.evaluate_request.assert_called_once_with(
            "sess-1", request_data, {"user_request_id": "req-1"}
        )
        engine.evaluate_response.assert_not_called()

    async def test_post_call_calls_evaluate_response(self):
        """POST_CALL phase calls engine.evaluate_response."""
        engine = _mock_engine()
        engine.evaluate_response.return_value = EngineResult(decision=Decision.ALLOW)
        response = {"answer": "hello"}
        request_data = {"messages": [{"role": "user", "content": "hi"}]}

        checker = PolicyEngineChecker(engine=engine, phase=CheckPhase.POST_CALL)
        await checker.evaluate(
            session_id="sess-1",
            request_data=request_data,
            response_data=response,
            context={"user_request_id": "req-1"},
        )

        engine.evaluate_response.assert_called_once_with(
            "sess-1", response, request_data, {"user_request_id": "req-1"}
        )
        engine.evaluate_request.assert_not_called()


# ===========================================================================
# Decision passthrough tests
# ===========================================================================


class TestDecisionPassthrough:

    @pytest.mark.parametrize("decision", [Decision.ALLOW, Decision.BLOCK, Decision.INTERVENE])
    async def test_decisions_pass_through(self, decision: Decision):
        """All decisions pass through unchanged."""
        engine = _mock_engine()
        engine.evaluate_request.return_value = EngineResult(
            decision=decision, message="test msg"
        )

        checker = PolicyEngineChecker(engine=engine, phase=CheckPhase.PRE_CALL)
        result = await checker.evaluate(session_id="sess-1", request_data={})

        assert result.decision == decision
        assert result.message == "test msg"


# ===========================================================================
# Naming
# ===========================================================================


class TestNaming:

    async def test_name_includes_engine_and_phase(self):
        """Checker name is {engine.name}_{phase}."""
        engine = _mock_engine(name="my_engine")
        checker = PolicyEngineChecker(engine=engine, phase=CheckPhase.PRE_CALL)

        assert checker.name == "my_engine_pre_call"

    async def test_mode_property(self):
        """Mode property reflects what was passed in."""
        engine = _mock_engine()
        checker = PolicyEngineChecker(
            engine=engine, phase=CheckPhase.PRE_CALL, mode=CheckerMode.ASYNC
        )

        assert checker.mode == CheckerMode.ASYNC
