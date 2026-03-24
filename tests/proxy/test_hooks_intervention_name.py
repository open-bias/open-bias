"""
Integration test: async POST_CALL evaluator returns INTERVENE with a message
-> next PRE_CALL applies it -> verify the modifications appear in the result.

Tests the full deferred intervention flow end-to-end through the Interceptor.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from openbias.core.interceptor import Interceptor
from openbias.policy.protocols import Decision, EngineResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION = "integration-session"


def _request(content: str = "hello") -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ],
        "model": "gpt-4",
    }


def _mock_async_engine(
    message: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Mock PolicyEngine that returns INTERVENE on evaluate_response and ALLOW on evaluate_request."""
    engine = MagicMock()
    type(engine).name = PropertyMock(return_value="fake_async_intervention")

    async def _evaluate_request(
        session_id: str,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        return EngineResult(decision=Decision.ALLOW)

    async def _evaluate_response(
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        return EngineResult(
            decision=Decision.INTERVENE,
            message=message,
            metadata=metadata or {},
        )

    engine.evaluate_request = _evaluate_request
    engine.evaluate_response = _evaluate_response
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeferredInterventionIntegration:

    async def test_system_prompt_append_applied(self):
        """Async evaluator returns INTERVENE -> system prompt is appended on next PRE_CALL."""
        engine = _mock_async_engine(
            message="Always verify identity first.",
            metadata={"strategy": "system_prompt_append"},
        )
        interceptor = Interceptor(pre_call_evaluators=[], post_call_evaluators=[engine], default_strategy="system_prompt_append")

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-001")
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request("next question"), "req-002")

        assert result.allowed is True
        assert result.modified_data is not None
        # System prompt should have the intervention appended
        system_msg = result.modified_data["messages"][0]
        assert "Always verify identity first." in system_msg["content"]

    async def test_user_message_inject_applied(self):
        """Async evaluator returns INTERVENE with user_message_inject strategy."""
        engine = _mock_async_engine(
            message="Please verify identity.",
        )
        interceptor = Interceptor(pre_call_evaluators=[], post_call_evaluators=[engine], default_strategy="user_message_inject")

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-001")
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request("next question"), "req-002")

        assert result.allowed is True
        assert result.modified_data is not None
        # Should have an injected user message
        injected_msgs = [
            m for m in result.modified_data["messages"]
            if m["role"] == "user" and "Please verify identity." in m.get("content", "")
        ]
        assert len(injected_msgs) == 1

    async def test_default_strategy_is_user_message_inject(self):
        """Without strategy in metadata, defaults to user_message_inject."""
        engine = _mock_async_engine(message="Be safe.")
        interceptor = Interceptor(pre_call_evaluators=[], post_call_evaluators=[engine])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-001")
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request("next"), "req-002")

        assert result.allowed is True
        assert result.modified_data is not None
        injected_msgs = [
            m for m in result.modified_data["messages"]
            if m["role"] == "user" and "Be safe." in m.get("content", "")
        ]
        assert len(injected_msgs) == 1
