"""
Integration test: async POST_CALL checker returns INTERVENE with a message
-> next PRE_CALL applies it -> verify the modifications appear in the result.

Tests the full deferred intervention flow end-to-end through the Interceptor.
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock

from opensentinel.core.interceptor import (
    CheckerMode,
    CheckPhase,
    Decision,
    EngineResult,
    Interceptor,
)
from opensentinel.core.interceptor.adapters import PolicyEngineChecker

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


def _mock_async_checker(
    message: str,
    metadata: dict[str, Any] | None = None,
) -> PolicyEngineChecker:
    """Async POST_CALL checker that returns INTERVENE with a message."""
    engine = MagicMock()
    engine.name = "fake_async_intervention"

    async def _evaluate(
        session_id: str,
        request_data: dict[str, Any],
        response_data: Any = None,
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        return EngineResult(
            decision=Decision.INTERVENE,
            message=message,
            metadata=metadata or {},
        )

    checker = PolicyEngineChecker(
        engine=engine, phase=CheckPhase.POST_CALL, mode=CheckerMode.ASYNC
    )
    checker.evaluate = _evaluate  # type: ignore[assignment]
    return checker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeferredInterventionIntegration:

    async def test_system_prompt_append_applied(self):
        """Async checker returns INTERVENE -> system prompt is appended on next PRE_CALL."""
        checker = _mock_async_checker(
            message="Always verify identity first.",
            metadata={"strategy": "system_prompt_append"},
        )
        interceptor = Interceptor([checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-001")
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request("next question"), "req-002")

        assert result.allowed is True
        assert result.modified_data is not None
        # System prompt should have the intervention appended
        system_msg = result.modified_data["messages"][0]
        assert "Always verify identity first." in system_msg["content"]

    async def test_user_message_inject_applied(self):
        """Async checker returns INTERVENE with user_message_inject strategy."""
        checker = _mock_async_checker(
            message="Please verify identity.",
            metadata={"strategy": "user_message_inject"},
        )
        interceptor = Interceptor([checker])

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

    async def test_default_strategy_is_system_prompt_append(self):
        """Without strategy in metadata, defaults to system_prompt_append."""
        checker = _mock_async_checker(message="Be safe.")
        interceptor = Interceptor([checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-001")
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request("next"), "req-002")

        assert result.allowed is True
        assert result.modified_data is not None
        system_msg = result.modified_data["messages"][0]
        assert "Be safe." in system_msg["content"]
