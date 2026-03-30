"""
Comprehensive tests for the Interceptor orchestrator.

Covers:
- Sync PRE_CALL evaluator flow (pass, fail, short-circuit)
- Sync POST_CALL evaluator flow (pass, fail)
- Async evaluator lifecycle (fire, collect, cross-request handoff)
- Async edge cases (task failure, still-running, cleanup, shutdown)
- Interceptor init categorization
- Session TTL and LRU eviction
- Context passing to evaluators
- fail_action upgrade logic
- max_intervention_attempts escalation
- Separate pre_call / post_call evaluator lists
- span_factory integration (sync, async applied, async dispatched)
"""

import asyncio
import time
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from openbias.core.interceptor import (
    Decision,
    EngineResult,
    Interceptor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION = "test-session"
REQUEST_ID = "req-001"


def _request(content: str = "hello") -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": content}], "model": "gpt-4"}


def _mock_engine(
    *,
    name: str = "fake",
    decision: Decision = Decision.ALLOW,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
    modified_messages: list[dict[str, Any]] | None = None,
    delay: float = 0,
    raise_on_evaluate: Exception | None = None,
) -> MagicMock:
    """Create a mock PolicyEngine with configurable behavior."""
    engine = MagicMock()
    engine.name = name

    async def _evaluate_request(
        session_id: str,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        if delay > 0:
            await asyncio.sleep(delay)
        if raise_on_evaluate:
            raise raise_on_evaluate
        return EngineResult(
            decision=decision,
            message=message,
            metadata=metadata or {},
            modified_messages=modified_messages,
        )

    async def _evaluate_response(
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        if delay > 0:
            await asyncio.sleep(delay)
        if raise_on_evaluate:
            raise raise_on_evaluate
        return EngineResult(
            decision=decision,
            message=message,
            metadata=metadata or {},
            modified_messages=modified_messages,
        )

    engine.evaluate_request = AsyncMock(side_effect=_evaluate_request)
    engine.evaluate_response = AsyncMock(side_effect=_evaluate_response)
    return engine


# ===========================================================================
# Sync PRE_CALL tests
# ===========================================================================


class TestSyncPreCall:

    async def test_pass_unchanged(self):
        """Single ALLOW evaluator — request goes through unchanged."""
        evaluator = _mock_engine()
        interceptor = Interceptor(pre_call_evaluators=[evaluator], post_call_evaluators=[])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is None

    async def test_block_blocks(self):
        """BLOCK evaluator blocks the request."""
        evaluator = _mock_engine(
            decision=Decision.BLOCK,
            message="forbidden",
        )
        interceptor = Interceptor(pre_call_evaluators=[evaluator], post_call_evaluators=[])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is False
        assert result.message == "forbidden"

    async def test_block_short_circuits(self):
        """First evaluator BLOCKs — second evaluator never runs."""
        e1 = _mock_engine(
            name="blocker",
            decision=Decision.BLOCK,
        )
        call_count = 0

        async def counting_evaluate_request(
            session_id: str,
            request_data: dict[str, Any],
            context: dict[str, Any] | None = None,
        ) -> EngineResult:
            nonlocal call_count
            call_count += 1
            return EngineResult(decision=Decision.ALLOW)

        e2 = _mock_engine(name="skipped")
        e2.evaluate_request = AsyncMock(side_effect=counting_evaluate_request)
        interceptor = Interceptor(pre_call_evaluators=[e1, e2], post_call_evaluators=[])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is False
        assert call_count == 0  # Never reached

    async def test_engine_exception_fails_open(self):
        """Exception in a sync evaluator fails open (ALLOW)."""
        evaluator = _mock_engine(
            raise_on_evaluate=RuntimeError("kaboom"),
        )
        interceptor = Interceptor(pre_call_evaluators=[evaluator], post_call_evaluators=[])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        # Fail-open: request is allowed
        assert result.allowed is True

    async def test_intervene_injects_user_message_by_default(self):
        """INTERVENE evaluator injects a user message by default."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="Stay on topic",
        )
        interceptor = Interceptor(pre_call_evaluators=[evaluator], post_call_evaluators=[])
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        contents = [m["content"] for m in result.modified_data["messages"]]
        assert any("Stay on topic" in c for c in contents)

    async def test_intervene_system_prompt_append_strategy(self):
        """INTERVENE with system_prompt_append strategy appends to system prompt."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="Verify identity first",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[evaluator],
            post_call_evaluators=[],
            default_strategy="system_prompt_append",
        )
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        system_msg = result.modified_data["messages"][0]
        assert system_msg["role"] == "system"
        assert "Verify identity first" in system_msg["content"]

    async def test_intervene_user_message_inject_strategy(self):
        """INTERVENE with user_message_inject strategy injects a user message."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="Verify identity first",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[evaluator],
            post_call_evaluators=[],
            default_strategy="user_message_inject",
        )
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        # Should have a user guidance message injected
        contents = [m["content"] for m in result.modified_data["messages"]]
        assert any("Verify identity first" in c for c in contents)

    async def test_intervene_without_message_no_modification(self):
        """INTERVENE with no message — no modification applied."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message=None,
        )
        interceptor = Interceptor(pre_call_evaluators=[evaluator], post_call_evaluators=[])
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is None

    async def test_intervene_modified_messages_replaces_messages(self):
        """INTERVENE with modified_messages replaces request messages directly."""
        sanitized = [{"role": "user", "content": "sanitized hello"}]
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            modified_messages=sanitized,
        )
        interceptor = Interceptor(pre_call_evaluators=[evaluator], post_call_evaluators=[])
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        assert result.modified_data["messages"] == sanitized

    async def test_intervene_modified_messages_takes_precedence_over_message(self):
        """modified_messages takes precedence over message text."""
        sanitized = [{"role": "user", "content": "sanitized"}]
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="This should be ignored",
            modified_messages=sanitized,
        )
        interceptor = Interceptor(pre_call_evaluators=[evaluator], post_call_evaluators=[])
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        assert result.modified_data["messages"] == sanitized

    async def test_intervene_response_modification_strategy_does_not_modify_request(self):
        """INTERVENE with response_modification strategy leaves request unmodified during PRE_CALL.

        response_modification is a response-time strategy and must not be applied
        at request-modification time.
        """
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="Some guidance",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[evaluator],
            post_call_evaluators=[],
            default_strategy="response_modification",
        )
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        # response_modification is response-time only; request must be returned unmodified
        assert result.modified_data is None

    async def test_intervene_does_not_mutate_original_request_data(self):
        """Intervention must not mutate the caller's original request_data dict."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="Stay on topic",
        )
        interceptor = Interceptor(pre_call_evaluators=[evaluator], post_call_evaluators=[])
        original_messages = [{"role": "user", "content": "hello"}]
        req: dict[str, Any] = {"messages": original_messages, "model": "gpt-4"}
        original_content = req["messages"][0]["content"]

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        # Caller's original dict must be untouched
        assert req["messages"] is original_messages
        assert req["messages"][0]["content"] == original_content


# ===========================================================================
# Sync POST_CALL tests
# ===========================================================================


class TestSyncPostCall:

    async def test_pass_unchanged(self):
        """ALLOW evaluator — response goes through unchanged."""
        evaluator = _mock_engine()
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[evaluator], mode="sync"
        )
        req = _request()

        result = await interceptor.run_post_call(SESSION, req, {"answer": "hi"}, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is None

    async def test_block_blocks(self):
        """BLOCK evaluator — response is blocked."""
        evaluator = _mock_engine(
            decision=Decision.BLOCK,
            message="toxic content",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[evaluator], mode="sync"
        )

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is False
        assert result.message == "toxic content"

    async def test_intervene_returns_intervention_data(self):
        """INTERVENE evaluator — modified_data contains intervention info."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="Dangerous tool call detected",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[evaluator], mode="sync"
        )

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is True
        assert result.modified_data is not None
        interventions = result.modified_data["_interventions"]
        assert len(interventions) == 1
        assert interventions[0]["message"] == "Dangerous tool call detected"

    async def test_intervene_with_modified_messages(self):
        """INTERVENE with modified_messages passes them through."""
        sanitized = [{"role": "assistant", "content": "I cannot do that."}]
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="Replaced dangerous response",
            modified_messages=sanitized,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[evaluator], mode="sync"
        )

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is True
        assert result.modified_data is not None
        interventions = result.modified_data["_interventions"]
        assert interventions[0]["modified_messages"] == sanitized

    async def test_multiple_intervene_evaluators_accumulate(self):
        """Multiple INTERVENE evaluators — all interventions are collected."""
        e1 = _mock_engine(
            name="evaluator1",
            decision=Decision.INTERVENE,
            message="Issue 1",
        )
        e2 = _mock_engine(
            name="evaluator2",
            decision=Decision.INTERVENE,
            message="Issue 2",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[e1, e2], mode="sync"
        )

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is True
        interventions = result.modified_data["_interventions"]
        assert len(interventions) == 2

    async def test_engine_exception_fails_open(self):
        """Exception in sync POST_CALL evaluator fails open."""
        evaluator = _mock_engine(
            raise_on_evaluate=RuntimeError("kaboom"),
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[evaluator], mode="sync"
        )

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "hi"}, REQUEST_ID
        )

        assert result.allowed is True


# ===========================================================================
# Async evaluator lifecycle tests
# ===========================================================================


class TestAsyncEvaluatorLifecycle:

    async def test_async_evaluator_started_during_post_call(self):
        """After run_post_call, async task is stored in _running_tasks."""
        async_evaluator = _mock_engine(
            name="async_e",
            delay=0.5,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[async_evaluator]
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        assert SESSION in interceptor._sessions
        assert len(interceptor._sessions.get(SESSION)) == 1
        await interceptor.shutdown()

    async def test_cross_request_handoff(self):
        """
        Full lifecycle: async evaluator runs during POST_CALL of req 1,
        its result is collected during PRE_CALL of req 2.
        """
        async_evaluator = _mock_engine(
            name="async_monitor",
            decision=Decision.ALLOW,
            delay=0.01,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[async_evaluator]
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True

    async def test_async_intervene_modifies_next_request(self):
        """Async evaluator returns INTERVENE — next PRE_CALL applies intervention."""
        async_evaluator = _mock_engine(
            name="async_guide",
            decision=Decision.INTERVENE,
            message="Remember the workflow",
            delay=0.01,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[async_evaluator]
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True
        assert result.modified_data is not None
        # Guidance is injected AFTER the last user message
        guidance_msg = result.modified_data["messages"][1]
        assert "Remember the workflow" in guidance_msg["content"]

    async def test_async_block_blocks_next_request(self):
        """Async evaluator returns BLOCK — next PRE_CALL blocks."""
        async_evaluator = _mock_engine(
            name="async_blocker",
            decision=Decision.BLOCK,
            message="violation detected async",
            delay=0.01,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[async_evaluator]
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is False
        assert "violation detected async" in (result.message or "")

    async def test_early_block_preserves_unprocessed_async_results(self):
        """
        When a BLOCK result causes an early return, async results that completed
        but had not yet been iterated must remain in the session and be picked up
        on the next request.

        Setup: three async evaluators complete — A=ALLOW, B=BLOCK, C=INTERVENE.
        On request 2: B triggers the early BLOCK return. C's result must not be
        discarded; it should be applied on request 3.

        Note: evaluate_request (sync pre-call) always returns ALLOW so that only
        the async post-call results (evaluate_response) drive the blocking behaviour.
        This isolates the async result-preservation logic from sync pre-call gating.
        """
        # Engine A: ALLOW in both phases
        engine_a = _mock_engine(name="engine_a", decision=Decision.ALLOW, delay=0.01)

        # Engine B: ALLOW at sync pre-call, BLOCK at async post-call
        engine_b = _mock_engine(name="engine_b", decision=Decision.ALLOW, delay=0.01)

        async def _b_response(
            session_id: str,
            response_data: Any,
            request_data: dict[str, Any],
            context: dict[str, Any] | None = None,
        ) -> EngineResult:
            await asyncio.sleep(0.01)
            return EngineResult(decision=Decision.BLOCK, message="blocked by B")

        engine_b.evaluate_response = AsyncMock(side_effect=_b_response)

        # Engine C: ALLOW at sync pre-call, INTERVENE at async post-call
        engine_c = _mock_engine(name="engine_c", decision=Decision.ALLOW, delay=0.01)

        async def _c_response(
            session_id: str,
            response_data: Any,
            request_data: dict[str, Any],
            context: dict[str, Any] | None = None,
        ) -> EngineResult:
            await asyncio.sleep(0.01)
            return EngineResult(decision=Decision.INTERVENE, message="intervention from C")

        engine_c.evaluate_response = AsyncMock(side_effect=_c_response)

        interceptor = Interceptor(
            pre_call_evaluators=[],
            post_call_evaluators=[engine_a, engine_b, engine_c],
        )

        # Request 1: fire all three async evaluators
        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        # Wait for all three to complete
        await asyncio.sleep(0.1)

        # Request 2: A=ALLOW processed, B=BLOCK causes early return.
        # C's result must remain in session.
        result2 = await interceptor.run_pre_call(SESSION, _request(), "req-002")
        assert result2.allowed is False
        assert "blocked by B" in (result2.message or "")

        # Request 3: C's INTERVENE result should be applied.
        result3 = await interceptor.run_pre_call(SESSION, _request(), "req-003")
        assert result3.allowed is True
        assert result3.modified_data is not None
        contents = [m["content"] for m in result3.modified_data["messages"]]
        assert any("intervention from C" in c for c in contents)


# ===========================================================================
# Async edge cases
# ===========================================================================


class TestAsyncEdgeCases:

    async def test_async_exception_fails_open(self):
        """Async evaluator that raises — fails open on next collection."""
        async_evaluator = _mock_engine(
            name="async_crasher",
            raise_on_evaluate=RuntimeError("async boom"),
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[async_evaluator]
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        # Fail-open: request is allowed despite async error
        assert result.allowed is True

    async def test_still_running_not_collected(self):
        """Async task that isn't done yet stays in _running_tasks."""
        slow_evaluator = _mock_engine(
            name="slow_async",
            delay=5.0,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[slow_evaluator]
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True
        assert SESSION in interceptor._sessions
        assert len(interceptor._sessions.get(SESSION)) == 1

        await interceptor.shutdown()

    async def test_cleanup_session_cancels_tasks(self):
        """cleanup_session cancels running tasks and clears pending results."""
        slow_evaluator = _mock_engine(
            name="cleanup_target",
            delay=5.0,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[slow_evaluator]
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        assert SESSION in interceptor._sessions

        await interceptor.cleanup_session(SESSION)

        assert SESSION not in interceptor._sessions

    async def test_shutdown_cleans_all_sessions(self):
        """shutdown cancels tasks across all sessions."""
        slow_evaluator = _mock_engine(
            delay=5.0,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[slow_evaluator]
        )

        await interceptor.run_post_call("session-a", _request(), {"r": 1}, REQUEST_ID)
        await interceptor.run_post_call("session-b", _request(), {"r": 2}, REQUEST_ID)

        assert len(interceptor._sessions) == 2

        await interceptor.shutdown()

        assert len(interceptor._sessions) == 0

    async def test_no_pending_async_on_first_request(self):
        """First PRE_CALL with no prior async results works cleanly."""
        interceptor = Interceptor(pre_call_evaluators=[], post_call_evaluators=[])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True

    async def test_cancelled_async_task_fails_open(self):
        """Cancelled async task is handled gracefully — next PRE_CALL is allowed."""
        slow_evaluator = _mock_engine(
            name="cancellable",
            delay=5.0,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[slow_evaluator]
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        # Manually cancel the task to simulate eviction/cap behaviour
        tasks = interceptor._sessions.get(SESSION) or []
        assert len(tasks) == 1
        tasks[0].cancel()
        # Allow the cancellation to propagate
        await asyncio.sleep(0)

        # Next PRE_CALL should collect the cancelled task and fail-open
        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True


# ===========================================================================
# Interceptor init categorization
# ===========================================================================


class TestInterceptorInit:

    async def test_categorizes_evaluators_sync_post_call(self):
        """With mode='sync': post_call_evaluators go to _sync_post_call_evaluators."""
        evaluator = _mock_engine(name="e1")

        interceptor = Interceptor(
            pre_call_evaluators=[evaluator],
            post_call_evaluators=[evaluator],
            mode="sync",
        )

        assert len(interceptor._sync_pre_call_evaluators) == 1
        assert evaluator in interceptor._sync_pre_call_evaluators
        assert len(interceptor._sync_post_call_evaluators) == 1
        assert evaluator in interceptor._sync_post_call_evaluators
        assert len(interceptor._async_post_call_evaluators) == 0

    async def test_categorizes_evaluators_async_post_call(self):
        """With mode='async': post_call_evaluators go to _async_post_call_evaluators."""
        evaluator = _mock_engine(name="e1")

        interceptor = Interceptor(
            pre_call_evaluators=[evaluator],
            post_call_evaluators=[evaluator],
            mode="async",
        )

        assert len(interceptor._sync_pre_call_evaluators) == 1
        assert evaluator in interceptor._sync_pre_call_evaluators
        assert len(interceptor._async_post_call_evaluators) == 1
        assert evaluator in interceptor._async_post_call_evaluators
        assert len(interceptor._sync_post_call_evaluators) == 0

    async def test_empty_evaluator_lists(self):
        """Interceptor with no evaluators still works."""
        interceptor = Interceptor(pre_call_evaluators=[], post_call_evaluators=[])

        pre = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)
        post = await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        assert pre.allowed is True
        assert post.allowed is True

    def test_invalid_default_strategy_raises(self) -> None:
        """Unknown default_strategy should raise ValueError at init time."""
        with pytest.raises(ValueError, match="hard_blok"):
            Interceptor(
                pre_call_evaluators=[], post_call_evaluators=[], default_strategy="hard_blok"
            )

    @pytest.mark.parametrize(
        "strategy",
        ["system_prompt_append", "user_message_inject", "response_modification"],
    )
    def test_valid_default_strategies_accepted(self, strategy: str) -> None:
        """All StrategyType values should be accepted without error."""
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[], default_strategy=strategy
        )
        assert interceptor._default_strategy == strategy


# ===========================================================================
# Session TTL and LRU eviction tests
# ===========================================================================


class TestSessionEviction:

    async def test_session_evicted_after_ttl(self):
        """Sessions older than TTL are cleaned up on next run_pre_call."""
        slow_evaluator = _mock_engine(
            name="async_ttl",
            delay=0.01,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[],
            post_call_evaluators=[slow_evaluator],
            session_ttl=1,
        )

        await interceptor.run_post_call("old-session", _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        # Backdate the timestamp to simulate TTL expiry
        interceptor._sessions._timestamps["old-session"] = time.monotonic() - 2

        # Next pre_call should evict the stale session
        await interceptor.run_pre_call("new-session", _request(), REQUEST_ID)

        assert "old-session" not in interceptor._sessions

    async def test_max_sessions_eviction(self):
        """When max_sessions is exceeded, oldest sessions are evicted."""
        async_evaluator = _mock_engine(
            name="evict_cap",
            delay=0.01,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[],
            post_call_evaluators=[async_evaluator],
            max_sessions=2,
        )

        await interceptor.run_post_call("session-1", _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)
        await interceptor.run_post_call("session-2", _request(), {"r": 2}, REQUEST_ID)
        await asyncio.sleep(0.05)
        await interceptor.run_post_call("session-3", _request(), {"r": 3}, REQUEST_ID)
        await asyncio.sleep(0.05)

        # Hard cap is 2 — oldest session should have been evicted
        assert "session-1" not in interceptor._sessions
        assert "session-2" in interceptor._sessions
        assert "session-3" in interceptor._sessions

        await interceptor.shutdown()

    async def test_cleanup_session_removes_timestamp(self):
        """cleanup_session removes both tasks and timestamp."""
        slow_evaluator = _mock_engine(
            name="cleanup_ts",
            delay=5.0,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[slow_evaluator]
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        assert SESSION in interceptor._sessions

        await interceptor.cleanup_session(SESSION)

        assert SESSION not in interceptor._sessions


# ===========================================================================
# Context passing to evaluators
# ===========================================================================


class TestAsyncContextPassing:

    async def test_async_post_call_receives_context(self):
        """Async POST_CALL evaluators receive context with user_request_id."""
        received_context: dict[str, Any] = {}

        async def _evaluate_response(
            session_id: str,
            response_data: Any,
            request_data: dict[str, Any],
            context: dict[str, Any] | None = None,
        ) -> EngineResult:
            received_context.update(context or {})
            return EngineResult(decision=Decision.ALLOW)

        evaluator = _mock_engine(name="ctx_engine")
        evaluator.evaluate_response = AsyncMock(side_effect=_evaluate_response)

        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[evaluator]
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-ctx-001")
        await asyncio.sleep(0.05)

        assert received_context.get("user_request_id") == "req-ctx-001"

    async def test_async_post_call_context_has_suppress_trace(self):
        """Async POST_CALL evaluators receive _suppress_trace=True and no _parent_span."""
        received_context: dict[str, Any] = {}

        async def _evaluate_response(
            session_id: str,
            response_data: Any,
            request_data: dict[str, Any],
            context: dict[str, Any] | None = None,
        ) -> EngineResult:
            received_context.update(context or {})
            return EngineResult(decision=Decision.ALLOW)

        evaluator = _mock_engine(name="ctx_suppress")
        evaluator.evaluate_response = AsyncMock(side_effect=_evaluate_response)

        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[evaluator]
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-ctx-003")
        await asyncio.sleep(0.05)

        assert received_context.get("_suppress_trace") is True
        assert "_parent_span" not in received_context

    async def test_sync_pre_call_receives_context(self):
        """Sync PRE_CALL evaluators receive context with user_request_id."""
        received_context: dict[str, Any] = {}

        async def _evaluate_request(
            session_id: str,
            request_data: dict[str, Any],
            context: dict[str, Any] | None = None,
        ) -> EngineResult:
            received_context.update(context or {})
            return EngineResult(decision=Decision.ALLOW)

        evaluator = _mock_engine(name="ctx_pre")
        evaluator.evaluate_request = AsyncMock(side_effect=_evaluate_request)

        interceptor = Interceptor(
            pre_call_evaluators=[evaluator], post_call_evaluators=[]
        )

        await interceptor.run_pre_call(SESSION, _request(), "req-ctx-002")

        assert received_context.get("user_request_id") == "req-ctx-002"


# ===========================================================================
# fail_action upgrade logic
# ===========================================================================


class TestFailAction:

    async def test_default_fail_action_is_intervene(self):
        """Default fail_action is 'intervene' — INTERVENE stays INTERVENE."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="Stay on topic",
        )
        interceptor = Interceptor(pre_call_evaluators=[evaluator], post_call_evaluators=[])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None

    async def test_fail_action_block_upgrades_intervene_pre_call(self):
        """fail_action='block' upgrades INTERVENE to BLOCK on PRE_CALL."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="violation",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[evaluator], post_call_evaluators=[], fail_action="block"
        )

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is False
        assert result.message == "violation"

    async def test_fail_action_block_upgrades_intervene_post_call(self):
        """fail_action='block' upgrades INTERVENE to BLOCK on POST_CALL."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="violation",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[],
            post_call_evaluators=[evaluator],
            mode="sync",
            fail_action="block",
        )

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is False
        assert result.message == "violation"

    async def test_fail_action_block_does_not_affect_allow(self):
        """fail_action='block' does not upgrade ALLOW decisions."""
        evaluator = _mock_engine(
            decision=Decision.ALLOW,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[evaluator], post_call_evaluators=[], fail_action="block"
        )

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True

    async def test_fail_action_block_upgrades_async_intervene(self):
        """fail_action='block' upgrades async INTERVENE to BLOCK on next request."""
        async_evaluator = _mock_engine(
            name="async_guide",
            decision=Decision.INTERVENE,
            message="async violation",
            delay=0.01,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[],
            post_call_evaluators=[async_evaluator],
            fail_action="block",
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is False
        assert "async violation" in (result.message or "")

    async def test_fail_action_shadow_downgrades_block_pre_call(self):
        """fail_action='shadow' downgrades BLOCK to ALLOW on PRE_CALL."""
        evaluator = _mock_engine(
            decision=Decision.BLOCK,
            message="blocked",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[evaluator], post_call_evaluators=[], fail_action="shadow"
        )

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True
        assert result.message is None

    async def test_fail_action_shadow_downgrades_intervene_pre_call(self):
        """fail_action='shadow' downgrades INTERVENE to ALLOW — no modifications."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="Stay on topic",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[evaluator], post_call_evaluators=[], fail_action="shadow"
        )

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is None

    async def test_fail_action_shadow_downgrades_block_post_call(self):
        """fail_action='shadow' downgrades BLOCK to ALLOW on POST_CALL."""
        evaluator = _mock_engine(
            decision=Decision.BLOCK,
            message="blocked",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[],
            post_call_evaluators=[evaluator],
            mode="sync",
            fail_action="shadow",
        )

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is True
        assert result.message is None

    async def test_fail_action_shadow_downgrades_async_intervene(self):
        """fail_action='shadow' downgrades async INTERVENE — no modification on next request."""
        async_evaluator = _mock_engine(
            name="async_guide",
            decision=Decision.INTERVENE,
            message="async violation",
            delay=0.01,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[],
            post_call_evaluators=[async_evaluator],
            fail_action="shadow",
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True
        assert result.modified_data is None


# ===========================================================================
# max_intervention_attempts escalation tests
# ===========================================================================


class TestMaxInterventionAttempts:

    async def test_escalation_after_max_attempts(self):
        """After max_intervention_attempts INTERVENEs, next INTERVENE becomes BLOCK."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="guidance",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[evaluator],
            post_call_evaluators=[],
            max_intervention_attempts=2,
        )

        # First INTERVENE — allowed (count=1)
        result1 = await interceptor.run_pre_call(SESSION, _request(), "req-001")
        assert result1.allowed is True
        assert result1.modified_data is not None

        # Second INTERVENE — allowed (count=2, at limit)
        result2 = await interceptor.run_pre_call(SESSION, _request(), "req-002")
        assert result2.allowed is True
        assert result2.modified_data is not None

        # Third INTERVENE — escalated to BLOCK (count=3, exceeds limit)
        result3 = await interceptor.run_pre_call(SESSION, _request(), "req-003")
        assert result3.allowed is False
        assert result3.message == "guidance"

    async def test_escalation_default_max_is_three(self):
        """Default max_intervention_attempts is 3."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="guidance",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[evaluator], post_call_evaluators=[]
        )

        # First three should be allowed
        for i in range(3):
            result = await interceptor.run_pre_call(SESSION, _request(), f"req-{i}")
            assert result.allowed is True

        # Fourth should be blocked
        result = await interceptor.run_pre_call(SESSION, _request(), "req-4")
        assert result.allowed is False

    async def test_escalation_per_session(self):
        """Intervention count is tracked per session independently."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="guidance",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[evaluator],
            post_call_evaluators=[],
            max_intervention_attempts=1,
        )

        # Session A: first INTERVENE allowed
        result_a1 = await interceptor.run_pre_call("session-a", _request(), "req-a1")
        assert result_a1.allowed is True

        # Session B: first INTERVENE allowed (independent counter)
        result_b1 = await interceptor.run_pre_call("session-b", _request(), "req-b1")
        assert result_b1.allowed is True

        # Session A: second INTERVENE escalated to BLOCK
        result_a2 = await interceptor.run_pre_call("session-a", _request(), "req-a2")
        assert result_a2.allowed is False

        # Session B: second INTERVENE also escalated (its own count)
        result_b2 = await interceptor.run_pre_call("session-b", _request(), "req-b2")
        assert result_b2.allowed is False

    async def test_cleanup_session_resets_intervention_count(self):
        """Cleaning up a session resets its intervention count."""
        evaluator = _mock_engine(
            decision=Decision.INTERVENE,
            message="guidance",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[evaluator],
            post_call_evaluators=[],
            max_intervention_attempts=1,
        )

        # Use up the intervention allowance
        result1 = await interceptor.run_pre_call(SESSION, _request(), "req-001")
        assert result1.allowed is True

        # Escalated
        result2 = await interceptor.run_pre_call(SESSION, _request(), "req-002")
        assert result2.allowed is False

        # Cleanup resets the counter
        await interceptor.cleanup_session(SESSION)

        # Should be allowed again
        result3 = await interceptor.run_pre_call(SESSION, _request(), "req-003")
        assert result3.allowed is True

    async def test_allow_decisions_do_not_increment_counter(self):
        """ALLOW decisions should not affect the intervention counter."""
        allow_eval = _mock_engine(name="allower", decision=Decision.ALLOW)
        intervene_eval = _mock_engine(
            name="intervener",
            decision=Decision.INTERVENE,
            message="guidance",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[allow_eval],
            post_call_evaluators=[],
            max_intervention_attempts=1,
        )

        # ALLOW does not increment counter
        await interceptor.run_pre_call(SESSION, _request(), "req-001")
        await interceptor.run_pre_call(SESSION, _request(), "req-002")

        # Now switch to an intervening evaluator
        interceptor._sync_pre_call_evaluators = [intervene_eval]

        # First INTERVENE should still be allowed
        result = await interceptor.run_pre_call(SESSION, _request(), "req-003")
        assert result.allowed is True

    async def test_escalation_with_async_intervene(self):
        """Async INTERVENE results also count toward max_intervention_attempts."""
        async_eval = _mock_engine(
            name="async_guide",
            decision=Decision.INTERVENE,
            message="async guidance",
            delay=0.01,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[],
            post_call_evaluators=[async_eval],
            max_intervention_attempts=1,
        )

        # First async INTERVENE
        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-001")
        await asyncio.sleep(0.05)
        result1 = await interceptor.run_pre_call(SESSION, _request(), "req-002")
        assert result1.allowed is True
        assert result1.modified_data is not None

        # Second async INTERVENE — should escalate to BLOCK
        await interceptor.run_post_call(SESSION, _request(), {"r": 2}, "req-003")
        await asyncio.sleep(0.05)
        result2 = await interceptor.run_pre_call(SESSION, _request(), "req-004")
        assert result2.allowed is False


# ===========================================================================
# Separate pre_call / post_call evaluator lists
# ===========================================================================


class TestSeparateEvaluatorLists:

    async def test_empty_pre_call_with_populated_post_call(self):
        """Empty pre_call list with populated post_call works correctly."""
        post_eval = _mock_engine(name="post_only", decision=Decision.ALLOW)
        interceptor = Interceptor(
            pre_call_evaluators=[],
            post_call_evaluators=[post_eval],
            mode="sync",
        )

        # Pre-call: no evaluators, should pass through
        pre_result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)
        assert pre_result.allowed is True
        assert pre_result.metadata["results"] == []

        # Post-call: evaluator runs
        post_result = await interceptor.run_post_call(
            SESSION, _request(), {"r": 1}, REQUEST_ID
        )
        assert post_result.allowed is True
        assert len(post_result.metadata["results"]) == 1
        assert post_result.metadata["results"][0]["evaluator"] == "post_only"

    async def test_populated_pre_call_with_empty_post_call(self):
        """Populated pre_call with empty post_call works correctly."""
        pre_eval = _mock_engine(name="pre_only", decision=Decision.ALLOW)
        interceptor = Interceptor(
            pre_call_evaluators=[pre_eval],
            post_call_evaluators=[],
            mode="sync",
        )

        # Pre-call: evaluator runs
        pre_result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)
        assert pre_result.allowed is True
        assert len(pre_result.metadata["results"]) == 1
        assert pre_result.metadata["results"][0]["evaluator"] == "pre_only"

        # Post-call: no evaluators
        post_result = await interceptor.run_post_call(
            SESSION, _request(), {"r": 1}, REQUEST_ID
        )
        assert post_result.allowed is True
        assert post_result.metadata["results"] == []

    async def test_different_evaluators_for_each_phase(self):
        """Different evaluators can be assigned to pre_call and post_call."""
        pre_eval = _mock_engine(name="pre_evaluator", decision=Decision.ALLOW)
        post_eval = _mock_engine(
            name="post_evaluator",
            decision=Decision.INTERVENE,
            message="post-call issue",
        )
        interceptor = Interceptor(
            pre_call_evaluators=[pre_eval],
            post_call_evaluators=[post_eval],
            mode="sync",
        )

        # Pre-call uses only pre_eval
        pre_result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)
        assert pre_result.allowed is True
        assert len(pre_result.metadata["results"]) == 1
        assert pre_result.metadata["results"][0]["evaluator"] == "pre_evaluator"

        # Post-call uses only post_eval
        post_result = await interceptor.run_post_call(
            SESSION, _request(), {"r": 1}, REQUEST_ID
        )
        assert post_result.allowed is True
        assert post_result.modified_data is not None
        interventions = post_result.modified_data["_interventions"]
        assert len(interventions) == 1
        assert interventions[0]["evaluator"] == "post_evaluator"

    async def test_pre_call_evaluator_not_used_in_post_call(self):
        """Pre-call evaluator's evaluate_response is never called during post_call."""
        pre_eval = _mock_engine(name="pre_only", decision=Decision.BLOCK, message="should not fire")
        interceptor = Interceptor(
            pre_call_evaluators=[pre_eval],
            post_call_evaluators=[],
            mode="sync",
        )

        # Post-call should not use the pre_call evaluator
        post_result = await interceptor.run_post_call(
            SESSION, _request(), {"r": 1}, REQUEST_ID
        )
        assert post_result.allowed is True
        pre_eval.evaluate_response.assert_not_called()


# ===========================================================================
# span_factory tests
# ===========================================================================


def _mock_span_factory():
    """Create a mock span factory that records calls and yields mock spans."""
    calls: list[tuple[str, str]] = []
    spans: list[MagicMock] = []

    @contextmanager
    def factory(evaluator_name, phase):
        span = MagicMock()
        calls.append((evaluator_name, phase))
        spans.append(span)
        yield span

    factory.calls = calls  # type: ignore[attr-defined]
    factory.spans = spans  # type: ignore[attr-defined]
    return factory



class TestSpanFactory:

    async def test_span_factory_called_per_pre_call_evaluator(self):
        """span_factory is called once per sync pre-call evaluator."""
        e1 = _mock_engine(name="eval_a")
        e2 = _mock_engine(name="eval_b")
        interceptor = Interceptor(pre_call_evaluators=[e1, e2], post_call_evaluators=[])

        factory = _mock_span_factory()
        await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID, span_factory=factory)

        assert factory.calls == [("eval_a", "pre_call"), ("eval_b", "pre_call")]

    async def test_span_factory_called_per_post_call_evaluator(self):
        """span_factory is called once per sync post-call evaluator."""
        e1 = _mock_engine(name="eval_x")
        e2 = _mock_engine(name="eval_y")
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[e1, e2], mode="sync"
        )

        factory = _mock_span_factory()
        await interceptor.run_post_call(
            SESSION, _request(), {"r": 1}, REQUEST_ID, span_factory=factory
        )

        assert factory.calls == [("eval_x", "post_call"), ("eval_y", "post_call")]

    async def test_span_factory_sets_decision_attribute(self):
        """The yielded span gets an openbias.evaluator.decision attribute."""
        evaluator = _mock_engine(name="decider", decision=Decision.ALLOW)
        interceptor = Interceptor(pre_call_evaluators=[evaluator], post_call_evaluators=[])

        factory = _mock_span_factory()
        await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID, span_factory=factory)

        assert len(factory.spans) == 1
        factory.spans[0].set_attribute.assert_any_call(
            "openbias.evaluator.decision", "allow"
        )

    async def test_span_factory_passes_parent_span_to_evaluator(self):
        """The yielded span is passed as _parent_span in the evaluator context."""
        evaluator = _mock_engine(name="ctx_check")
        interceptor = Interceptor(pre_call_evaluators=[evaluator], post_call_evaluators=[])

        factory = _mock_span_factory()
        await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID, span_factory=factory)

        assert len(factory.spans) == 1
        call_args = evaluator.evaluate_request.call_args
        ctx = call_args.kwargs.get("context") or call_args[1].get("context")
        assert ctx["_parent_span"] is factory.spans[0]

    async def test_span_factory_called_for_applied_async_results(self):
        """span_factory is called with pre_call phase when applying pending async results."""
        async_evaluator = _mock_engine(
            name="async_eval",
            decision=Decision.ALLOW,
            delay=0.01,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[async_evaluator]
        )

        # Fire async evaluator during post_call
        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)  # Let the async task complete

        # Now run pre_call which collects async results
        factory = _mock_span_factory()
        await interceptor.run_pre_call(
            SESSION, _request(), "req-002", span_factory=factory
        )

        assert ("async_eval", "pre_call") in factory.calls
        # The span should have the async_applied source attribute
        applied_span = factory.spans[factory.calls.index(("async_eval", "pre_call"))]
        applied_span.set_attribute.assert_any_call(
            "openbias.evaluator.source", "async_applied"
        )

    async def test_trace_callback_invoked_for_applied_async_results(self):
        """trace_callback is called with result metadata and evaluator span at apply-time."""
        verdict_metadata = {"judge": {"verdicts": [{"rubric_name": "safety"}]}}
        async_evaluator = _mock_engine(
            name="traced_eval",
            decision=Decision.ALLOW,
            metadata=verdict_metadata,
            delay=0.01,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[async_evaluator]
        )

        # Fire async evaluator during post_call
        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        # Now run pre_call with both span_factory and trace_callback
        factory = _mock_span_factory()
        trace_cb = MagicMock()
        await interceptor.run_pre_call(
            SESSION, _request(), "req-003",
            span_factory=factory, trace_callback=trace_cb,
        )

        # trace_callback should have been called with the result metadata and span
        trace_cb.assert_called_once()
        call_args = trace_cb.call_args
        assert call_args[0][0] == verdict_metadata  # result.metadata
        assert call_args[0][1] is factory.spans[0]  # the evaluator span

    async def test_trace_callback_not_called_without_span(self):
        """trace_callback is NOT called when span_factory is not provided."""
        async_evaluator = _mock_engine(
            name="no_span_eval",
            decision=Decision.ALLOW,
            delay=0.01,
        )
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[async_evaluator]
        )

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        trace_cb = MagicMock()
        await interceptor.run_pre_call(
            SESSION, _request(), "req-004", trace_callback=trace_cb,
        )

        # No span_factory means nullcontext yields None, so trace_callback should not fire
        trace_cb.assert_not_called()

    async def test_span_factory_called_for_dispatched_async(self):
        """span_factory is called with post_call phase when dispatching async evaluators."""
        async_evaluator = _mock_engine(name="async_dispatch", delay=0.5)
        interceptor = Interceptor(
            pre_call_evaluators=[], post_call_evaluators=[async_evaluator]
        )

        factory = _mock_span_factory()
        await interceptor.run_post_call(
            SESSION, _request(), {"r": 1}, REQUEST_ID, span_factory=factory
        )

        assert ("async_dispatch", "post_call") in factory.calls
        # The span should have the async_dispatched source attribute
        dispatched_span = factory.spans[factory.calls.index(("async_dispatch", "post_call"))]
        dispatched_span.set_attribute.assert_any_call(
            "openbias.evaluator.source", "async_dispatched"
        )
        await interceptor.shutdown()

    async def test_no_span_factory_preserves_behavior(self):
        """Calling run_pre_call and run_post_call without span_factory works normally."""
        pre_eval = _mock_engine(name="pre", decision=Decision.ALLOW)
        post_eval = _mock_engine(name="post", decision=Decision.ALLOW)
        interceptor = Interceptor(
            pre_call_evaluators=[pre_eval],
            post_call_evaluators=[post_eval],
            mode="sync",
        )

        pre_result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)
        assert pre_result.allowed is True

        post_result = await interceptor.run_post_call(
            SESSION, _request(), {"r": 1}, REQUEST_ID
        )
        assert post_result.allowed is True
