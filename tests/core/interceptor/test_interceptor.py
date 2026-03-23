"""
Comprehensive tests for the Interceptor orchestrator.

Covers:
- Sync PRE_CALL engine flow (pass, fail, short-circuit)
- Sync POST_CALL engine flow (pass, fail)
- Async engine lifecycle (fire, collect, cross-request handoff)
- Async edge cases (task failure, still-running, cleanup, shutdown)
- Interceptor init categorization
- Session TTL and LRU eviction
- Async task cap enforcement
- Context passing to engines
"""

import asyncio
import time
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
        """Single ALLOW engine — request goes through unchanged."""
        engine = _mock_engine()
        interceptor = Interceptor(engines=[engine])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is None

    async def test_block_blocks(self):
        """BLOCK engine blocks the request."""
        engine = _mock_engine(
            decision=Decision.BLOCK,
            message="forbidden",
        )
        interceptor = Interceptor(engines=[engine])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is False
        assert result.message == "forbidden"

    async def test_block_short_circuits(self):
        """First engine BLOCKs — second engine never runs."""
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
        interceptor = Interceptor(engines=[e1, e2])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is False
        assert call_count == 0  # Never reached

    async def test_engine_exception_fails_open(self):
        """Exception in a sync engine fails open (ALLOW)."""
        engine = _mock_engine(
            raise_on_evaluate=RuntimeError("kaboom"),
        )
        interceptor = Interceptor(engines=[engine])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        # Fail-open: request is allowed
        assert result.allowed is True

    async def test_intervene_injects_user_message_by_default(self):
        """INTERVENE engine injects a user message by default."""
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            message="Stay on topic",
        )
        interceptor = Interceptor(engines=[engine])
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        contents = [m["content"] for m in result.modified_data["messages"]]
        assert any("Stay on topic" in c for c in contents)

    async def test_intervene_system_prompt_append_strategy(self):
        """INTERVENE with system_prompt_append strategy appends to system prompt."""
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            message="Verify identity first",
        )
        interceptor = Interceptor(engines=[engine], default_strategy="system_prompt_append")
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        system_msg = result.modified_data["messages"][0]
        assert system_msg["role"] == "system"
        assert "Verify identity first" in system_msg["content"]

    async def test_intervene_user_message_inject_strategy(self):
        """INTERVENE with user_message_inject strategy injects a user message."""
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            message="Verify identity first",
        )
        interceptor = Interceptor(engines=[engine], default_strategy="user_message_inject")
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        # Should have a user guidance message injected
        contents = [m["content"] for m in result.modified_data["messages"]]
        assert any("Verify identity first" in c for c in contents)

    async def test_intervene_without_message_no_modification(self):
        """INTERVENE with no message — no modification applied."""
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            message=None,
        )
        interceptor = Interceptor(engines=[engine])
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is None

    async def test_intervene_modified_messages_replaces_messages(self):
        """INTERVENE with modified_messages replaces request messages directly."""
        sanitized = [{"role": "user", "content": "sanitized hello"}]
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            modified_messages=sanitized,
        )
        interceptor = Interceptor(engines=[engine])
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        assert result.modified_data["messages"] == sanitized

    async def test_intervene_modified_messages_takes_precedence_over_message(self):
        """modified_messages takes precedence over message text."""
        sanitized = [{"role": "user", "content": "sanitized"}]
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            message="This should be ignored",
            modified_messages=sanitized,
        )
        interceptor = Interceptor(engines=[engine])
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
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            message="Some guidance",
        )
        interceptor = Interceptor(engines=[engine], default_strategy="response_modification")
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        # response_modification is response-time only; request must be returned unmodified
        assert result.modified_data is None

    async def test_intervene_does_not_mutate_original_request_data(self):
        """Intervention must not mutate the caller's original request_data dict."""
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            message="Stay on topic",
        )
        interceptor = Interceptor(engines=[engine])
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
        """ALLOW engine — response goes through unchanged."""
        engine = _mock_engine()
        interceptor = Interceptor(engines=[engine], post_call_mode="sync")
        req = _request()

        result = await interceptor.run_post_call(SESSION, req, {"answer": "hi"}, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is None

    async def test_block_blocks(self):
        """BLOCK engine — response is blocked."""
        engine = _mock_engine(
            decision=Decision.BLOCK,
            message="toxic content",
        )
        interceptor = Interceptor(engines=[engine], post_call_mode="sync")

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is False
        assert result.message == "toxic content"

    async def test_intervene_returns_intervention_data(self):
        """INTERVENE engine — modified_data contains intervention info."""
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            message="Dangerous tool call detected",
        )
        interceptor = Interceptor(engines=[engine], post_call_mode="sync")

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
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            message="Replaced dangerous response",
            modified_messages=sanitized,
        )
        interceptor = Interceptor(engines=[engine], post_call_mode="sync")

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is True
        assert result.modified_data is not None
        interventions = result.modified_data["_interventions"]
        assert interventions[0]["modified_messages"] == sanitized

    async def test_multiple_intervene_engines_accumulate(self):
        """Multiple INTERVENE engines — all interventions are collected."""
        e1 = _mock_engine(
            name="engine1",
            decision=Decision.INTERVENE,
            message="Issue 1",
        )
        e2 = _mock_engine(
            name="engine2",
            decision=Decision.INTERVENE,
            message="Issue 2",
        )
        interceptor = Interceptor(engines=[e1, e2], post_call_mode="sync")

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is True
        interventions = result.modified_data["_interventions"]
        assert len(interventions) == 2

    async def test_engine_exception_fails_open(self):
        """Exception in sync POST_CALL engine fails open."""
        engine = _mock_engine(
            raise_on_evaluate=RuntimeError("kaboom"),
        )
        interceptor = Interceptor(engines=[engine], post_call_mode="sync")

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "hi"}, REQUEST_ID
        )

        assert result.allowed is True


# ===========================================================================
# Async engine lifecycle tests
# ===========================================================================


class TestAsyncCheckerLifecycle:

    async def test_async_engine_started_during_post_call(self):
        """After run_post_call, async task is stored in _running_tasks."""
        async_engine = _mock_engine(
            name="async_e",
            delay=0.5,
        )
        interceptor = Interceptor(engines=[async_engine])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        assert SESSION in interceptor._sessions
        assert len(interceptor._sessions.get(SESSION)) == 1
        await interceptor.shutdown()

    async def test_cross_request_handoff(self):
        """
        Full lifecycle: async engine runs during POST_CALL of req 1,
        its result is collected during PRE_CALL of req 2.
        """
        async_engine = _mock_engine(
            name="async_monitor",
            decision=Decision.ALLOW,
            delay=0.01,
        )
        interceptor = Interceptor(engines=[async_engine])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True

    async def test_async_intervene_modifies_next_request(self):
        """Async engine returns INTERVENE — next PRE_CALL applies intervention."""
        async_engine = _mock_engine(
            name="async_guide",
            decision=Decision.INTERVENE,
            message="Remember the workflow",
            delay=0.01,
        )
        interceptor = Interceptor(engines=[async_engine])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True
        assert result.modified_data is not None
        # Guidance is injected AFTER the last user message
        guidance_msg = result.modified_data["messages"][1]
        assert "Remember the workflow" in guidance_msg["content"]

    async def test_async_block_blocks_next_request(self):
        """Async engine returns BLOCK — next PRE_CALL blocks."""
        async_engine = _mock_engine(
            name="async_blocker",
            decision=Decision.BLOCK,
            message="violation detected async",
            delay=0.01,
        )
        interceptor = Interceptor(engines=[async_engine])

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

        Setup: three async engines complete — A=ALLOW, B=BLOCK, C=INTERVENE.
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
        engine_b.evaluate_response = AsyncMock(
            side_effect=lambda session_id, response_data, request_data, context=None: (
                asyncio.sleep(0.01)
            )
        )

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

        interceptor = Interceptor(engines=[engine_a, engine_b, engine_c])

        # Request 1: fire all three async engines
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
        """Async engine that raises — fails open on next collection."""
        async_engine = _mock_engine(
            name="async_crasher",
            raise_on_evaluate=RuntimeError("async boom"),
        )
        interceptor = Interceptor(engines=[async_engine])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        # Fail-open: request is allowed despite async error
        assert result.allowed is True

    async def test_still_running_not_collected(self):
        """Async task that isn't done yet stays in _running_tasks."""
        slow_engine = _mock_engine(
            name="slow_async",
            delay=5.0,
        )
        interceptor = Interceptor(engines=[slow_engine])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True
        assert SESSION in interceptor._sessions
        assert len(interceptor._sessions.get(SESSION)) == 1

        await interceptor.shutdown()

    async def test_cleanup_session_cancels_tasks(self):
        """cleanup_session cancels running tasks and clears pending results."""
        slow_engine = _mock_engine(
            name="cleanup_target",
            delay=5.0,
        )
        interceptor = Interceptor(engines=[slow_engine])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        assert SESSION in interceptor._sessions

        await interceptor.cleanup_session(SESSION)

        assert SESSION not in interceptor._sessions

    async def test_shutdown_cleans_all_sessions(self):
        """shutdown cancels tasks across all sessions."""
        slow_engine = _mock_engine(
            delay=5.0,
        )
        interceptor = Interceptor(engines=[slow_engine])

        await interceptor.run_post_call("session-a", _request(), {"r": 1}, REQUEST_ID)
        await interceptor.run_post_call("session-b", _request(), {"r": 2}, REQUEST_ID)

        assert len(interceptor._sessions) == 2

        await interceptor.shutdown()

        assert len(interceptor._sessions) == 0

    async def test_no_pending_async_on_first_request(self):
        """First PRE_CALL with no prior async results works cleanly."""
        interceptor = Interceptor(engines=[])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True

    async def test_cancelled_async_task_fails_open(self):
        """Cancelled async task is handled gracefully — next PRE_CALL is allowed."""
        slow_engine = _mock_engine(
            name="cancellable",
            delay=5.0,
        )
        interceptor = Interceptor(engines=[slow_engine])

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

    async def test_categorizes_engines_sync_post_call(self):
        """With post_call_mode='sync': engines go to _sync_pre_call and _sync_post_call."""
        engine = _mock_engine(name="e1")

        interceptor = Interceptor(engines=[engine], post_call_mode="sync")

        assert len(interceptor._sync_pre_call) == 1
        assert engine in interceptor._sync_pre_call
        assert len(interceptor._sync_post_call) == 1
        assert engine in interceptor._sync_post_call
        assert len(interceptor._async_post_call) == 0

    async def test_categorizes_engines_async_post_call(self):
        """With post_call_mode='async': engines go to _sync_pre_call and _async_post_call."""
        engine = _mock_engine(name="e1")

        interceptor = Interceptor(engines=[engine], post_call_mode="async")

        assert len(interceptor._sync_pre_call) == 1
        assert engine in interceptor._sync_pre_call
        assert len(interceptor._async_post_call) == 1
        assert engine in interceptor._async_post_call
        assert len(interceptor._sync_post_call) == 0

    async def test_empty_engines_list(self):
        """Interceptor with no engines still works."""
        interceptor = Interceptor(engines=[])

        pre = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)
        post = await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        assert pre.allowed is True
        assert post.allowed is True

    def test_invalid_default_strategy_raises(self) -> None:
        """Unknown default_strategy should raise ValueError at init time."""
        with pytest.raises(ValueError, match="hard_blok"):
            Interceptor(engines=[], default_strategy="hard_blok")

    @pytest.mark.parametrize(
        "strategy",
        ["system_prompt_append", "user_message_inject", "response_modification"],
    )
    def test_valid_default_strategies_accepted(self, strategy: str) -> None:
        """All StrategyType values should be accepted without error."""
        interceptor = Interceptor(engines=[], default_strategy=strategy)
        assert interceptor._default_strategy == strategy


# ===========================================================================
# Session TTL and LRU eviction tests
# ===========================================================================


class TestSessionEviction:

    async def test_session_evicted_after_ttl(self):
        """Sessions older than TTL are cleaned up on next run_pre_call."""
        slow_engine = _mock_engine(
            name="async_ttl",
            delay=0.01,
        )
        interceptor = Interceptor(engines=[slow_engine], session_ttl=1)

        await interceptor.run_post_call("old-session", _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        # Backdate the timestamp to simulate TTL expiry
        interceptor._sessions._timestamps["old-session"] = time.monotonic() - 2

        # Next pre_call should evict the stale session
        await interceptor.run_pre_call("new-session", _request(), REQUEST_ID)

        assert "old-session" not in interceptor._sessions

    async def test_max_sessions_eviction(self):
        """When max_sessions is exceeded, oldest sessions are evicted."""
        async_engine = _mock_engine(
            name="evict_cap",
            delay=0.01,
        )
        interceptor = Interceptor(engines=[async_engine], max_sessions=2)

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
        slow_engine = _mock_engine(
            name="cleanup_ts",
            delay=5.0,
        )
        interceptor = Interceptor(engines=[slow_engine])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        assert SESSION in interceptor._sessions

        await interceptor.cleanup_session(SESSION)

        assert SESSION not in interceptor._sessions


# ===========================================================================
# Async task cap tests
# ===========================================================================


class TestAsyncTaskCap:

    async def test_task_cap_drops_oldest(self):
        """When task cap is reached, oldest task is cancelled."""
        slow_engine = _mock_engine(
            name="capped",
            delay=5.0,
        )
        interceptor = Interceptor(engines=[slow_engine], max_async_tasks_per_session=2)

        # Start 3 tasks — cap is 2, so first should be cancelled
        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await interceptor.run_post_call(SESSION, _request(), {"r": 2}, REQUEST_ID)
        await interceptor.run_post_call(SESSION, _request(), {"r": 3}, REQUEST_ID)

        tasks = interceptor._sessions.get(SESSION)
        # Should have at most 2 active tasks
        active = [t for t in tasks if not t.done()]
        assert len(active) <= 2

        await interceptor.shutdown()

    async def test_completed_tasks_pruned_before_cap_check(self):
        """Completed tasks are pruned before enforcing the cap."""
        fast_engine = _mock_engine(
            name="fast",
            delay=0.01,
        )
        interceptor = Interceptor(engines=[fast_engine], max_async_tasks_per_session=2)

        # Start a task and let it complete
        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        # Start two more — the completed one should be pruned, so no drop needed
        await interceptor.run_post_call(SESSION, _request(), {"r": 2}, REQUEST_ID)
        await interceptor.run_post_call(SESSION, _request(), {"r": 3}, REQUEST_ID)

        # Should not have exceeded cap since first task completed
        tasks = interceptor._sessions.get(SESSION) or []
        active = [t for t in tasks if not t.done()]
        assert len(active) <= 2

        await interceptor.shutdown()


# ===========================================================================
# Context passing to engines
# ===========================================================================


class TestAsyncContextPassing:

    async def test_async_post_call_receives_context(self):
        """Async POST_CALL engines receive context with user_request_id."""
        received_context: dict[str, Any] = {}

        async def _evaluate_response(
            session_id: str,
            response_data: Any,
            request_data: dict[str, Any],
            context: dict[str, Any] | None = None,
        ) -> EngineResult:
            received_context.update(context or {})
            return EngineResult(decision=Decision.ALLOW)

        engine = _mock_engine(name="ctx_engine")
        engine.evaluate_response = AsyncMock(side_effect=_evaluate_response)

        interceptor = Interceptor(engines=[engine])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-ctx-001")
        await asyncio.sleep(0.05)

        assert received_context.get("user_request_id") == "req-ctx-001"

    async def test_sync_pre_call_receives_context(self):
        """Sync PRE_CALL engines receive context with user_request_id."""
        received_context: dict[str, Any] = {}

        async def _evaluate_request(
            session_id: str,
            request_data: dict[str, Any],
            context: dict[str, Any] | None = None,
        ) -> EngineResult:
            received_context.update(context or {})
            return EngineResult(decision=Decision.ALLOW)

        engine = _mock_engine(name="ctx_pre")
        engine.evaluate_request = AsyncMock(side_effect=_evaluate_request)

        interceptor = Interceptor(engines=[engine])

        await interceptor.run_pre_call(SESSION, _request(), "req-ctx-002")

        assert received_context.get("user_request_id") == "req-ctx-002"


# ===========================================================================
# fail_action upgrade logic
# ===========================================================================


class TestFailAction:

    async def test_default_fail_action_is_intervene(self):
        """Default fail_action is 'intervene' — INTERVENE stays INTERVENE."""
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            message="Stay on topic",
        )
        interceptor = Interceptor(engines=[engine])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None

    async def test_fail_action_block_upgrades_intervene_pre_call(self):
        """fail_action='block' upgrades INTERVENE to BLOCK on PRE_CALL."""
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            message="violation",
        )
        interceptor = Interceptor(engines=[engine], fail_action="block")

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is False
        assert result.message == "violation"

    async def test_fail_action_block_upgrades_intervene_post_call(self):
        """fail_action='block' upgrades INTERVENE to BLOCK on POST_CALL."""
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            message="violation",
        )
        interceptor = Interceptor(engines=[engine], post_call_mode="sync", fail_action="block")

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is False
        assert result.message == "violation"

    async def test_fail_action_block_does_not_affect_allow(self):
        """fail_action='block' does not upgrade ALLOW decisions."""
        engine = _mock_engine(
            decision=Decision.ALLOW,
        )
        interceptor = Interceptor(engines=[engine], fail_action="block")

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True

    async def test_fail_action_block_upgrades_async_intervene(self):
        """fail_action='block' upgrades async INTERVENE to BLOCK on next request."""
        async_engine = _mock_engine(
            name="async_guide",
            decision=Decision.INTERVENE,
            message="async violation",
            delay=0.01,
        )
        interceptor = Interceptor(engines=[async_engine], fail_action="block")

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is False
        assert "async violation" in (result.message or "")

    async def test_fail_action_shadow_downgrades_block_pre_call(self):
        """fail_action='shadow' downgrades BLOCK to ALLOW on PRE_CALL."""
        engine = _mock_engine(
            decision=Decision.BLOCK,
            message="blocked",
        )
        interceptor = Interceptor(engines=[engine], fail_action="shadow")

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True
        assert result.message is None

    async def test_fail_action_shadow_downgrades_intervene_pre_call(self):
        """fail_action='shadow' downgrades INTERVENE to ALLOW — no modifications."""
        engine = _mock_engine(
            decision=Decision.INTERVENE,
            message="Stay on topic",
        )
        interceptor = Interceptor(engines=[engine], fail_action="shadow")

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is None

    async def test_fail_action_shadow_downgrades_block_post_call(self):
        """fail_action='shadow' downgrades BLOCK to ALLOW on POST_CALL."""
        engine = _mock_engine(
            decision=Decision.BLOCK,
            message="blocked",
        )
        interceptor = Interceptor(engines=[engine], post_call_mode="sync", fail_action="shadow")

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is True
        assert result.message is None

    async def test_fail_action_shadow_downgrades_async_intervene(self):
        """fail_action='shadow' downgrades async INTERVENE — no modification on next request."""
        async_engine = _mock_engine(
            name="async_guide",
            decision=Decision.INTERVENE,
            message="async violation",
            delay=0.01,
        )
        interceptor = Interceptor(engines=[async_engine], fail_action="shadow")

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True
        assert result.modified_data is None
