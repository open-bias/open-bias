"""
Comprehensive tests for the Interceptor orchestrator.

Covers:
- Sync PRE_CALL checker flow (pass, fail, short-circuit)
- Sync POST_CALL checker flow (pass, fail)
- Async checker lifecycle (fire, collect, cross-request handoff)
- Async edge cases (task failure, still-running, cleanup, shutdown)
- Interceptor init categorization (4 buckets)
- Session TTL and LRU eviction
- Async task cap enforcement
- Context passing to async checkers
"""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

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

SESSION = "test-session"
REQUEST_ID = "req-001"


def _request(content: str = "hello") -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": content}], "model": "gpt-4"}


def _mock_checker(
    *,
    name: str = "fake",
    phase: CheckPhase = CheckPhase.PRE_CALL,
    mode: CheckerMode = CheckerMode.SYNC,
    decision: Decision = Decision.ALLOW,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
    modified_messages: list[dict[str, Any]] | None = None,
    delay: float = 0,
    raise_on_evaluate: Exception | None = None,
) -> PolicyEngineChecker:
    """Create a mock PolicyEngineChecker with configurable behavior."""
    engine = MagicMock()
    engine.name = name

    async def _evaluate(
        session_id: str,
        request_data: dict[str, Any],
        response_data: Any = None,
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

    checker = PolicyEngineChecker(engine=engine, phase=phase, mode=mode)
    checker.evaluate = _evaluate  # type: ignore[assignment]
    # Override name property
    type(checker).name = property(lambda self, n=name: f"{n}_{phase.value}")  # type: ignore[assignment]
    return checker


# ===========================================================================
# Sync PRE_CALL tests
# ===========================================================================


class TestSyncPreCall:

    async def test_pass_unchanged(self):
        """Single ALLOW checker — request goes through unchanged."""
        checker = _mock_checker(phase=CheckPhase.PRE_CALL)
        interceptor = Interceptor([checker])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is None

    async def test_block_blocks(self):
        """BLOCK checker blocks the request."""
        checker = _mock_checker(
            phase=CheckPhase.PRE_CALL,
            decision=Decision.BLOCK,
            message="forbidden",
        )
        interceptor = Interceptor([checker])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is False
        assert result.message == "forbidden"

    async def test_block_short_circuits(self):
        """First checker BLOCKs — second checker never runs."""
        c1 = _mock_checker(
            name="blocker",
            phase=CheckPhase.PRE_CALL,
            decision=Decision.BLOCK,
        )
        call_count = 0
        original_evaluate = _mock_checker(
            name="skipped", phase=CheckPhase.PRE_CALL
        ).evaluate

        async def counting_evaluate(*args: Any, **kwargs: Any) -> EngineResult:
            nonlocal call_count
            call_count += 1
            return await original_evaluate(*args, **kwargs)

        c2 = _mock_checker(name="skipped", phase=CheckPhase.PRE_CALL)
        c2.evaluate = counting_evaluate  # type: ignore[assignment]
        interceptor = Interceptor([c1, c2])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is False
        assert call_count == 0  # Never reached

    async def test_checker_exception_fails_open(self):
        """Exception in a sync checker fails open (ALLOW)."""
        checker = _mock_checker(
            phase=CheckPhase.PRE_CALL,
            raise_on_evaluate=RuntimeError("kaboom"),
        )
        interceptor = Interceptor([checker])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        # Fail-open: request is allowed
        assert result.allowed is True


    async def test_intervene_modifies_request_system_prompt(self):
        """INTERVENE checker appends guidance to system prompt by default."""
        checker = _mock_checker(
            phase=CheckPhase.PRE_CALL,
            decision=Decision.INTERVENE,
            message="Stay on topic",
        )
        interceptor = Interceptor([checker])
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        system_msg = result.modified_data["messages"][0]
        assert system_msg["role"] == "system"
        assert "Stay on topic" in system_msg["content"]

    async def test_intervene_user_message_inject_strategy(self):
        """INTERVENE with user_message_inject strategy injects a user message."""
        checker = _mock_checker(
            phase=CheckPhase.PRE_CALL,
            decision=Decision.INTERVENE,
            message="Verify identity first",
        )
        interceptor = Interceptor([checker], default_strategy="user_message_inject")
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        # Should have a user guidance message injected
        contents = [m["content"] for m in result.modified_data["messages"]]
        assert any("Verify identity first" in c for c in contents)

    async def test_intervene_without_message_no_modification(self):
        """INTERVENE with no message — no modification applied."""
        checker = _mock_checker(
            phase=CheckPhase.PRE_CALL,
            decision=Decision.INTERVENE,
            message=None,
        )
        interceptor = Interceptor([checker])
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is None

    async def test_intervene_modified_messages_replaces_messages(self):
        """INTERVENE with modified_messages replaces request messages directly."""
        sanitized = [{"role": "user", "content": "sanitized hello"}]
        checker = _mock_checker(
            phase=CheckPhase.PRE_CALL,
            decision=Decision.INTERVENE,
            modified_messages=sanitized,
        )
        interceptor = Interceptor([checker])
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        assert result.modified_data["messages"] == sanitized

    async def test_intervene_modified_messages_takes_precedence_over_message(self):
        """modified_messages takes precedence over message text."""
        sanitized = [{"role": "user", "content": "sanitized"}]
        checker = _mock_checker(
            phase=CheckPhase.PRE_CALL,
            decision=Decision.INTERVENE,
            message="This should be ignored",
            modified_messages=sanitized,
        )
        interceptor = Interceptor([checker])
        req = _request()

        result = await interceptor.run_pre_call(SESSION, req, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is not None
        assert result.modified_data["messages"] == sanitized


# ===========================================================================
# Sync POST_CALL tests
# ===========================================================================


class TestSyncPostCall:

    async def test_pass_unchanged(self):
        """ALLOW checker — response goes through unchanged."""
        checker = _mock_checker(phase=CheckPhase.POST_CALL)
        interceptor = Interceptor([checker])
        req = _request()

        result = await interceptor.run_post_call(SESSION, req, {"answer": "hi"}, REQUEST_ID)

        assert result.allowed is True
        assert result.modified_data is None

    async def test_block_blocks(self):
        """BLOCK checker — response is blocked."""
        checker = _mock_checker(
            phase=CheckPhase.POST_CALL,
            decision=Decision.BLOCK,
            message="toxic content",
        )
        interceptor = Interceptor([checker])

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is False
        assert result.message == "toxic content"

    async def test_intervene_returns_intervention_data(self):
        """INTERVENE checker — modified_data contains intervention info."""
        checker = _mock_checker(
            phase=CheckPhase.POST_CALL,
            decision=Decision.INTERVENE,
            message="Dangerous tool call detected",
        )
        interceptor = Interceptor([checker])

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
        checker = _mock_checker(
            phase=CheckPhase.POST_CALL,
            decision=Decision.INTERVENE,
            message="Replaced dangerous response",
            modified_messages=sanitized,
        )
        interceptor = Interceptor([checker])

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is True
        assert result.modified_data is not None
        interventions = result.modified_data["_interventions"]
        assert interventions[0]["modified_messages"] == sanitized

    async def test_multiple_intervene_checkers_accumulate(self):
        """Multiple INTERVENE checkers — all interventions are collected."""
        c1 = _mock_checker(
            name="checker1",
            phase=CheckPhase.POST_CALL,
            decision=Decision.INTERVENE,
            message="Issue 1",
        )
        c2 = _mock_checker(
            name="checker2",
            phase=CheckPhase.POST_CALL,
            decision=Decision.INTERVENE,
            message="Issue 2",
        )
        interceptor = Interceptor([c1, c2])

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "bad"}, REQUEST_ID
        )

        assert result.allowed is True
        interventions = result.modified_data["_interventions"]
        assert len(interventions) == 2

    async def test_checker_exception_fails_open(self):
        """Exception in sync POST_CALL checker fails open."""
        checker = _mock_checker(
            phase=CheckPhase.POST_CALL,
            raise_on_evaluate=RuntimeError("kaboom"),
        )
        interceptor = Interceptor([checker])

        result = await interceptor.run_post_call(
            SESSION, _request(), {"answer": "hi"}, REQUEST_ID
        )

        assert result.allowed is True


# ===========================================================================
# Async checker lifecycle tests
# ===========================================================================


class TestAsyncCheckerLifecycle:

    async def test_async_checker_started_during_post_call(self):
        """After run_post_call, async task is stored in _running_tasks."""
        async_checker = _mock_checker(
            name="async_c",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            delay=0.5,
        )
        interceptor = Interceptor([async_checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        assert SESSION in interceptor._running_tasks
        assert len(interceptor._running_tasks[SESSION]) == 1
        await interceptor.shutdown()

    async def test_cross_request_handoff(self):
        """
        Full lifecycle: async checker runs during POST_CALL of req 1,
        its result is collected during PRE_CALL of req 2.
        """
        async_checker = _mock_checker(
            name="async_monitor",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            decision=Decision.ALLOW,
            delay=0.01,
        )
        interceptor = Interceptor([async_checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True

    async def test_async_intervene_modifies_next_request(self):
        """Async checker returns INTERVENE — next PRE_CALL applies intervention."""
        async_checker = _mock_checker(
            name="async_guide",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            decision=Decision.INTERVENE,
            message="Remember the workflow",
            delay=0.01,
        )
        interceptor = Interceptor([async_checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True
        assert result.modified_data is not None
        system_msg = result.modified_data["messages"][0]
        assert "Remember the workflow" in system_msg["content"]

    async def test_async_block_blocks_next_request(self):
        """Async checker returns BLOCK — next PRE_CALL blocks."""
        async_checker = _mock_checker(
            name="async_blocker",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            decision=Decision.BLOCK,
            message="violation detected async",
            delay=0.01,
        )
        interceptor = Interceptor([async_checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is False
        assert "violation detected async" in (result.message or "")


# ===========================================================================
# Async edge cases
# ===========================================================================


class TestAsyncEdgeCases:

    async def test_async_exception_fails_open(self):
        """Async checker that raises — fails open on next collection."""
        async_checker = _mock_checker(
            name="async_crasher",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            raise_on_evaluate=RuntimeError("async boom"),
        )
        interceptor = Interceptor([async_checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        # Fail-open: request is allowed despite async error
        assert result.allowed is True

    async def test_still_running_not_collected(self):
        """Async task that isn't done yet stays in _running_tasks."""
        slow_checker = _mock_checker(
            name="slow_async",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            delay=5.0,
        )
        interceptor = Interceptor([slow_checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        result = await interceptor.run_pre_call(SESSION, _request(), "req-002")

        assert result.allowed is True
        assert SESSION in interceptor._running_tasks
        assert len(interceptor._running_tasks[SESSION]) == 1

        await interceptor.shutdown()

    async def test_cleanup_session_cancels_tasks(self):
        """cleanup_session cancels running tasks and clears pending results."""
        slow_checker = _mock_checker(
            name="cleanup_target",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            delay=5.0,
        )
        interceptor = Interceptor([slow_checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        assert SESSION in interceptor._running_tasks

        await interceptor.cleanup_session(SESSION)

        assert SESSION not in interceptor._running_tasks

    async def test_shutdown_cleans_all_sessions(self):
        """shutdown cancels tasks across all sessions."""
        slow_checker = _mock_checker(
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            delay=5.0,
        )
        interceptor = Interceptor([slow_checker])

        await interceptor.run_post_call("session-a", _request(), {"r": 1}, REQUEST_ID)
        await interceptor.run_post_call("session-b", _request(), {"r": 2}, REQUEST_ID)

        assert len(interceptor._running_tasks) == 2

        await interceptor.shutdown()

        assert len(interceptor._running_tasks) == 0

    async def test_no_pending_async_on_first_request(self):
        """First PRE_CALL with no prior async results works cleanly."""
        interceptor = Interceptor([])

        result = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)

        assert result.allowed is True


# ===========================================================================
# Interceptor init categorization
# ===========================================================================


class TestInterceptorInit:

    async def test_categorizes_checkers_correctly(self):
        """Checkers are bucketed into sync_pre, sync_post, async_pre, async_post."""
        sync_pre = _mock_checker(
            name="sp", phase=CheckPhase.PRE_CALL, mode=CheckerMode.SYNC
        )
        sync_post = _mock_checker(
            name="spo", phase=CheckPhase.POST_CALL, mode=CheckerMode.SYNC
        )
        async_pre = _mock_checker(
            name="ap", phase=CheckPhase.PRE_CALL, mode=CheckerMode.ASYNC
        )
        async_post = _mock_checker(
            name="apo", phase=CheckPhase.POST_CALL, mode=CheckerMode.ASYNC
        )

        interceptor = Interceptor([sync_pre, sync_post, async_pre, async_post])

        assert len(interceptor._sync_pre_call) == 1
        assert len(interceptor._sync_post_call) == 1
        assert len(interceptor._async_pre_call) == 1
        assert len(interceptor._async_post_call) == 1

    async def test_empty_checkers_list(self):
        """Interceptor with no checkers still works."""
        interceptor = Interceptor([])

        pre = await interceptor.run_pre_call(SESSION, _request(), REQUEST_ID)
        post = await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)

        assert pre.allowed is True
        assert post.allowed is True


# ===========================================================================
# Session TTL and LRU eviction tests
# ===========================================================================


class TestSessionEviction:

    async def test_session_evicted_after_ttl(self):
        """Sessions older than TTL are cleaned up on next run_pre_call."""
        slow_checker = _mock_checker(
            name="async_ttl",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            delay=0.01,
        )
        interceptor = Interceptor([slow_checker], session_ttl=1)

        await interceptor.run_post_call("old-session", _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        # Backdate the timestamp to simulate TTL expiry
        interceptor._session_timestamps["old-session"] = time.monotonic() - 2

        # Next pre_call should evict the stale session
        await interceptor.run_pre_call("new-session", _request(), REQUEST_ID)

        assert "old-session" not in interceptor._running_tasks
        assert "old-session" not in interceptor._session_timestamps

    async def test_max_sessions_eviction(self):
        """When max_sessions is exceeded, oldest sessions are evicted."""
        interceptor = Interceptor([], max_sessions=2)

        await interceptor.run_pre_call("session-1", _request(), REQUEST_ID)
        await interceptor.run_pre_call("session-2", _request(), REQUEST_ID)
        await interceptor.run_pre_call("session-3", _request(), REQUEST_ID)

        assert len(interceptor._session_timestamps) <= 2
        assert "session-3" in interceptor._session_timestamps

    async def test_cleanup_session_removes_timestamp(self):
        """cleanup_session removes both tasks and timestamp."""
        slow_checker = _mock_checker(
            name="cleanup_ts",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            delay=5.0,
        )
        interceptor = Interceptor([slow_checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        assert SESSION in interceptor._session_timestamps

        await interceptor.cleanup_session(SESSION)

        assert SESSION not in interceptor._running_tasks
        assert SESSION not in interceptor._session_timestamps


# ===========================================================================
# Async task cap tests
# ===========================================================================


class TestAsyncTaskCap:

    async def test_task_cap_drops_oldest(self):
        """When task cap is reached, oldest task is cancelled."""
        slow_checker = _mock_checker(
            name="capped",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            delay=5.0,
        )
        interceptor = Interceptor([slow_checker], max_async_tasks_per_session=2)

        # Start 3 tasks — cap is 2, so first should be cancelled
        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await interceptor.run_post_call(SESSION, _request(), {"r": 2}, REQUEST_ID)
        await interceptor.run_post_call(SESSION, _request(), {"r": 3}, REQUEST_ID)

        tasks = interceptor._running_tasks[SESSION]
        # Should have at most 2 active tasks
        active = [t for t in tasks if not t.done()]
        assert len(active) <= 2

        await interceptor.shutdown()

    async def test_completed_tasks_pruned_before_cap_check(self):
        """Completed tasks are pruned before enforcing the cap."""
        fast_checker = _mock_checker(
            name="fast",
            phase=CheckPhase.POST_CALL,
            mode=CheckerMode.ASYNC,
            delay=0.01,
        )
        interceptor = Interceptor([fast_checker], max_async_tasks_per_session=2)

        # Start a task and let it complete
        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, REQUEST_ID)
        await asyncio.sleep(0.05)

        # Start two more — the completed one should be pruned, so no drop needed
        await interceptor.run_post_call(SESSION, _request(), {"r": 2}, REQUEST_ID)
        await interceptor.run_post_call(SESSION, _request(), {"r": 3}, REQUEST_ID)

        # Should not have exceeded cap since first task completed
        tasks = interceptor._running_tasks.get(SESSION, [])
        active = [t for t in tasks if not t.done()]
        assert len(active) <= 2

        await interceptor.shutdown()


# ===========================================================================
# Context passing to async checkers
# ===========================================================================


class TestAsyncContextPassing:

    async def test_async_post_call_receives_context(self):
        """Async POST_CALL checkers receive context with user_request_id."""
        received_context: dict[str, Any] = {}

        engine = MagicMock()
        engine.name = "ctx_checker"

        async def _evaluate(
            session_id: str,
            request_data: dict[str, Any],
            response_data: Any = None,
            context: dict[str, Any] | None = None,
        ) -> EngineResult:
            received_context.update(context or {})
            return EngineResult(decision=Decision.ALLOW)

        checker = PolicyEngineChecker(
            engine=engine, phase=CheckPhase.POST_CALL, mode=CheckerMode.ASYNC
        )
        checker.evaluate = _evaluate  # type: ignore[assignment]
        type(checker).name = property(lambda self: "ctx_checker_post_call")  # type: ignore[assignment]

        interceptor = Interceptor([checker])

        await interceptor.run_post_call(SESSION, _request(), {"r": 1}, "req-ctx-001")
        await asyncio.sleep(0.05)

        assert received_context.get("user_request_id") == "req-ctx-001"

    async def test_async_pre_call_receives_context(self):
        """Async PRE_CALL checkers receive context with user_request_id."""
        received_context: dict[str, Any] = {}

        engine = MagicMock()
        engine.name = "ctx_pre"

        async def _evaluate(
            session_id: str,
            request_data: dict[str, Any],
            response_data: Any = None,
            context: dict[str, Any] | None = None,
        ) -> EngineResult:
            received_context.update(context or {})
            return EngineResult(decision=Decision.ALLOW)

        checker = PolicyEngineChecker(
            engine=engine, phase=CheckPhase.PRE_CALL, mode=CheckerMode.ASYNC
        )
        checker.evaluate = _evaluate  # type: ignore[assignment]
        type(checker).name = property(lambda self: "ctx_pre_pre_call")  # type: ignore[assignment]

        interceptor = Interceptor([checker])

        await interceptor.run_pre_call(SESSION, _request(), "req-ctx-002")
        await asyncio.sleep(0.05)

        assert received_context.get("user_request_id") == "req-ctx-002"
