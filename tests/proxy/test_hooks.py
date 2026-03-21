"""Tests for fail-open hardening of Open Sentinel proxy hooks."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from opensentinel.core.intervention.strategies import WorkflowViolationError
from opensentinel.proxy.hooks import safe_hook, _fail_open_counter, get_fail_open_counts


# ---------------------------------------------------------------------------
# safe_hook unit tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_fail_open_counter():
    """Reset the module-level counter between tests."""
    _fail_open_counter.clear()
    yield
    _fail_open_counter.clear()


async def test_safe_hook_success():
    """Normal execution passes through unchanged."""

    async def good_hook(x, y):
        return x + y

    result = await safe_hook(good_hook, 3, 4, timeout=1.0, hook_name="test_good")
    assert result == 7
    assert get_fail_open_counts() == {}


async def test_safe_hook_timeout_returns_fallback():
    """A slow hook is cancelled and the fallback is returned."""

    async def slow_hook():
        await asyncio.sleep(10)
        return "should not reach"

    result = await safe_hook(
        slow_hook, timeout=0.05, fallback="fallback_val", hook_name="test_slow"
    )
    assert result == "fallback_val"
    assert get_fail_open_counts()["test_slow"] == 1


async def test_safe_hook_exception_returns_fallback():
    """A crashing hook returns the fallback value."""

    async def bad_hook():
        raise RuntimeError("boom")

    result = await safe_hook(
        bad_hook, timeout=1.0, fallback={"original": True}, hook_name="test_crash"
    )
    assert result == {"original": True}
    assert get_fail_open_counts()["test_crash"] == 1


async def test_safe_hook_propagates_workflow_violation():
    """WorkflowViolationError is NOT swallowed -- intentional blocks must propagate."""

    async def blocking_hook():
        raise WorkflowViolationError("policy block", context={"reason": "test"})

    with pytest.raises(WorkflowViolationError, match="policy block"):
        await safe_hook(
            blocking_hook, timeout=1.0, fallback=None, hook_name="test_block"
        )
    # Counter should NOT be incremented for intentional blocks
    assert get_fail_open_counts() == {}


async def test_safe_hook_counter_increments_on_repeated_failures():
    """The fail-open counter correctly tracks multiple failures for the same hook."""

    async def flaky_hook():
        raise ValueError("flaky")

    for i in range(5):
        await safe_hook(flaky_hook, timeout=1.0, fallback=None, hook_name="flaky")

    assert get_fail_open_counts()["flaky"] == 5


async def test_safe_hook_passes_kwargs():
    """Keyword arguments are forwarded to the hook function."""

    async def kw_hook(*, greeting, name):
        return f"{greeting}, {name}!"

    result = await safe_hook(
        kw_hook, timeout=1.0, hook_name="kw_test", greeting="Hello", name="World"
    )
    assert result == "Hello, World!"


# ---------------------------------------------------------------------------
# safe_hook fail_open=False tests
# ---------------------------------------------------------------------------


async def test_safe_hook_fail_closed_exception_propagates():
    """When fail_open=False, exceptions propagate instead of returning fallback."""

    async def bad_hook():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await safe_hook(
            bad_hook, timeout=1.0, fallback="ignored", hook_name="test_closed",
            fail_open=False,
        )
    # Counter should NOT be incremented when fail_open=False
    assert get_fail_open_counts() == {}


async def test_safe_hook_fail_closed_timeout_propagates():
    """When fail_open=False, timeout propagates instead of returning fallback."""

    async def slow_hook():
        await asyncio.sleep(10)
        return "should not reach"

    with pytest.raises(asyncio.TimeoutError):
        await safe_hook(
            slow_hook, timeout=0.05, fallback="ignored", hook_name="test_closed_timeout",
            fail_open=False,
        )
    # Counter should NOT be incremented when fail_open=False
    assert get_fail_open_counts() == {}


async def test_safe_hook_fail_closed_workflow_violation_still_propagates():
    """WorkflowViolationError propagates regardless of fail_open setting."""

    async def blocking_hook():
        raise WorkflowViolationError("policy block", context={"reason": "test"})

    with pytest.raises(WorkflowViolationError, match="policy block"):
        await safe_hook(
            blocking_hook, timeout=1.0, fallback=None, hook_name="test_closed_block",
            fail_open=False,
        )
    assert get_fail_open_counts() == {}


async def test_safe_hook_fail_closed_success_still_works():
    """When fail_open=False, successful hooks still return normally."""

    async def good_hook():
        return 42

    result = await safe_hook(
        good_hook, timeout=1.0, hook_name="test_closed_ok", fail_open=False,
    )
    assert result == 42


# ---------------------------------------------------------------------------
# Integration tests: SentinelCallback hooks with fail-open
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings():
    """Create mock SentinelSettings for testing."""
    settings = MagicMock()
    settings.policy.fail_open = True
    settings.policy.hook_timeout_seconds = 0.1  # Aggressive timeout for tests
    settings.otel.enabled = False
    settings.debug = False
    return settings


@pytest.fixture
def callback(mock_settings):
    """Create SentinelCallback with mocked settings."""
    with patch("opensentinel.proxy.hooks.SentinelSettings", return_value=mock_settings):
        from opensentinel.proxy.hooks import SentinelCallback

        cb = SentinelCallback(settings=mock_settings)
        # Ensure tracer is None
        cb._tracer = None
        return cb


@pytest.fixture
def mock_api_key():
    return MagicMock()


@pytest.fixture
def mock_cache():
    return MagicMock()


async def test_pre_call_hook_timeout_returns_original_data(
    callback, mock_api_key, mock_cache
):
    """When interceptor is slow, pre_call_hook returns original data unchanged."""
    original_data = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-4",
    }

    async def slow_pre_call(*a, **kw):
        await asyncio.sleep(10)

    # Mock _get_interceptor to return a slow interceptor
    slow_interceptor = AsyncMock()
    slow_interceptor.run_pre_call = slow_pre_call
    callback._get_interceptor = AsyncMock(return_value=slow_interceptor)
    callback._interceptor_initialized = True

    result = await callback.async_pre_call_hook(
        mock_api_key, mock_cache, original_data, "completion"
    )
    # Should return original data unchanged (fail-open)
    assert result is original_data


async def test_post_call_hook_timeout_returns_original_response(
    callback, mock_api_key
):
    """When interceptor is slow, post_call_success_hook returns original response."""
    original_response = MagicMock()
    original_response.choices = []
    data = {"messages": [{"role": "user", "content": "hello"}]}

    async def slow_post_call(*a, **kw):
        await asyncio.sleep(10)

    slow_interceptor = AsyncMock()
    slow_interceptor.run_post_call = slow_post_call
    callback._get_interceptor = AsyncMock(return_value=slow_interceptor)
    callback._get_policy_engine = AsyncMock(return_value=None)
    callback._interceptor_initialized = True

    result = await callback.async_post_call_success_hook(
        data, mock_api_key, original_response
    )
    # Should return original response unchanged (fail-open)
    assert result is original_response


async def test_pre_call_hook_exception_returns_original_data(
    callback, mock_api_key, mock_cache
):
    """When interceptor crashes, pre_call_hook returns original data unchanged."""
    original_data = {"messages": [{"role": "user", "content": "hello"}]}

    crashing_interceptor = AsyncMock()
    crashing_interceptor.run_pre_call = AsyncMock(
        side_effect=RuntimeError("interceptor broke")
    )
    callback._get_interceptor = AsyncMock(return_value=crashing_interceptor)

    result = await callback.async_pre_call_hook(
        mock_api_key, mock_cache, original_data, "completion"
    )
    assert result is original_data


async def test_pre_call_hook_returns_exception_on_block(
    callback, mock_api_key, mock_cache
):
    """When interceptor blocks, _pre_call_impl returns an Exception object (not raises)."""
    from opensentinel.core.interceptor.types import InterceptionResult

    data = {"messages": [{"role": "user", "content": "hello"}]}

    mock_interceptor = MagicMock()
    mock_interceptor.run_pre_call = AsyncMock(
        return_value=InterceptionResult(allowed=False, message="blocked by policy")
    )
    callback._get_interceptor = AsyncMock(return_value=mock_interceptor)

    result = await callback.async_pre_call_hook(
        mock_api_key, mock_cache, data, "completion"
    )
    assert isinstance(result, Exception)
    assert "blocked by policy" in str(result)


async def test_post_call_hook_block_raises_workflow_violation(
    callback, mock_api_key
):
    """When sync POST_CALL checker blocks, WorkflowViolationError propagates."""
    from opensentinel.core.interceptor.types import InterceptionResult

    data = {"messages": [{"role": "user", "content": "hello"}]}
    response = MagicMock()
    response.choices = []

    mock_interceptor = MagicMock()
    mock_interceptor.run_post_call = AsyncMock(
        return_value=InterceptionResult(
            allowed=False, message="dangerous tool call blocked"
        )
    )
    callback._get_interceptor = AsyncMock(return_value=mock_interceptor)
    callback._get_policy_engine = AsyncMock(return_value=None)
    callback._interceptor_initialized = True

    with pytest.raises(WorkflowViolationError, match="dangerous tool call blocked"):
        await callback.async_post_call_success_hook(data, mock_api_key, response)


async def test_post_call_failure_hook_exception_is_swallowed(
    callback, mock_api_key
):
    """Exceptions in post_call_failure_hook are swallowed (fail-open)."""
    data = {"messages": [{"role": "user", "content": "hello"}]}

    async def crashing_impl(*a, **kw):
        raise RuntimeError("failure hook crashed")

    callback._post_call_failure_impl = crashing_impl

    # Should NOT raise
    result = await callback.async_post_call_failure_hook(
        data, mock_api_key, RuntimeError("original error")
    )
    assert result is None


async def test_log_success_event_exception_is_swallowed(callback):
    """Exceptions in async_log_success_event are swallowed (fail-open)."""

    async def crashing_impl(*a, **kw):
        raise RuntimeError("log hook crashed")

    callback._log_success_impl = crashing_impl
    now = datetime.now(timezone.utc)

    # Should NOT raise
    result = await callback.async_log_success_event(
        {"messages": []}, MagicMock(), now, now
    )
    assert result is None


async def test_log_failure_event_exception_is_swallowed(callback):
    """Exceptions in async_log_failure_event are swallowed (fail-open)."""

    async def crashing_impl(*a, **kw):
        raise RuntimeError("log failure hook crashed")

    callback._log_failure_impl = crashing_impl
    now = datetime.now(timezone.utc)

    # Should NOT raise
    result = await callback.async_log_failure_event(
        {"messages": []}, MagicMock(), now, now
    )
    assert result is None
