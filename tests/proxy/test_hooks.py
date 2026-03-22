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


async def test_post_call_intervention_modifies_returned_response(
    callback, mock_api_key
):
    """POST_CALL intervention modifies the response returned by the hook."""
    from opensentinel.core.interceptor.types import InterceptionResult
    from litellm import ModelResponse

    response = ModelResponse(
        choices=[{"message": {"role": "assistant", "content": "original answer"}}],
        model="test-model",
    )

    mock_interceptor = MagicMock()
    mock_interceptor.run_post_call = AsyncMock(
        return_value=InterceptionResult(
            allowed=True,
            modified_data={
                "_interventions": [
                    {
                        "checker": "test_checker",
                        "message": "policy warning: be careful",
                    }
                ]
            },
        )
    )
    callback._get_interceptor = AsyncMock(return_value=mock_interceptor)
    callback._interceptor_initialized = True

    result = await callback.async_post_call_success_hook(
        {"messages": [{"role": "user", "content": "hello"}]},
        mock_api_key,
        response,
    )

    # The returned response must contain the intervention text
    content = result.choices[0].message.content
    assert "original answer" in content
    assert "[POLICY WARNING]: policy warning: be careful" in content
    # Verify it's the same object (in-place mutation)
    assert result is response


async def test_post_call_intervention_replaces_response_content(
    callback, mock_api_key
):
    """POST_CALL intervention with modified_messages replaces response content entirely."""
    from opensentinel.core.interceptor.types import InterceptionResult
    from litellm import ModelResponse

    response = ModelResponse(
        choices=[{"message": {"role": "assistant", "content": "dangerous answer"}}],
        model="test-model",
    )

    mock_interceptor = MagicMock()
    mock_interceptor.run_post_call = AsyncMock(
        return_value=InterceptionResult(
            allowed=True,
            modified_data={
                "_interventions": [
                    {
                        "checker": "test_checker",
                        "modified_messages": [
                            {"role": "assistant", "content": "safe replacement"}
                        ],
                    }
                ]
            },
        )
    )
    callback._get_interceptor = AsyncMock(return_value=mock_interceptor)
    callback._interceptor_initialized = True

    result = await callback.async_post_call_success_hook(
        {"messages": [{"role": "user", "content": "hello"}]},
        mock_api_key,
        response,
    )

    assert result.choices[0].message.content == "safe replacement"
    assert result is response


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


# ---------------------------------------------------------------------------
# Eager startup initialization tests: SentinelCallback.initialize()
# ---------------------------------------------------------------------------


async def test_callback_initialize_raises_on_bad_engine_config(mock_settings):
    """initialize() raises immediately when the engine fails to initialize."""
    from opensentinel.proxy.hooks import SentinelCallback

    # Simulate a judge engine with models configured (non-skip path)
    mock_settings.get_policy_config.return_value = {
        "type": "judge",
        "config": {"models": [{"name": "primary", "model": "gpt-4o"}]},
    }

    cb = SentinelCallback(settings=mock_settings)

    with patch(
        "opensentinel.policy.registry.PolicyEngineRegistry.create_and_initialize",
        new=AsyncMock(side_effect=ValueError("bad model config")),
    ):
        with pytest.raises(ValueError, match="bad model config"):
            await cb.initialize()


async def test_callback_initialize_sets_engine_on_success(mock_settings):
    """initialize() eagerly sets the policy engine when config is valid."""
    from opensentinel.proxy.hooks import SentinelCallback

    mock_engine = MagicMock()
    mock_engine.name = "mock-judge"

    mock_settings.get_policy_config.return_value = {
        "type": "judge",
        "config": {"models": [{"name": "primary", "model": "gpt-4o"}]},
    }
    mock_settings.policy.post_call_mode = "async"
    mock_settings.policy.default_strategy = "user_message_inject"
    mock_settings.policy.fail_action = "intervene"

    cb = SentinelCallback(settings=mock_settings)

    with patch(
        "opensentinel.policy.registry.PolicyEngineRegistry.create_and_initialize",
        new=AsyncMock(return_value=mock_engine),
    ):
        await cb.initialize()

    assert cb._policy_engine is mock_engine
    assert cb._policy_engine_initialized is True
    assert cb._interceptor_initialized is True


async def test_callback_initialize_skips_unconfigured_engine(mock_settings):
    """initialize() skips the engine and stays None when no config is provided."""
    from opensentinel.proxy.hooks import SentinelCallback

    # Judge engine with no models — should be skipped
    mock_settings.get_policy_config.return_value = {
        "type": "judge",
        "config": {},
    }
    mock_settings.policy.post_call_mode = "async"
    mock_settings.policy.default_strategy = "pass"
    mock_settings.policy.fail_action = "pass"

    cb = SentinelCallback(settings=mock_settings)

    with patch(
        "opensentinel.policy.registry.PolicyEngineRegistry.create_and_initialize",
        new=AsyncMock(),
    ) as mock_create:
        await cb.initialize()

    mock_create.assert_not_called()
    assert cb._policy_engine is None
    assert cb._policy_engine_initialized is True


async def test_callback_initialize_idempotent(mock_settings):
    """Calling initialize() twice does not re-initialize the engine."""
    from opensentinel.proxy.hooks import SentinelCallback

    mock_engine = MagicMock()
    mock_engine.name = "mock-judge"

    mock_settings.get_policy_config.return_value = {
        "type": "judge",
        "config": {"models": [{"name": "primary", "model": "gpt-4o"}]},
    }
    mock_settings.policy.post_call_mode = "async"
    mock_settings.policy.default_strategy = "user_message_inject"
    mock_settings.policy.fail_action = "intervene"

    cb = SentinelCallback(settings=mock_settings)

    with patch(
        "opensentinel.policy.registry.PolicyEngineRegistry.create_and_initialize",
        new=AsyncMock(return_value=mock_engine),
    ) as mock_create:
        await cb.initialize()
        await cb.initialize()  # second call should be a no-op

    # Registry should only have been called once
    mock_create.assert_called_once()


# ---------------------------------------------------------------------------
# _compute_hook_timeout / _effective_hook_timeout tests
# ---------------------------------------------------------------------------


async def test_compute_hook_timeout_no_engine(mock_settings):
    """Without an engine, effective timeout equals the configured value."""
    from opensentinel.proxy.hooks import SentinelCallback

    mock_settings.policy.hook_timeout_seconds = 30.0
    mock_settings.get_policy_config.return_value = {"type": "judge", "config": {}}
    mock_settings.policy.post_call_mode = "async"

    cb = SentinelCallback(settings=mock_settings)
    cb._policy_engine = None

    assert cb._compute_hook_timeout() == 30.0
    assert cb._effective_hook_timeout == 30.0  # set at __init__ before engine exists


async def test_compute_hook_timeout_engine_without_timeout_attr(mock_settings):
    """Engines that don't expose a 'timeout' attribute use the configured value."""
    from opensentinel.proxy.hooks import SentinelCallback

    mock_settings.policy.hook_timeout_seconds = 30.0
    cb = SentinelCallback(settings=mock_settings)

    engine = MagicMock(spec=[])  # no 'timeout' attribute at all
    cb._policy_engine = engine

    assert cb._compute_hook_timeout() == 30.0


async def test_compute_hook_timeout_engine_timeout_within_configured(mock_settings):
    """When engine timeout + buffer <= configured, no adjustment is made."""
    from opensentinel.proxy.hooks import SentinelCallback

    mock_settings.policy.hook_timeout_seconds = 30.0
    cb = SentinelCallback(settings=mock_settings)

    engine = MagicMock()
    engine.timeout = 15.0  # 15 + 5 = 20 < 30 → no change
    cb._policy_engine = engine

    assert cb._compute_hook_timeout() == 30.0


async def test_compute_hook_timeout_engine_timeout_exceeds_configured(mock_settings):
    """When engine timeout + buffer > configured, effective timeout is auto-adjusted."""
    from opensentinel.proxy.hooks import SentinelCallback

    mock_settings.policy.hook_timeout_seconds = 10.0
    cb = SentinelCallback(settings=mock_settings)

    engine = MagicMock()
    engine.timeout = 20.0  # 20 + 5 = 25 > 10 → adjust
    cb._policy_engine = engine

    assert cb._compute_hook_timeout() == 25.0


async def test_compute_hook_timeout_logs_warning_when_adjusting(mock_settings, caplog):
    """A warning is logged when the timeout is auto-adjusted upward."""
    import logging
    from opensentinel.proxy.hooks import SentinelCallback

    mock_settings.policy.hook_timeout_seconds = 10.0
    cb = SentinelCallback(settings=mock_settings)

    engine = MagicMock()
    engine.timeout = 20.0
    cb._policy_engine = engine

    with caplog.at_level(logging.WARNING, logger="opensentinel.proxy.hooks"):
        result = cb._compute_hook_timeout()

    assert result == 25.0
    assert "Auto-adjusting" in caplog.text
    assert "10.0" in caplog.text  # configured value mentioned
    assert "20.0" in caplog.text  # engine timeout mentioned


async def test_compute_hook_timeout_non_numeric_engine_timeout(mock_settings):
    """Non-numeric engine timeout is ignored and configured value is used."""
    from opensentinel.proxy.hooks import SentinelCallback

    mock_settings.policy.hook_timeout_seconds = 30.0
    cb = SentinelCallback(settings=mock_settings)

    engine = MagicMock()
    engine.timeout = "not-a-number"
    cb._policy_engine = engine

    assert cb._compute_hook_timeout() == 30.0


async def test_effective_hook_timeout_updated_after_eager_init(mock_settings):
    """After initialize(), _effective_hook_timeout reflects the engine configuration."""
    from opensentinel.proxy.hooks import SentinelCallback

    mock_settings.policy.hook_timeout_seconds = 10.0
    mock_settings.get_policy_config.return_value = {
        "type": "judge",
        "config": {"models": [{"name": "primary", "model": "gpt-4o"}]},
    }
    mock_settings.policy.post_call_mode = "async"
    mock_settings.policy.default_strategy = "user_message_inject"
    mock_settings.policy.fail_action = "intervene"

    mock_engine = MagicMock()
    mock_engine.name = "mock-judge"
    mock_engine.timeout = 20.0  # 20 + 5 = 25 > 10 → should adjust

    cb = SentinelCallback(settings=mock_settings)
    assert cb._effective_hook_timeout == 10.0  # pre-init: configured value

    with patch(
        "opensentinel.policy.registry.PolicyEngineRegistry.create_and_initialize",
        new=AsyncMock(return_value=mock_engine),
    ):
        await cb.initialize()

    assert cb._effective_hook_timeout == 25.0  # post-init: auto-adjusted


async def test_effective_hook_timeout_updated_after_lazy_init(mock_settings):
    """After lazy _get_interceptor(), _effective_hook_timeout is updated."""
    from opensentinel.proxy.hooks import SentinelCallback

    mock_settings.policy.hook_timeout_seconds = 10.0
    mock_settings.get_policy_config.return_value = {
        "type": "judge",
        "config": {"models": [{"name": "primary", "model": "gpt-4o"}]},
    }
    mock_settings.policy.post_call_mode = "async"
    mock_settings.policy.default_strategy = "user_message_inject"
    mock_settings.policy.fail_action = "intervene"

    mock_engine = MagicMock()
    mock_engine.name = "mock-judge"
    mock_engine.timeout = 20.0  # triggers auto-adjust

    cb = SentinelCallback(settings=mock_settings)
    assert cb._effective_hook_timeout == 10.0

    with patch(
        "opensentinel.policy.registry.PolicyEngineRegistry.create_and_initialize",
        new=AsyncMock(return_value=mock_engine),
    ):
        await cb._get_interceptor()

    assert cb._effective_hook_timeout == 25.0


# ---------------------------------------------------------------------------
# JudgePolicyEngine.timeout property tests
# ---------------------------------------------------------------------------


def test_judge_engine_timeout_no_client():
    """Before initialization, judge engine timeout returns 0.0."""
    from opensentinel.policy.engines.judge.engine import JudgePolicyEngine

    engine = JudgePolicyEngine()
    assert engine.timeout == 0.0


def test_judge_engine_timeout_single_model():
    """Judge engine timeout accounts for retries: timeout * (max_retries + 1)."""
    from opensentinel.policy.engines.judge.engine import JudgePolicyEngine
    from opensentinel.policy.engines.judge.client import JudgeClient

    engine = JudgePolicyEngine()
    engine._client = JudgeClient()
    engine._client.add_model("primary", model="gpt-4o", timeout=15.0, max_retries=2)

    # worst-case = 15.0 * (2 + 1) = 45.0
    assert engine.timeout == 45.0


def test_judge_engine_timeout_multiple_models_returns_max():
    """Judge engine timeout returns the max worst-case across all models."""
    from opensentinel.policy.engines.judge.engine import JudgePolicyEngine
    from opensentinel.policy.engines.judge.client import JudgeClient

    engine = JudgePolicyEngine()
    engine._client = JudgeClient()
    engine._client.add_model("primary", model="gpt-4o", timeout=15.0, max_retries=2)
    engine._client.add_model("secondary", model="gpt-4o-mini", timeout=30.0, max_retries=2)

    # max(15*3, 30*3) = max(45, 90) = 90.0
    assert engine.timeout == 90.0


def test_judge_engine_timeout_empty_client():
    """Judge engine with an empty client (no models) returns 0.0."""
    from opensentinel.policy.engines.judge.engine import JudgePolicyEngine
    from opensentinel.policy.engines.judge.client import JudgeClient

    engine = JudgePolicyEngine()
    engine._client = JudgeClient()  # no models added

    assert engine.timeout == 0.0


# ---------------------------------------------------------------------------
# Duplicate trace entry guard tests
# ---------------------------------------------------------------------------


async def test_post_call_traces_once_when_both_hooks_fire(callback, mock_api_key):
    """log_llm_call is called exactly once even if both hooks process the same data dict."""
    from opensentinel.proxy.hooks import SentinelCallback
    from datetime import datetime, timezone

    mock_tracer = MagicMock()
    callback._tracer = mock_tracer

    # No interceptor, so _post_call_success_impl goes straight to tracing
    callback._get_interceptor = AsyncMock(return_value=None)
    callback._interceptor_initialized = True

    data: dict = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o"}
    response = MagicMock()
    response.model = "gpt-4o"
    response.choices = []

    # Simulate _post_call_success_impl running (the primary hook)
    await callback._post_call_success_impl(data, mock_api_key, response)

    # Verify the traced flag was set
    assert data.get("metadata", {}).get("_opensentinel_traced") is True

    # Simulate _log_success_impl running for the same request data
    now = datetime.now(timezone.utc)
    await callback._log_success_impl(data, response, now, now)

    # log_llm_call should have been called exactly once (from _post_call_success_impl)
    mock_tracer.log_llm_call.assert_called_once()


async def test_post_call_guard_skips_tracing_if_already_traced(callback, mock_api_key):
    """_post_call_success_impl does not call log_llm_call when _opensentinel_traced is set."""
    mock_tracer = MagicMock()
    callback._tracer = mock_tracer
    callback._get_interceptor = AsyncMock(return_value=None)
    callback._interceptor_initialized = True

    # Pre-set the guard flag as if some other code path already traced this request
    data: dict = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o",
        "metadata": {"_opensentinel_traced": True},
    }
    response = MagicMock()
    response.model = "gpt-4o"
    response.choices = []

    await callback._post_call_success_impl(data, mock_api_key, response)

    mock_tracer.log_llm_call.assert_not_called()


async def test_log_success_impl_guard_returns_early_if_traced(callback):
    """_log_success_impl returns early when _opensentinel_traced flag is set."""
    from datetime import datetime, timezone

    mock_tracer = MagicMock()
    callback._tracer = mock_tracer
    callback._get_interceptor = AsyncMock(return_value=None)
    callback._interceptor_initialized = True

    kwargs: dict = {
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"_opensentinel_traced": True},
    }
    now = datetime.now(timezone.utc)

    # Should return without calling log_llm_call
    await callback._log_success_impl(kwargs, MagicMock(), now, now)

    mock_tracer.log_llm_call.assert_not_called()


# ---------------------------------------------------------------------------
# UUID request ID tests
# ---------------------------------------------------------------------------


async def test_pre_call_generates_uuid_request_id(callback, mock_api_key, mock_cache):
    """_pre_call_impl generates a UUID and stores it in data metadata."""
    import uuid as uuid_mod
    from opensentinel.core.interceptor.types import InterceptionResult

    data: dict = {"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4"}

    mock_interceptor = MagicMock()
    mock_interceptor.run_pre_call = AsyncMock(
        return_value=InterceptionResult(allowed=True)
    )
    callback._get_interceptor = AsyncMock(return_value=mock_interceptor)
    callback._interceptor_initialized = True

    await callback.async_pre_call_hook(mock_api_key, mock_cache, data, "completion")

    # Verify UUID was stored in metadata
    request_id = data["metadata"]["_opensentinel_request_id"]
    # Should be a valid UUID4 string
    parsed = uuid_mod.UUID(request_id, version=4)
    assert str(parsed) == request_id

    # Verify it was passed to run_pre_call
    call_kwargs = mock_interceptor.run_pre_call.call_args
    assert call_kwargs.kwargs.get("user_request_id") == request_id


async def test_post_call_reuses_uuid_from_pre_call(callback, mock_api_key):
    """_post_call_success_impl reads the UUID stored by _pre_call_impl."""
    import uuid as uuid_mod
    from opensentinel.core.interceptor.types import InterceptionResult

    stored_id = str(uuid_mod.uuid4())
    data: dict = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-4",
        "metadata": {"_opensentinel_request_id": stored_id},
    }
    response = MagicMock()
    response.choices = []

    mock_interceptor = MagicMock()
    mock_interceptor.run_post_call = AsyncMock(
        return_value=InterceptionResult(allowed=True)
    )
    callback._get_interceptor = AsyncMock(return_value=mock_interceptor)
    callback._get_policy_engine = AsyncMock(return_value=None)
    callback._interceptor_initialized = True

    await callback.async_post_call_success_hook(data, mock_api_key, response)

    call_kwargs = mock_interceptor.run_post_call.call_args
    assert call_kwargs.kwargs.get("user_request_id") == stored_id


async def test_post_call_generates_fallback_uuid_when_missing(callback, mock_api_key):
    """_post_call_success_impl generates a new UUID if metadata has no stored ID."""
    import uuid as uuid_mod
    from opensentinel.core.interceptor.types import InterceptionResult

    data: dict = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-4",
    }
    response = MagicMock()
    response.choices = []

    mock_interceptor = MagicMock()
    mock_interceptor.run_post_call = AsyncMock(
        return_value=InterceptionResult(allowed=True)
    )
    callback._get_interceptor = AsyncMock(return_value=mock_interceptor)
    callback._get_policy_engine = AsyncMock(return_value=None)
    callback._interceptor_initialized = True

    await callback.async_post_call_success_hook(data, mock_api_key, response)

    call_kwargs = mock_interceptor.run_post_call.call_args
    request_id = call_kwargs.kwargs.get("user_request_id")
    # Should be a valid UUID4 string
    parsed = uuid_mod.UUID(request_id, version=4)
    assert str(parsed) == request_id
