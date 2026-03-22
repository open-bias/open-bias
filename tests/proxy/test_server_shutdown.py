"""Tests for SentinelProxy shutdown cleanup."""

from unittest.mock import AsyncMock, patch

import litellm

from opensentinel.proxy.server import SentinelProxy


class TestShutdownCallbackCleanup:
    """Verify callbacks are removed from litellm.callbacks on shutdown."""

    async def test_shutdown_removes_callback_from_litellm_callbacks(self) -> None:
        """After shutdown, the callback should no longer be in litellm.callbacks."""
        proxy = SentinelProxy()
        with patch(
            "opensentinel.proxy.hooks.SentinelCallback.initialize", new_callable=AsyncMock
        ):
            await proxy.initialize()

        assert proxy._callback is not None
        callback = proxy._callback
        assert callback in litellm.callbacks

        proxy._shutdown()

        assert callback not in litellm.callbacks
        assert proxy._callback is None

    async def test_shutdown_removes_callback_from_async_success_list(self) -> None:
        """After shutdown, the callback should not remain in _async_success_callback."""
        proxy = SentinelProxy()
        with patch(
            "opensentinel.proxy.hooks.SentinelCallback.initialize", new_callable=AsyncMock
        ):
            await proxy.initialize()

        callback = proxy._callback
        assert callback in litellm._async_success_callback

        proxy._shutdown()

        assert callback not in litellm._async_success_callback

    async def test_callbacks_do_not_accumulate_on_reinit(self) -> None:
        """Re-initializing the proxy should not leave stale callbacks."""
        proxy = SentinelProxy()

        with patch(
            "opensentinel.proxy.hooks.SentinelCallback.initialize", new_callable=AsyncMock
        ):
            await proxy.initialize()

        initial_count = len(litellm.callbacks)
        proxy._shutdown()

        with patch(
            "opensentinel.proxy.hooks.SentinelCallback.initialize", new_callable=AsyncMock
        ):
            proxy._hooks_registered = False
            await proxy.initialize()

        assert len(litellm.callbacks) == initial_count

        # Clean up
        proxy._shutdown()

    async def test_shutdown_idempotent(self) -> None:
        """Calling _shutdown() twice should not raise."""
        proxy = SentinelProxy()
        with patch(
            "opensentinel.proxy.hooks.SentinelCallback.initialize", new_callable=AsyncMock
        ):
            await proxy.initialize()

        proxy._shutdown()
        proxy._shutdown()  # second call should be a no-op
