"""Tests for Proxy shutdown cleanup."""

from unittest.mock import AsyncMock, patch

import litellm

from openbias.proxy.server import Proxy


class TestShutdownCallbackCleanup:
    """Verify callbacks are removed from litellm.callbacks on shutdown."""

    def test_create_router_attaches_router_to_callback(self) -> None:
        """The callback keeps a router reference for sync post-call replays."""
        proxy = Proxy()

        with patch("openbias.proxy.server.Router") as mock_router_cls:
            mock_router = mock_router_cls.return_value
            with patch("openbias.proxy.hooks.Callback") as mock_callback_cls:
                mock_callback = mock_callback_cls.return_value

                router = proxy._create_router()

        assert router is mock_router
        mock_callback.attach_router.assert_called_once_with(mock_router)

    async def test_shutdown_removes_callback_from_litellm_callbacks(self) -> None:
        """After shutdown, the callback should no longer be in litellm.callbacks."""
        proxy = Proxy()
        with patch(
            "openbias.proxy.hooks.Callback.initialize", new_callable=AsyncMock
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
        proxy = Proxy()
        with patch(
            "openbias.proxy.hooks.Callback.initialize", new_callable=AsyncMock
        ):
            await proxy.initialize()

        callback = proxy._callback
        assert callback in litellm._async_success_callback

        proxy._shutdown()

        assert callback not in litellm._async_success_callback

    async def test_callbacks_do_not_accumulate_on_reinit(self) -> None:
        """Re-initializing the proxy should not leave stale callbacks."""
        proxy = Proxy()

        with patch(
            "openbias.proxy.hooks.Callback.initialize", new_callable=AsyncMock
        ):
            await proxy.initialize()

        initial_count = len(litellm.callbacks)
        proxy._shutdown()

        with patch(
            "openbias.proxy.hooks.Callback.initialize", new_callable=AsyncMock
        ):
            proxy._hooks_registered = False
            await proxy.initialize()

        assert len(litellm.callbacks) == initial_count

        # Clean up
        proxy._shutdown()

    async def test_shutdown_idempotent(self) -> None:
        """Calling _shutdown() twice should not raise."""
        proxy = Proxy()
        with patch(
            "openbias.proxy.hooks.Callback.initialize", new_callable=AsyncMock
        ):
            await proxy.initialize()

        proxy._shutdown()
        proxy._shutdown()  # second call should be a no-op
        assert proxy._callback is None
