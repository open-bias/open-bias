"""Generic session store with TTL and hard-cap eviction."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import KeysView
from typing import Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

V = TypeVar("V")


class SessionStore(Generic[V]):
    """Thread-unsafe, ordered session store with TTL + hard-cap eviction.

    Maintains an ``OrderedDict`` of values keyed by session ID alongside a
    parallel ``OrderedDict`` of monotonic timestamps for LRU ordering.

    Parameters
    ----------
    ttl:
        Maximum age (seconds) before a session is considered stale.
    max_sessions:
        Hard upper bound on the number of tracked sessions.
    on_evict:
        Optional callback invoked with ``(key, value)`` whenever a session
        is removed by eviction.  Lets consumers run cleanup (cancel tasks,
        end spans, etc.) without the store knowing about those concerns.
    """

    def __init__(
        self,
        ttl: int,
        max_sessions: int,
        on_evict: Callable[[str, V], None] | None = None,
    ) -> None:
        self._ttl = ttl
        self._max_sessions = max_sessions
        self._on_evict = on_evict
        self._data: OrderedDict[str, V] = OrderedDict()
        self._timestamps: OrderedDict[str, float] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def touch(self, key: str) -> None:
        """Update the last-access timestamp and move *key* to the LRU tail."""
        self._timestamps[key] = time.monotonic()
        self._timestamps.move_to_end(key)
        if key in self._data:
            self._data.move_to_end(key)

    def get(self, key: str) -> V | None:
        """Return the value for *key*, or ``None`` if absent or stale."""
        if key not in self._data:
            return None
        ts = self._timestamps.get(key)
        if ts is not None and time.monotonic() - ts > self._ttl:
            self._evict_one(key)
            return None
        return self._data[key]

    def put(self, key: str, value: V) -> None:
        """Insert or replace *key*, touch it, then auto-evict stale entries."""
        self._data[key] = value
        self.touch(key)
        self.evict_stale()

    def remove(self, key: str) -> V | None:
        """Remove *key* and return its value (``None`` if absent)."""
        self._timestamps.pop(key, None)
        return self._data.pop(key, None)

    def evict_stale(self) -> int:
        """Remove sessions that have exceeded TTL or breach the hard cap.

        Returns the total number of evicted entries.
        """
        evicted = 0
        now = time.monotonic()

        # --- TTL eviction (oldest-first) ---
        stale: list[str] = []
        for sid, ts in self._timestamps.items():
            if now - ts > self._ttl:
                stale.append(sid)
            else:
                break

        for sid in stale:
            self._evict_one(sid)
        evicted += len(stale)

        # --- Hard cap eviction ---
        overflow = len(self._data) - self._max_sessions
        if overflow > 0:
            oldest = list(self._data.keys())[:overflow]
            for sid in oldest:
                self._evict_one(sid)
            evicted += len(oldest)

        return evicted

    def __contains__(self, key: str) -> bool:
        ts = self._timestamps.get(key)
        if ts is not None and time.monotonic() - ts > self._ttl:
            self._evict_one(key)
            return False
        return key in self._data

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def clear(self) -> None:
        """Remove all entries without invoking the eviction callback."""
        self._data.clear()
        self._timestamps.clear()

    def configure(self, ttl: int | None = None, max_sessions: int | None = None) -> None:
        """Update TTL and/or max-sessions at runtime."""
        if ttl is not None:
            self._ttl = ttl
        if max_sessions is not None:
            self._max_sessions = max_sessions

    def __len__(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_one(self, key: str) -> None:
        value = self._data.pop(key, None)
        self._timestamps.pop(key, None)
        if value is not None and self._on_evict is not None:
            self._on_evict(key, value)
