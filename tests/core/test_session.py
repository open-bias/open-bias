"""Tests for opensentinel.core.session.SessionStore."""

from __future__ import annotations

import time
from unittest.mock import patch

from opensentinel.core.session import SessionStore


class TestSessionStoreBasics:
    def test_put_and_get(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=60, max_sessions=100)
        store.put("s1", "value1")
        assert store.get("s1") == "value1"

    def test_get_missing_returns_none(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=60, max_sessions=100)
        assert store.get("missing") is None

    def test_contains(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=60, max_sessions=100)
        store.put("s1", "v")
        assert "s1" in store
        assert "s2" not in store

    def test_remove(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=60, max_sessions=100)
        store.put("s1", "v1")
        removed = store.remove("s1")
        assert removed == "v1"
        assert "s1" not in store

    def test_remove_missing_returns_none(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=60, max_sessions=100)
        assert store.remove("nope") is None

    def test_keys(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=60, max_sessions=100)
        store.put("a", "1")
        store.put("b", "2")
        assert set(store.keys()) == {"a", "b"}

    def test_len(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=60, max_sessions=100)
        assert len(store) == 0
        store.put("a", "1")
        assert len(store) == 1

    def test_clear(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=60, max_sessions=100)
        store.put("a", "1")
        store.put("b", "2")
        store.clear()
        assert len(store) == 0
        assert store.get("a") is None

    def test_put_overwrites(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=60, max_sessions=100)
        store.put("s1", "old")
        store.put("s1", "new")
        assert store.get("s1") == "new"
        assert len(store) == 1


class TestSessionStoreTTLEviction:
    def test_ttl_evicts_stale_entries(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=10, max_sessions=100)
        base = time.monotonic()

        with patch("opensentinel.core.session.time") as mock_time:
            mock_time.monotonic.return_value = base
            store.put("old1", "v1")
            store.put("old2", "v2")

            # Advance past TTL
            mock_time.monotonic.return_value = base + 15
            store.put("fresh", "v3")

        assert "old1" not in store
        assert "old2" not in store
        assert "fresh" in store

    def test_ttl_preserves_fresh_entries(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=10, max_sessions=100)
        base = time.monotonic()

        with patch("opensentinel.core.session.time") as mock_time:
            mock_time.monotonic.return_value = base
            store.put("s1", "v1")

            mock_time.monotonic.return_value = base + 5
            store.put("s2", "v2")

            # Only s1 is stale
            mock_time.monotonic.return_value = base + 12
            evicted = store.evict_stale()

        assert evicted == 1
        assert "s1" not in store
        assert "s2" in store


class TestSessionStoreHardCapEviction:
    def test_hard_cap_evicts_oldest(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=3600, max_sessions=3)
        store.put("s1", "v1")
        store.put("s2", "v2")
        store.put("s3", "v3")
        # This should evict s1 (oldest)
        store.put("s4", "v4")

        assert "s1" not in store
        assert len(store) == 3

    def test_touch_prevents_eviction(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=3600, max_sessions=3)
        store.put("s1", "v1")
        store.put("s2", "v2")
        store.put("s3", "v3")

        # Touch s1, making s2 the oldest
        store.touch("s1")
        store.put("s4", "v4")

        assert "s2" not in store
        assert "s1" in store
        assert len(store) == 3


class TestSessionStoreOnEvict:
    def test_on_evict_called_for_ttl(self) -> None:
        evicted: list[tuple[str, str]] = []
        store: SessionStore[str] = SessionStore(
            ttl=10, max_sessions=100, on_evict=lambda k, v: evicted.append((k, v))
        )
        base = time.monotonic()

        with patch("opensentinel.core.session.time") as mock_time:
            mock_time.monotonic.return_value = base
            store.put("s1", "v1")

            mock_time.monotonic.return_value = base + 15
            store.evict_stale()

        assert evicted == [("s1", "v1")]

    def test_on_evict_called_for_hard_cap(self) -> None:
        evicted: list[tuple[str, str]] = []
        store: SessionStore[str] = SessionStore(
            ttl=3600, max_sessions=2, on_evict=lambda k, v: evicted.append((k, v))
        )
        store.put("s1", "v1")
        store.put("s2", "v2")
        store.put("s3", "v3")

        assert ("s1", "v1") in evicted

    def test_on_evict_not_called_on_remove(self) -> None:
        evicted: list[tuple[str, str]] = []
        store: SessionStore[str] = SessionStore(
            ttl=3600, max_sessions=100, on_evict=lambda k, v: evicted.append((k, v))
        )
        store.put("s1", "v1")
        store.remove("s1")
        assert evicted == []

    def test_on_evict_not_called_on_clear(self) -> None:
        evicted: list[tuple[str, str]] = []
        store: SessionStore[str] = SessionStore(
            ttl=3600, max_sessions=100, on_evict=lambda k, v: evicted.append((k, v))
        )
        store.put("s1", "v1")
        store.clear()
        assert evicted == []


class TestSessionStoreLRUOrdering:
    def test_put_maintains_lru_order(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=3600, max_sessions=100)
        store.put("s1", "v1")
        store.put("s2", "v2")
        store.put("s3", "v3")
        assert list(store.keys()) == ["s1", "s2", "s3"]

    def test_touch_moves_to_end(self) -> None:
        store: SessionStore[str] = SessionStore(ttl=3600, max_sessions=100)
        store.put("s1", "v1")
        store.put("s2", "v2")
        store.put("s3", "v3")
        store.touch("s1")
        assert list(store.keys()) == ["s2", "s3", "s1"]
