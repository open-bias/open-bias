"""Unit tests for openbias.policy.registry."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from openbias.policy.protocols import Decision, EngineResult, PolicyEngine
from openbias.policy.registry import PolicyEngineRegistry, register_engine


# ---------------------------------------------------------------------------
# Minimal concrete engine for testing
# ---------------------------------------------------------------------------

class _DummyEngine(PolicyEngine):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def engine_type(self) -> str:
        return "dummy"

    async def initialize(self, config: Dict[str, Any]) -> None:
        pass

    async def evaluate_request(self, session_id, request_data, context=None):
        return EngineResult(decision=Decision.ALLOW)

    async def evaluate_response(self, session_id, response_data, request_data, context=None):
        return EngineResult(decision=Decision.ALLOW)

    async def get_session_state(self, session_id) -> Optional[Dict[str, Any]]:
        return None

    async def reset_session(self, session_id) -> None:
        pass


class _AnotherEngine(_DummyEngine):
    @property
    def name(self):
        return "another"

    @property
    def engine_type(self):
        return "another"


# ---------------------------------------------------------------------------
# PolicyEngineRegistry tests
# ---------------------------------------------------------------------------

class TestPolicyEngineRegistry:
    def setup_method(self):
        """Save and clear registry before each test."""
        self._saved = dict(PolicyEngineRegistry._registry)
        PolicyEngineRegistry.clear()

    def teardown_method(self):
        """Restore original registry after each test."""
        PolicyEngineRegistry._registry.clear()
        PolicyEngineRegistry._registry.update(self._saved)

    # register / get
    def test_register_and_get(self):
        PolicyEngineRegistry.register("dummy", _DummyEngine)
        cls = PolicyEngineRegistry.get("dummy")
        assert cls is _DummyEngine

    def test_get_unknown_returns_none(self):
        assert PolicyEngineRegistry.get("nonexistent") is None

    def test_register_overwrites_existing(self):
        PolicyEngineRegistry.register("e", _DummyEngine)
        PolicyEngineRegistry.register("e", _AnotherEngine)
        assert PolicyEngineRegistry.get("e") is _AnotherEngine

    # create
    def test_create_returns_instance(self):
        PolicyEngineRegistry.register("dummy", _DummyEngine)
        engine = PolicyEngineRegistry.create("dummy")
        assert isinstance(engine, _DummyEngine)

    def test_create_raises_for_unknown(self):
        with pytest.raises(ValueError, match="Unknown policy engine type"):
            PolicyEngineRegistry.create("does_not_exist")

    def test_create_error_message_lists_available(self):
        PolicyEngineRegistry.register("alpha", _DummyEngine)
        with pytest.raises(ValueError, match="alpha"):
            PolicyEngineRegistry.create("nonexistent")

    # create_and_initialize
    async def test_create_and_initialize(self):
        PolicyEngineRegistry.register("dummy", _DummyEngine)
        engine = await PolicyEngineRegistry.create_and_initialize("dummy", {})
        assert isinstance(engine, _DummyEngine)

    async def test_create_and_initialize_unknown_raises(self):
        with pytest.raises(ValueError):
            await PolicyEngineRegistry.create_and_initialize("ghost", {})

    # list_engines
    def test_list_engines_empty(self):
        assert PolicyEngineRegistry.list_engines() == []

    def test_list_engines_returns_registered(self):
        PolicyEngineRegistry.register("a", _DummyEngine)
        PolicyEngineRegistry.register("b", _AnotherEngine)
        engines = PolicyEngineRegistry.list_engines()
        assert "a" in engines
        assert "b" in engines

    # is_registered
    def test_is_registered_true(self):
        PolicyEngineRegistry.register("dummy", _DummyEngine)
        assert PolicyEngineRegistry.is_registered("dummy") is True

    def test_is_registered_false(self):
        assert PolicyEngineRegistry.is_registered("not_there") is False

    # clear
    def test_clear_removes_all(self):
        PolicyEngineRegistry.register("a", _DummyEngine)
        PolicyEngineRegistry.register("b", _AnotherEngine)
        PolicyEngineRegistry.clear()
        assert PolicyEngineRegistry.list_engines() == []


# ---------------------------------------------------------------------------
# register_engine decorator
# ---------------------------------------------------------------------------

class TestRegisterEngineDecorator:
    def setup_method(self):
        self._saved = dict(PolicyEngineRegistry._registry)
        PolicyEngineRegistry.clear()

    def teardown_method(self):
        PolicyEngineRegistry._registry.clear()
        PolicyEngineRegistry._registry.update(self._saved)

    def test_decorator_registers_class(self):
        @register_engine("decorated_test")
        class DecoratedEngine(_DummyEngine):
            @property
            def engine_type(self):
                return "decorated_test"

        assert PolicyEngineRegistry.is_registered("decorated_test")
        assert PolicyEngineRegistry.get("decorated_test") is DecoratedEngine

    def test_decorator_returns_class_unchanged(self):
        @register_engine("pass_through")
        class PTE(_DummyEngine):
            @property
            def engine_type(self):
                return "pass_through"

        assert PTE.__name__ == "PTE"
        instance = PTE()
        assert isinstance(instance, PolicyEngine)

    def test_decorator_allows_instantiation(self):
        @register_engine("inst_test")
        class IE(_DummyEngine):
            @property
            def engine_type(self):
                return "inst_test"

        engine = PolicyEngineRegistry.create("inst_test")
        assert isinstance(engine, IE)
