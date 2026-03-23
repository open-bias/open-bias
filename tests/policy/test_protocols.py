"""Unit tests for openbias.policy.protocols."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from openbias.policy.protocols import (
    Decision,
    EngineResult,
    PolicyEngine,
    require_initialized,
)
from openbias.policy.engines.stateful import (
    StateClassificationResult,
    StatefulPolicyEngine,
)


# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------

class TestDecision:
    def test_values(self):
        assert Decision.ALLOW.value == "allow"
        assert Decision.BLOCK.value == "block"
        assert Decision.INTERVENE.value == "intervene"

    def test_all_members(self):
        members = {d.name for d in Decision}
        assert members == {"ALLOW", "BLOCK", "INTERVENE"}

    def test_equality(self):
        assert Decision.ALLOW == Decision.ALLOW
        assert Decision.ALLOW != Decision.BLOCK


# ---------------------------------------------------------------------------
# EngineResult
# ---------------------------------------------------------------------------

class TestEngineResult:
    def test_defaults(self):
        r = EngineResult(decision=Decision.ALLOW)
        assert r.decision == Decision.ALLOW
        assert r.message is None
        assert r.metadata == {}
        assert r.modified_messages is None

    def test_with_all_fields(self):
        r = EngineResult(
            decision=Decision.BLOCK,
            message="blocked",
            metadata={"reason": "policy"},
            modified_messages=[{"role": "system", "content": "stop"}],
        )
        assert r.decision == Decision.BLOCK
        assert r.message == "blocked"
        assert r.metadata == {"reason": "policy"}
        assert r.modified_messages == [{"role": "system", "content": "stop"}]

    def test_metadata_is_independent(self):
        r1 = EngineResult(decision=Decision.ALLOW)
        r2 = EngineResult(decision=Decision.ALLOW)
        r1.metadata["key"] = "val"
        assert "key" not in r2.metadata


# ---------------------------------------------------------------------------
# StateClassificationResult
# ---------------------------------------------------------------------------

class TestStateClassificationResult:
    def test_basic(self):
        r = StateClassificationResult(
            state_name="greeting",
            confidence=0.95,
            method="semantic",
        )
        assert r.state_name == "greeting"
        assert r.confidence == 0.95
        assert r.method == "semantic"
        assert r.details == {}

    def test_with_details(self):
        r = StateClassificationResult(
            state_name="verify",
            confidence=0.7,
            method="keyword",
            details={"score": 0.7},
        )
        assert r.details == {"score": 0.7}


# ---------------------------------------------------------------------------
# require_initialized
# ---------------------------------------------------------------------------

class TestRequireInitialized:
    """Tests for the @require_initialized decorator."""

    async def test_raises_when_not_initialized(self):
        class Dummy:
            _initialized = False

            @require_initialized
            async def do_thing(self):
                return "done"

        d = Dummy()
        with pytest.raises(RuntimeError, match="not initialized"):
            await d.do_thing()

    async def test_passes_when_initialized(self):
        class Dummy:
            _initialized = True

            @require_initialized
            async def do_thing(self):
                return "done"

        d = Dummy()
        result = await d.do_thing()
        assert result == "done"

    async def test_missing_initialized_attr_raises(self):
        class Dummy:
            @require_initialized
            async def do_thing(self):
                return "done"

        d = Dummy()
        with pytest.raises(RuntimeError, match="not initialized"):
            await d.do_thing()

    async def test_preserves_function_name(self):
        class Dummy:
            _initialized = True

            @require_initialized
            async def my_method(self):
                return 42

        assert Dummy.my_method.__name__ == "my_method"

    async def test_passes_args_and_kwargs(self):
        class Dummy:
            _initialized = True

            @require_initialized
            async def add(self, x, y=0):
                return x + y

        d = Dummy()
        assert await d.add(3, y=4) == 7

    async def test_sync_inner_method_still_works(self):
        """Wrapping a sync method inside the async wrapper still returns correctly."""
        class Dummy:
            _initialized = True

            @require_initialized
            async def compute(self):
                return 99

        d = Dummy()
        assert await d.compute() == 99


# ---------------------------------------------------------------------------
# PolicyEngine abstract interface
# ---------------------------------------------------------------------------

class TestPolicyEngineInterface:
    """Verify that PolicyEngine cannot be instantiated directly."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            PolicyEngine()  # type: ignore[abstract]

    def test_concrete_subclass(self):
        class ConcreteEngine(PolicyEngine):
            @property
            def name(self):
                return "concrete"

            @property
            def engine_type(self):
                return "test"

            async def initialize(self, config):
                pass

            async def evaluate_request(self, session_id, request_data, context=None):
                return EngineResult(decision=Decision.ALLOW)

            async def evaluate_response(self, session_id, response_data, request_data, context=None):
                return EngineResult(decision=Decision.ALLOW)

            async def get_session_state(self, session_id):
                return None

            async def reset_session(self, session_id):
                pass

        engine = ConcreteEngine()
        assert engine.name == "concrete"
        assert engine.engine_type == "test"

    async def test_shutdown_default_noop(self):
        class MinimalEngine(PolicyEngine):
            @property
            def name(self):
                return "m"

            @property
            def engine_type(self):
                return "t"

            async def initialize(self, config):
                pass

            async def evaluate_request(self, session_id, request_data, context=None):
                return EngineResult(decision=Decision.ALLOW)

            async def evaluate_response(self, session_id, response_data, request_data, context=None):
                return EngineResult(decision=Decision.ALLOW)

            async def get_session_state(self, session_id):
                return None

            async def reset_session(self, session_id):
                pass

        e = MinimalEngine()
        # Default shutdown should not raise
        await e.shutdown()

    def test_get_compiler_returns_none_by_default(self):
        class MinimalEngine(PolicyEngine):
            @property
            def name(self):
                return "m"

            @property
            def engine_type(self):
                return "t"

            async def initialize(self, config):
                pass

            async def evaluate_request(self, session_id, request_data, context=None):
                return EngineResult(decision=Decision.ALLOW)

            async def evaluate_response(self, session_id, response_data, request_data, context=None):
                return EngineResult(decision=Decision.ALLOW)

            async def get_session_state(self, session_id):
                return None

            async def reset_session(self, session_id):
                pass

        assert MinimalEngine().get_compiler() is None
