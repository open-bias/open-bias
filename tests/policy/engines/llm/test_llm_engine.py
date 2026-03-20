"""
Tests for LLMPolicyEngine.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from opensentinel.policy.engines.llm import LLMPolicyEngine
from opensentinel.policy.protocols import Decision


@pytest.fixture
def sample_workflow():
    """Sample workflow definition for testing."""
    return {
        "name": "test-workflow",
        "version": "1.0",
        "states": [
            {
                "name": "greeting",
                "is_initial": True,
                "description": "Initial greeting state",
                "classification": {
                    "patterns": ["hello", "hi", "welcome"],
                },
            },
            {
                "name": "identify_issue",
                "description": "Identifying customer issue",
                "classification": {
                    "tool_calls": ["search_kb"],
                    "exemplars": ["Let me look that up", "I'll search our docs"],
                },
            },
            {
                "name": "resolution",
                "is_terminal": True,
                "description": "Issue resolved",
                "classification": {
                    "patterns": ["resolved", "fixed", "done"],
                },
            },
        ],
        "transitions": [
            {"from_state": "greeting", "to_state": "identify_issue"},
            {"from_state": "identify_issue", "to_state": "resolution"},
        ],
        "constraints": [
            {
                "name": "must_greet_first",
                "type": "precedence",
                "trigger": "identify_issue",
                "target": "greeting",
                "severity": "error",
            }
        ],
        "interventions": {},
    }


@pytest.fixture
def engine():
    """Create an uninitialized engine."""
    return LLMPolicyEngine()


class TestInitialization:
    """Tests for engine initialization."""

    async def test_initialize_with_workflow_dict(self, engine, sample_workflow):
        """Test initialization with inline workflow dict."""
        await engine.initialize({"workflow": sample_workflow})
        
        assert engine._initialized
        assert engine.name == "llm:test-workflow"
        assert engine.engine_type == "llm"

    async def test_initialize_missing_config(self, engine):
        """Test initialization fails without config_path or workflow."""
        with pytest.raises(ValueError, match="Either config_path or workflow"):
            await engine.initialize({})

    async def test_engine_type(self, engine):
        """Test engine_type property."""
        assert engine.engine_type == "llm"


class TestEvaluateRequest:
    """Tests for evaluate_request method."""

    async def test_raises_when_uninitialized(self, engine):
        """Uninitialized engine should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await engine.evaluate_request("session1", {"messages": []})

    async def test_allow_when_initialized(self, engine, sample_workflow):
        """Initialized engine should allow requests (pass-through)."""
        await engine.initialize({"workflow": sample_workflow})

        result = await engine.evaluate_request("session1", {"messages": []})
        assert result.decision == Decision.ALLOW


class TestEvaluateResponse:
    """Tests for evaluate_response method."""

    async def test_raises_when_uninitialized(self, engine):
        """Uninitialized engine should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await engine.evaluate_response(
                "session1",
                {"choices": [{"message": {"content": "Hello!"}}]},
                {"messages": []},
            )

    async def test_classify_and_evaluate(self, engine, sample_workflow):
        """Test basic classification and evaluation flow."""
        await engine.initialize({"workflow": sample_workflow})
        
        # Mock the LLM client
        mock_response = [
            {"state_id": "greeting", "confidence": 0.9, "reasoning": "Greeting detected"}
        ]
        engine._llm_client.complete_json = AsyncMock(return_value=mock_response)
        
        # Also mock constraint evaluator
        engine._constraint_evaluator.evaluate = AsyncMock(return_value=[])
        
        result = await engine.evaluate_response(
            "session1",
            {"choices": [{"message": {"content": "Hello! How can I help you?"}}]},
            {"messages": []},
        )
        
        assert result.decision in [Decision.ALLOW, Decision.INTERVENE]
        assert "state" in result.metadata


class TestSessionManagement:
    """Tests for session state management."""

    async def test_get_current_state(self, engine, sample_workflow):
        """Test getting current state."""
        await engine.initialize({"workflow": sample_workflow})
        
        state = await engine.get_current_state("session1")
        assert state == "greeting"  # Initial state

    async def test_get_state_history_empty(self, engine, sample_workflow):
        """Test getting state history for new session."""
        await engine.initialize({"workflow": sample_workflow})
        
        # New session without any evaluation has no history
        history = await engine.get_state_history("session1")
        assert history == []

    async def test_get_valid_next_states(self, engine, sample_workflow):
        """Test getting valid next states."""
        await engine.initialize({"workflow": sample_workflow})
        
        next_states = await engine.get_valid_next_states("session1")
        assert "identify_issue" in next_states

    async def test_reset_session(self, engine, sample_workflow):
        """Test resetting session."""
        await engine.initialize({"workflow": sample_workflow})

        # Create a session by evaluating a response
        await engine.evaluate_response(
            "session1",
            {"choices": [{"message": {"content": "Hello!"}}]},
            {"messages": []},
        )
        assert "session1" in engine._sessions

        # Reset it
        await engine.reset_session("session1")
        assert "session1" not in engine._sessions

    async def test_get_session_state(self, engine, sample_workflow):
        """Test getting session state dict."""
        await engine.initialize({"workflow": sample_workflow})

        # Create session via evaluate_response
        await engine.evaluate_response(
            "session1",
            {"choices": [{"message": {"content": "Hello!"}}]},
            {"messages": []},
        )
        
        state = await engine.get_session_state("session1")
        assert state is not None
        assert state["session_id"] == "session1"
        assert state["workflow_name"] == "test-workflow"

    async def test_get_session_state_nonexistent(self, engine, sample_workflow):
        """Test getting state for nonexistent session."""
        await engine.initialize({"workflow": sample_workflow})
        
        state = await engine.get_session_state("nonexistent")
        assert state is None


class TestShutdown:
    """Tests for engine shutdown."""

    async def test_shutdown_clears_sessions(self, engine, sample_workflow):
        """Test that shutdown clears sessions."""
        await engine.initialize({"workflow": sample_workflow})

        # Create some sessions via evaluate_response
        response = {"choices": [{"message": {"content": "Hi"}}]}
        await engine.evaluate_response("session1", response, {"messages": []})
        await engine.evaluate_response("session2", response, {"messages": []})
        assert len(engine._sessions) == 2

        await engine.shutdown()
        assert len(engine._sessions) == 0
