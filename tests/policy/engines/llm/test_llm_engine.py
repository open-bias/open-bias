"""
Tests for LLMPolicyEngine.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openbias.policy.engines.llm import LLMPolicyEngine
from openbias.policy.engines.llm.models import ConfidenceTier, SessionContext
from openbias.policy.protocols import EvaluationStatus


@pytest.fixture
def sample_workflow():
    """Sample workflow definition for testing."""
    return {
        "name": "test-workflow",
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
            }
        ],
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
        assert result.status == EvaluationStatus.ALLOW


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
        
        assert result.status == EvaluationStatus.ALLOW
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


class TestCriticalViolationDecision:
    """Tests that critical violations no longer produce BLOCK directly."""

    async def test_critical_violation_returns_violation_status(self, engine, sample_workflow):
        """Critical constraint violation should produce VIOLATION status.

        The engine is a pure evaluator — it reports violations with diagnostic
        reasons and lets the interceptor decide enforcement (block/intervene).
        """
        await engine.initialize({"workflow": sample_workflow})

        # Mock the LLM client
        engine._llm_client.complete_json = AsyncMock(
            return_value=[{"state_id": "greeting", "confidence": 0.9, "reasoning": "ok"}]
        )

        # Inject a critical violation via the constraint evaluator
        critical_cv = MagicMock()
        critical_cv.violated = True
        critical_cv.constraint_id = "critical_rule"
        critical_cv.severity = "critical"
        critical_cv.evidence = "critical violation occurred"
        critical_cv.confidence = 1.0
        engine._constraint_evaluator.evaluate = AsyncMock(return_value=[critical_cv])

        result = await engine.evaluate_response(
            "session_crit",
            {"choices": [{"message": {"content": "Hello!"}}]},
            {"messages": []},
        )

        assert result.status == EvaluationStatus.VIOLATION
        assert len(result.violations) == 1
        assert result.violations[0].rule_id == "critical_rule"
        assert result.violations[0].severity == "critical"
        assert result.violations[0].reason == "critical violation occurred"

    async def test_max_severity_metadata_critical(self, engine, sample_workflow):
        """max_severity metadata should reflect the highest severity violation."""
        await engine.initialize({"workflow": sample_workflow})

        engine._llm_client.complete_json = AsyncMock(
            return_value=[{"state_id": "greeting", "confidence": 0.9, "reasoning": "ok"}]
        )

        critical_cv = MagicMock()
        critical_cv.violated = True
        critical_cv.constraint_id = "critical_rule"
        critical_cv.severity = "critical"
        critical_cv.evidence = "critical violation occurred"
        critical_cv.confidence = 1.0

        warning_cv = MagicMock()
        warning_cv.violated = True
        warning_cv.constraint_id = "warn_rule"
        warning_cv.severity = "warning"
        warning_cv.evidence = "minor issue"
        warning_cv.confidence = 0.8

        engine._constraint_evaluator.evaluate = AsyncMock(return_value=[warning_cv, critical_cv])

        result = await engine.evaluate_response(
            "session_meta",
            {"choices": [{"message": {"content": "Hello!"}}]},
            {"messages": []},
        )

        assert result.metadata["max_severity"] == "critical"

    async def test_max_severity_metadata_none_when_no_violations(self, engine, sample_workflow):
        """max_severity should be None when there are no violations."""
        await engine.initialize({"workflow": sample_workflow})

        engine._llm_client.complete_json = AsyncMock(
            return_value=[{"state_id": "greeting", "confidence": 0.9, "reasoning": "ok"}]
        )
        engine._constraint_evaluator.evaluate = AsyncMock(return_value=[])

        result = await engine.evaluate_response(
            "session_no_viol",
            {"choices": [{"message": {"content": "Hello!"}}]},
            {"messages": []},
        )

        assert result.metadata["max_severity"] is None

    async def test_max_severity_metadata_error(self, engine, sample_workflow):
        """max_severity should be 'error' when highest violation is error severity."""
        await engine.initialize({"workflow": sample_workflow})

        engine._llm_client.complete_json = AsyncMock(
            return_value=[{"state_id": "greeting", "confidence": 0.9, "reasoning": "ok"}]
        )

        error_cv = MagicMock()
        error_cv.violated = True
        error_cv.constraint_id = "error_rule"
        error_cv.severity = "error"
        error_cv.evidence = "error violation"
        error_cv.confidence = 0.9

        engine._constraint_evaluator.evaluate = AsyncMock(return_value=[error_cv])

        result = await engine.evaluate_response(
            "session_err",
            {"choices": [{"message": {"content": "Hello!"}}]},
            {"messages": []},
        )

        assert result.metadata["max_severity"] == "error"


class TestGetStateSequence:
    """Tests for SessionContext.get_state_sequence()."""

    def test_empty_history_returns_current_state(self):
        """Empty history returns [current_state]."""
        ctx = SessionContext(session_id="s1", workflow_name="wf", current_state="A")
        assert ctx.get_state_sequence() == ["A"]

    def test_empty_history_no_current_state_returns_empty(self):
        """Empty history with no current_state returns []."""
        ctx = SessionContext(session_id="s1", workflow_name="wf", current_state="")
        assert ctx.get_state_sequence() == []

    def test_single_transition(self):
        """Single A→B transition returns [A, B]."""
        ctx = SessionContext(session_id="s1", workflow_name="wf", current_state="A")
        ctx.record_transition("A", "B", 0.9, ConfidenceTier.CONFIDENT, 0.1)
        assert ctx.get_state_sequence() == ["A", "B"]

    def test_multiple_transitions(self):
        """A→B→C transitions return [A, B, C]."""
        ctx = SessionContext(session_id="s1", workflow_name="wf", current_state="A")
        ctx.record_transition("A", "B", 0.9, ConfidenceTier.CONFIDENT, 0.1)
        ctx.record_transition("B", "C", 0.85, ConfidenceTier.CONFIDENT, 0.15)
        assert ctx.get_state_sequence() == ["A", "B", "C"]


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

    async def test_shutdown_resets_initialized_flag(self, engine, sample_workflow):
        """Shutdown should reset _initialized to False."""
        await engine.initialize({"workflow": sample_workflow})
        assert engine._initialized
        await engine.shutdown()
        assert not engine._initialized

    async def test_evaluate_response_raises_after_shutdown(self, engine, sample_workflow):
        """evaluate_response should raise RuntimeError after shutdown."""
        await engine.initialize({"workflow": sample_workflow})
        await engine.shutdown()
        with pytest.raises(RuntimeError, match="not initialized"):
            await engine.evaluate_response(
                "s1",
                {"choices": [{"message": {"content": "Hello!"}}]},
                {"messages": []},
            )
