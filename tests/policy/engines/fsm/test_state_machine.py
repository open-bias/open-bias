"""Tests for workflow state machine."""

import asyncio
from datetime import datetime, timezone

import pytest

from opensentinel.policy.engines.fsm.workflow.state_machine import (
    SessionState,
    TransitionResult,
    WorkflowStateMachine,
)


class TestWorkflowStateMachine:
    """Tests for WorkflowStateMachine."""

    @pytest.fixture
    def machine(self, simple_workflow):
        """Create a state machine for testing."""
        return WorkflowStateMachine(simple_workflow)

    async def test_create_session(self, machine):
        """Test creating a new session."""
        session = await machine.get_or_create_session("test-session")

        assert session.session_id == "test-session"
        assert session.current_state == "start"
        assert len(session.history) == 1

    async def test_get_existing_session(self, machine):
        """Test getting an existing session."""
        session1 = await machine.get_or_create_session("test-session")
        session2 = await machine.get_or_create_session("test-session")

        assert session1 is session2

    async def test_valid_transition(self, machine):
        """Test a valid state transition."""
        await machine.get_or_create_session("test-session")

        result, error = await machine.transition("test-session", "middle")

        assert result == TransitionResult.SUCCESS
        assert error is None

        session = await machine.get_session("test-session")
        assert session.current_state == "middle"

    async def test_invalid_transition(self, machine):
        """Test an invalid state transition."""
        await machine.get_or_create_session("test-session")

        # Try to jump directly to end (should go through middle first)
        result, error = await machine.transition("test-session", "end")

        assert result == TransitionResult.INVALID_TRANSITION
        assert error is not None

    async def test_same_state_transition(self, machine):
        """Test transitioning to the same state."""
        await machine.get_or_create_session("test-session")

        result, error = await machine.transition("test-session", "start")

        assert result == TransitionResult.SAME_STATE

    async def test_history_tracking(self, machine):
        """Test that state history is tracked."""
        await machine.get_or_create_session("test-session")
        await machine.transition("test-session", "middle")
        await machine.transition("test-session", "end")

        history = await machine.get_state_history("test-session")

        assert history == ["start", "middle", "end"]

    async def test_valid_transitions_from_state(self, machine):
        """Test getting valid transitions from current state."""
        await machine.get_or_create_session("test-session")

        valid = await machine.get_valid_transitions("test-session")

        assert "middle" in valid

    async def test_terminal_state_detection(self, machine):
        """Test detecting terminal state."""
        await machine.get_or_create_session("test-session")
        await machine.transition("test-session", "middle")
        await machine.transition("test-session", "end")

        is_terminal = await machine.is_in_terminal_state("test-session")

        assert is_terminal is True

    async def test_reset_session(self, machine):
        """Test resetting a session."""
        await machine.get_or_create_session("test-session")
        await machine.transition("test-session", "middle")

        await machine.reset_session("test-session")

        session = await machine.get_or_create_session("test-session")
        assert session.current_state == "start"
        assert len(session.history) == 1


class TestTerminalStateEnforcement:
    """Terminal states must reject all outgoing transitions."""

    @pytest.fixture
    def machine(self, simple_workflow):
        return WorkflowStateMachine(simple_workflow)

    async def test_terminal_state_rejects_transition(self, machine):
        """Cannot transition out of a terminal state."""
        await machine.get_or_create_session("s1")
        await machine.transition("s1", "middle")
        await machine.transition("s1", "end")

        # "end" is terminal — any transition should be rejected
        result, error = await machine.transition("s1", "start")

        assert result == TransitionResult.INVALID_TRANSITION
        assert "terminal" in error.lower()

    async def test_terminal_state_same_state_allowed(self, machine):
        """Same-state transition in terminal state returns SAME_STATE, not rejected."""
        await machine.get_or_create_session("s1")
        await machine.transition("s1", "middle")
        await machine.transition("s1", "end")

        result, _ = await machine.transition("s1", "end")

        assert result == TransitionResult.SAME_STATE

    async def test_terminal_state_preserves_history(self, machine):
        """Rejected transition from terminal state does not alter history."""
        await machine.get_or_create_session("s1")
        await machine.transition("s1", "middle")
        await machine.transition("s1", "end")

        await machine.transition("s1", "start")

        history = await machine.get_state_history("s1")
        assert history == ["start", "middle", "end"]

    async def test_non_terminal_state_allows_transition(self, machine):
        """Non-terminal states continue to allow valid transitions."""
        await machine.get_or_create_session("s1")

        result, _ = await machine.transition("s1", "middle")
        assert result == TransitionResult.SUCCESS


class TestSessionState:
    """Tests for SessionState dataclass."""

    def test_get_state_sequence(self):
        """Test getting state sequence from history."""
        from opensentinel.policy.engines.fsm.workflow.state_machine import StateHistoryEntry

        session = SessionState(
            session_id="test",
            workflow_name="test",
            current_state="end",
            history=[
                StateHistoryEntry(state_name="start", entered_at=datetime.now(timezone.utc)),
                StateHistoryEntry(state_name="middle", entered_at=datetime.now(timezone.utc)),
                StateHistoryEntry(state_name="end", entered_at=datetime.now(timezone.utc)),
            ],
        )

        sequence = session.get_state_sequence()
        assert sequence == ["start", "middle", "end"]


class TestPerSessionLocks:
    """Per-session locks allow concurrent transitions on different sessions."""

    @pytest.fixture
    def machine(self, simple_workflow):
        return WorkflowStateMachine(simple_workflow)

    async def test_different_sessions_not_serialized(self, machine):
        """Transitions on different sessions can proceed concurrently."""
        await machine.get_or_create_session("s1")
        await machine.get_or_create_session("s2")

        # Both transitions should succeed without blocking each other
        results = await asyncio.gather(
            machine.transition("s1", "middle"),
            machine.transition("s2", "middle"),
        )

        assert results[0] == (TransitionResult.SUCCESS, None)
        assert results[1] == (TransitionResult.SUCCESS, None)

    async def test_session_lock_created_on_session_creation(self, machine):
        """Per-session lock is created when session is created."""
        await machine.get_or_create_session("s1")
        assert "s1" in machine._session_locks

    async def test_session_lock_cleaned_on_reset(self, machine):
        """Per-session lock is removed when session is reset."""
        await machine.get_or_create_session("s1")
        assert "s1" in machine._session_locks

        await machine.reset_session("s1")
        assert "s1" not in machine._session_locks

    async def test_meta_lock_protects_session_store(self, machine):
        """Concurrent get_or_create_session calls don't create duplicates."""
        results = await asyncio.gather(
            machine.get_or_create_session("s1"),
            machine.get_or_create_session("s1"),
        )
        # Both should return the same session object
        assert results[0] is results[1]


class TestExpectedFromState:
    """Optimistic concurrency: expected_from_state guards against TOCTOU races."""

    @pytest.fixture
    def machine(self, simple_workflow):
        return WorkflowStateMachine(simple_workflow)

    async def test_expected_from_state_matches(self, machine):
        """Transition succeeds when expected_from_state matches current state."""
        await machine.get_or_create_session("s1")

        result, error = await machine.transition(
            "s1", "middle", expected_from_state="start"
        )

        assert result == TransitionResult.SUCCESS
        assert error is None

    async def test_expected_from_state_mismatch(self, machine):
        """Transition rejected when expected_from_state differs from current."""
        await machine.get_or_create_session("s1")
        await machine.transition("s1", "middle")

        # Caller thinks we're still in "start", but we moved to "middle"
        result, error = await machine.transition(
            "s1", "end", expected_from_state="start"
        )

        assert result == TransitionResult.INVALID_TRANSITION
        assert "Stale state" in error
        assert "'start'" in error
        assert "'middle'" in error

    async def test_expected_from_state_none_skips_check(self, machine):
        """When expected_from_state is None, no staleness check is performed."""
        await machine.get_or_create_session("s1")

        # Should work without expected_from_state (backward compat)
        result, error = await machine.transition("s1", "middle")
        assert result == TransitionResult.SUCCESS

    async def test_stale_state_preserves_session(self, machine):
        """Stale state rejection doesn't alter session state."""
        await machine.get_or_create_session("s1")
        await machine.transition("s1", "middle")

        # Stale transition attempt
        await machine.transition("s1", "end", expected_from_state="start")

        session = await machine.get_session("s1")
        assert session.current_state == "middle"
        assert session.get_state_sequence() == ["start", "middle"]
