"""
Workflow state machine.

Tracks agent progress through workflow states with:
- State transition validation
- History tracking for constraint evaluation
- Concurrent session support
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from opensentinel.core.session import SessionStore
from opensentinel.policy.engines.fsm.workflow.schema import State, Transition, WorkflowDefinition

logger = logging.getLogger(__name__)

class TransitionResult(Enum):
    """Result of a transition attempt."""

    SUCCESS = "success"
    INVALID_TRANSITION = "invalid_transition"
    SAME_STATE = "same_state"
    CONSTRAINT_VIOLATED = "constraint_violated"

@dataclass
class StateHistoryEntry:
    """Record of a state in the history."""

    state_name: str
    entered_at: datetime
    exited_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    classification_confidence: float = 0.0
    classification_method: str = "unknown"

@dataclass
class SessionState:
    """
    State tracking for a single session.

    Maintains:
    - Current state
    - Full state history for constraint evaluation
    - Constraint violations
    """

    session_id: str
    workflow_name: str
    current_state: str
    history: list[StateHistoryEntry] = field(default_factory=list)
    constraint_violations: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_state_sequence(self) -> list[str]:
        """Get the sequence of states visited."""
        return [h.state_name for h in self.history]

    def get_current_duration(self) -> float:
        """Get duration in current state (seconds)."""
        if not self.history:
            return 0.0
        current_entry = self.history[-1]
        return (datetime.now(timezone.utc) - current_entry.entered_at).total_seconds()

class WorkflowStateMachine:
    """
    State machine managing workflow execution.

    Thread-safe for concurrent session handling via asyncio locks.

    Example:
        ```python
        from opensentinel.policy.engines.fsm.workflow import WorkflowParser, WorkflowStateMachine

        workflow = WorkflowParser.parse_file("workflow.yaml")
        machine = WorkflowStateMachine(workflow)

        # Get or create session
        session = await machine.get_or_create_session("session-123")

        # Attempt transition
        result, error = await machine.transition("session-123", "identify_issue")
        ```
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        session_ttl: int = 3600,
        max_sessions: int = 10000,
        max_history: int = 1000,
    ):
        self.workflow = workflow
        self._max_history = max_history
        self._sessions: SessionStore[SessionState] = SessionStore(
            ttl=session_ttl,
            max_sessions=max_sessions,
        )
        self._meta_lock = asyncio.Lock()  # protects SessionStore operations
        self._session_locks: dict[str, asyncio.Lock] = {}  # per-session state locks

        # Build lookup tables for fast access
        self._states: dict[str, State] = {s.name: s for s in workflow.states}
        self._transitions: dict[str, list[Transition]] = self._build_transition_map()

        # Find initial state
        initial_states = workflow.get_initial_states()
        self._initial_state = initial_states[0].name if initial_states else None

        logger.info(
            f"WorkflowStateMachine initialized for '{workflow.name}' "
            f"with {len(self._states)} states"
        )

    def _build_transition_map(self) -> dict[str, list[Transition]]:
        """Build from_state -> [transitions] lookup."""
        result: dict[str, list[Transition]] = {}
        for t in self.workflow.transitions:
            if t.from_state not in result:
                result[t.from_state] = []
            result[t.from_state].append(t)

        # Sort by priority (descending)
        for transitions in result.values():
            transitions.sort(key=lambda t: t.priority, reverse=True)

        return result

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session lock for state transitions."""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    async def get_or_create_session(self, session_id: str) -> SessionState:
        """
        Get existing session or create new one.

        Args:
            session_id: Unique session identifier

        Returns:
            SessionState for the session
        """
        async with self._meta_lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                self._sessions.touch(session_id)
                return existing

            if not self._initial_state:
                raise ValueError("Workflow has no initial state")

            session = SessionState(
                session_id=session_id,
                workflow_name=self.workflow.name,
                current_state=self._initial_state,
                history=[
                    StateHistoryEntry(
                        state_name=self._initial_state,
                        entered_at=datetime.now(timezone.utc),
                    )
                ],
            )
            self._sessions.put(session_id, session)
            self._get_session_lock(session_id)  # pre-create per-session lock
            logger.debug(
                f"Created session {session_id} in state '{self._initial_state}'"
            )

            return session

    async def get_session(self, session_id: str) -> SessionState | None:
        """Get session if it exists."""
        session = self._sessions.get(session_id)
        if session is not None:
            self._sessions.touch(session_id)
        return session

    async def transition(
        self,
        session_id: str,
        target_state: str,
        confidence: float = 1.0,
        method: str = "explicit",
        expected_from_state: str | None = None,
    ) -> tuple[TransitionResult, str | None]:
        """
        Attempt to transition to target state.

        Args:
            session_id: Session identifier
            target_state: State to transition to
            confidence: Classification confidence (0-1)
            method: Classification method used
            expected_from_state: If set, validates that the session is still in this
                state before transitioning. Guards against TOCTOU races when
                classification runs without holding the lock.

        Returns:
            Tuple of (TransitionResult, error_message)
        """
        session = await self.get_or_create_session(session_id)

        # Check if target state exists (immutable lookup, safe outside lock)
        if target_state not in self._states:
            return (
                TransitionResult.INVALID_TRANSITION,
                f"Unknown state: {target_state}",
            )

        # Retrieve per-session lock under _meta_lock to prevent races
        # with reset_session which clears lock entries.
        async with self._meta_lock:
            session_lock = self._get_session_lock(session_id)
        async with session_lock:
            current = session.current_state

            # Optimistic concurrency: verify state hasn't changed since classification
            if expected_from_state is not None and current != expected_from_state:
                return (
                    TransitionResult.INVALID_TRANSITION,
                    f"Stale state: expected '{expected_from_state}' "
                    f"but session is now in '{current}'",
                )

            # Same state - no transition needed
            if current == target_state:
                return (TransitionResult.SAME_STATE, None)

            # Terminal states reject all outgoing transitions
            current_state_def = self._states.get(current)
            if current_state_def and current_state_def.is_terminal:
                return (
                    TransitionResult.INVALID_TRANSITION,
                    f"Cannot transition from terminal state '{current}'",
                )

            # Check if transition is valid
            valid_transitions = self._transitions.get(current, [])
            matching = [t for t in valid_transitions if t.to_state == target_state]

            # If no explicit transitions defined from current state, allow any
            if valid_transitions and not matching:
                return (
                    TransitionResult.INVALID_TRANSITION,
                    f"No transition from '{current}' to '{target_state}'",
                )

            # Close current history entry
            if session.history:
                session.history[-1].exited_at = datetime.now(timezone.utc)

            # Add new entry
            session.history.append(
                StateHistoryEntry(
                    state_name=target_state,
                    entered_at=datetime.now(timezone.utc),
                    classification_confidence=confidence,
                    classification_method=method,
                )
            )
            session.current_state = target_state
            session.last_updated = datetime.now(timezone.utc)

            # Trim history to prevent unbounded growth
            if len(session.history) > self._max_history:
                session.history = session.history[-self._max_history:]

        logger.debug(
            f"Session {session_id}: '{current}' -> '{target_state}' "
            f"(confidence={confidence:.2f}, method={method})"
        )

        return (TransitionResult.SUCCESS, None)

    async def get_valid_transitions(self, session_id: str) -> set[str]:
        """
        Get set of valid next states from current state.

        Args:
            session_id: Session identifier

        Returns:
            Set of valid target state names
        """
        session = await self.get_or_create_session(session_id)
        current = session.current_state

        transitions = self._transitions.get(current, [])
        if transitions:
            return {t.to_state for t in transitions}

        # If no explicit transitions, return all non-current states
        return {s for s in self._states.keys() if s != current}

    async def get_state_history(self, session_id: str) -> list[str]:
        """Get state history for a session."""
        session = await self.get_session(session_id)
        if not session:
            return []
        return session.get_state_sequence()

    async def is_in_terminal_state(self, session_id: str) -> bool:
        """Check if session is in a terminal state."""
        session = await self.get_session(session_id)
        if not session:
            return False

        state = self._states.get(session.current_state)
        return state.is_terminal if state else False

    async def reset_session(self, session_id: str) -> None:
        """Reset a session to initial state."""
        async with self._meta_lock:
            self._sessions.remove(session_id)
            self._session_locks.pop(session_id, None)
        # Next access will create fresh session

    def set_eviction_callback(
        self, callback: Callable[[str, SessionState], None]
    ) -> None:
        """Set a callback invoked when sessions are evicted from the store."""
        self._sessions._on_evict = callback

    async def get_session_count(self) -> int:
        """Get number of active sessions."""
        self._sessions.evict_stale()
        return len(self._sessions)
