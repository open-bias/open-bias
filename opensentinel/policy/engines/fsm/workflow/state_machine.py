"""
Workflow state machine.

Tracks agent progress through workflow states with:
- State transition validation
- History tracking for constraint evaluation
- Concurrent session support
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition, State, Transition

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
    exited_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
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

    def __init__(self, workflow: WorkflowDefinition):
        self.workflow = workflow
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

        # Build lookup tables for fast access
        self._states: Dict[str, State] = {s.name: s for s in workflow.states}
        self._transitions: Dict[str, list[Transition]] = self._build_transition_map()

        # Find initial state
        initial_states = workflow.get_initial_states()
        self._initial_state = initial_states[0].name if initial_states else None

        logger.info(
            f"WorkflowStateMachine initialized for '{workflow.name}' "
            f"with {len(self._states)} states"
        )

    def _build_transition_map(self) -> Dict[str, list[Transition]]:
        """Build from_state -> [transitions] lookup."""
        result: Dict[str, list[Transition]] = {}
        for t in self.workflow.transitions:
            if t.from_state not in result:
                result[t.from_state] = []
            result[t.from_state].append(t)

        # Sort by priority (descending)
        for transitions in result.values():
            transitions.sort(key=lambda t: t.priority, reverse=True)

        return result

    async def get_or_create_session(self, session_id: str) -> SessionState:
        """
        Get existing session or create new one.

        Args:
            session_id: Unique session identifier

        Returns:
            SessionState for the session
        """
        async with self._lock:
            if session_id not in self._sessions:
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
                self._sessions[session_id] = session
                logger.debug(
                    f"Created session {session_id} in state '{self._initial_state}'"
                )

            return self._sessions[session_id]

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session if it exists."""
        return self._sessions.get(session_id)

    async def transition(
        self,
        session_id: str,
        target_state: str,
        confidence: float = 1.0,
        method: str = "explicit",
    ) -> Tuple[TransitionResult, Optional[str]]:
        """
        Attempt to transition to target state.

        Args:
            session_id: Session identifier
            target_state: State to transition to
            confidence: Classification confidence (0-1)
            method: Classification method used

        Returns:
            Tuple of (TransitionResult, error_message)
        """
        session = await self.get_or_create_session(session_id)
        current = session.current_state

        # Same state - no transition needed
        if current == target_state:
            return (TransitionResult.SAME_STATE, None)

        # Check if target state exists
        if target_state not in self._states:
            return (
                TransitionResult.INVALID_TRANSITION,
                f"Unknown state: {target_state}",
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

        # Perform transition
        async with self._lock:
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

        logger.debug(
            f"Session {session_id}: '{current}' -> '{target_state}' "
            f"(confidence={confidence:.2f}, method={method})"
        )

        return (TransitionResult.SUCCESS, None)

    async def get_valid_transitions(self, session_id: str) -> Set[str]:
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
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
        # Next access will create fresh session

    async def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)
