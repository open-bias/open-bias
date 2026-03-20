"""
Stateful policy engine base class.

Extends PolicyEngine with state classification capabilities for engines
that track state across requests (FSM, LLM).
"""

from typing import Any
from dataclasses import dataclass, field
from abc import abstractmethod

from opensentinel.policy.protocols import PolicyEngine

@dataclass
class StateClassificationResult:
    """Result of classifying a response to a state/intent."""

    state_name: str
    confidence: float
    method: str
    details: dict[str, Any] = field(default_factory=dict)

class StatefulPolicyEngine(PolicyEngine):
    """
    Policy engine that tracks state across requests.

    Extends PolicyEngine with state classification capabilities.
    Used by FSM and similar state-machine-based engines.
    """

    @abstractmethod
    async def classify_response(
        self,
        session_id: str,
        response_data: Any,
        current_state: str | None = None,
    ) -> StateClassificationResult:
        """
        Classify a response to a state.

        Args:
            session_id: Unique session identifier
            response_data: The LLM response to classify
            current_state: Current state (optional, will be looked up if not provided)

        Returns:
            StateClassificationResult with detected state and confidence
        """
        ...

    @abstractmethod
    async def get_current_state(self, session_id: str) -> str:
        """
        Get current state name for session.

        Args:
            session_id: Unique session identifier

        Returns:
            Current state name
        """
        ...

    @abstractmethod
    async def get_state_history(self, session_id: str) -> list[str]:
        """
        Get state transition history.

        Args:
            session_id: Unique session identifier

        Returns:
            List of state names in chronological order
        """
        ...

    @abstractmethod
    async def get_valid_next_states(self, session_id: str) -> list[str]:
        """
        Get valid next states from current state.

        Args:
            session_id: Unique session identifier

        Returns:
            List of valid state names that can be transitioned to
        """
        ...
