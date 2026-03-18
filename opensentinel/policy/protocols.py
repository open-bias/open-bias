"""
Core protocol definitions for policy engines.

These protocols define the contract that all policy engines must implement,
enabling pluggable policy evaluation while maintaining a consistent API.
"""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import functools
import inspect

if TYPE_CHECKING:
    from opensentinel.policy.compiler.protocol import PolicyCompiler


class Decision(Enum):
    """Result of policy evaluation."""

    ALLOW = "allow"
    BLOCK = "block"
    INTERVENE = "intervene"


@dataclass
class EngineResult:
    """Result returned by a policy engine evaluation."""

    decision: Decision
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    modified_messages: Optional[List[Dict[str, Any]]] = None



@dataclass
class StateClassificationResult:
    """Result of classifying a response to a state/intent."""

    state_name: str
    confidence: float
    method: str
    details: Dict[str, Any] = field(default_factory=dict)


class PolicyEngine(ABC):
    """
    Base class for all policy engines.

    Policy engines evaluate requests/responses against configured policies
    and determine what interventions (if any) are needed.

    Implementations should be registered using the @register_engine decorator.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this policy engine instance."""
        ...

    @property
    @abstractmethod
    def engine_type(self) -> str:
        """Type identifier (e.g., 'fsm', 'nemo', 'judge')."""
        ...

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the engine with configuration.

        Args:
            config: Engine-specific configuration dictionary
        """
        ...

    @abstractmethod
    async def evaluate_request(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EngineResult:
        """
        Evaluate an incoming request against policies.

        Called BEFORE the LLM call. Can allow, intervene, or block the request.

        Args:
            session_id: Unique session identifier
            request_data: The LLM request data (messages, model, etc.)
            context: Additional context for evaluation

        Returns:
            EngineResult with decision and optional message
        """
        ...

    @abstractmethod
    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EngineResult:
        """
        Evaluate an LLM response against policies.

        Called AFTER the LLM call. Records violations for potential
        intervention on next call.

        Args:
            session_id: Unique session identifier
            response_data: The LLM response
            request_data: The original request data
            context: Additional context for evaluation

        Returns:
            EngineResult with decision and optional message
        """
        ...

    @abstractmethod
    async def get_session_state(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get current session state for debugging/tracing.

        Args:
            session_id: Unique session identifier

        Returns:
            Session state dictionary or None if session doesn't exist
        """
        ...

    @abstractmethod
    async def reset_session(self, session_id: str) -> None:
        """
        Reset session state.

        Args:
            session_id: Unique session identifier
        """
        ...

    async def shutdown(self) -> None:
        """
        Cleanup resources.

        Override in subclasses that need cleanup.
        """
        pass

    def get_compiler(self) -> Optional["PolicyCompiler"]:
        """Get the compiler for this engine type.

        Override in subclasses that have a dedicated compiler.
        Returns None by default (engine has no compiler).
        """
        return None


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
        current_state: Optional[str] = None,
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
    async def get_state_history(self, session_id: str) -> List[str]:
        """
        Get state transition history.

        Args:
            session_id: Unique session identifier

        Returns:
            List of state names in chronological order
        """
        ...

    @abstractmethod
    async def get_valid_next_states(self, session_id: str) -> List[str]:
        """
        Get valid next states from current state.

        Args:
            session_id: Unique session identifier

        Returns:
            List of valid state names that can be transitioned to
        """
        ...


def require_initialized(method):
    """
    Decorator to ensure engine is initialized before method call.
    Raises RuntimeError if self._initialized is False.
    """
    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        if not getattr(self, "_initialized", False):
            raise RuntimeError(f"{type(self).__name__} not initialized. Call initialize() first.")
        
        # Check if original method is coroutine
        if inspect.iscoroutinefunction(method):
            return await method(self, *args, **kwargs)
        return method(self, *args, **kwargs)
    return wrapper



