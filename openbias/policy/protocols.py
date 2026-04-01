"""
Core protocol definitions for policy engines.

These protocols define the contract that all policy engines must implement,
enabling pluggable policy evaluation while maintaining a consistent API.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import functools

if TYPE_CHECKING:
    from openbias.policy.compiler.protocol import PolicyCompiler

class Decision(Enum):
    """Result of policy evaluation."""

    ALLOW = "allow"
    BLOCK = "block"
    INTERVENE = "intervene"


# ---------------------------------------------------------------------------
# Unified evaluation contract
# ---------------------------------------------------------------------------


class EvaluationStatus(Enum):
    """Pure evaluation outcome — no enforcement semantics."""

    ALLOW = "allow"
    VIOLATION = "violation"


@dataclass
class ViolationRecord:
    """Normalized violation metadata produced by an engine."""

    rule_id: str
    rule_name: str
    reason: str
    severity: str = "error"
    scope: str = "turn"
    engine: str = ""
    evidence: list[str] | None = None
    confidence: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Unified evaluation result returned by policy engines.

    Engines are pure evaluators: they return ALLOW or VIOLATION with
    normalized violation metadata.  The interceptor is the sole
    enforcement gateway that maps violations to fail_action behavior.
    """

    status: EvaluationStatus
    violations: list[ViolationRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)



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
    async def initialize(self, config: dict[str, Any]) -> None:
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
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate an incoming request against policies.

        Called BEFORE the LLM call. Returns a pure evaluation outcome
        (ALLOW or VIOLATION). The interceptor decides enforcement.

        Args:
            session_id: Unique session identifier
            request_data: The LLM request data (messages, model, etc.)
            context: Additional context for evaluation

        Returns:
            EvaluationResult with status and optional violations
        """
        ...

    @abstractmethod
    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate an LLM response against policies.

        Called AFTER the LLM call. Returns a pure evaluation outcome.
        The interceptor decides enforcement on the next call.

        Args:
            session_id: Unique session identifier
            response_data: The LLM response
            request_data: The original request data
            context: Additional context for evaluation

        Returns:
            EvaluationResult with status and optional violations
        """
        ...

    @abstractmethod
    async def get_session_state(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
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

    def get_compiler(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> "PolicyCompiler" | None:
        """Get the compiler for this engine type.

        Override in subclasses that have a dedicated compiler.
        Returns None by default (engine has no compiler).
        """
        return None

def require_initialized(method: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to ensure engine is initialized before method call.
    Raises RuntimeError if self._initialized is False.
    """
    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not getattr(self, "_initialized", False):
            raise RuntimeError(f"{type(self).__name__} not initialized. Call initialize() first.")

        return await method(self, *args, **kwargs)
    return wrapper
