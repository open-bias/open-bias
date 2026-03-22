"""
Finite State Machine policy engine implementation.

Wraps the existing workflow/state machine implementation as a PolicyEngine,
enabling it to be used alongside other policy mechanisms.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opensentinel.policy.compiler.protocol import PolicyCompiler

from opensentinel.policy.engines.fsm.classifier import StateClassifier
from opensentinel.policy.engines.fsm.workflow.constraints import ConstraintEvaluator
from opensentinel.policy.engines.fsm.workflow.parser import WorkflowParser
from opensentinel.policy.engines.fsm.workflow.schema import ConstraintType, WorkflowDefinition
from opensentinel.policy.engines.fsm.workflow.state_machine import (
    TransitionResult,
    WorkflowStateMachine,
)
from opensentinel.policy.engines.stateful import (
    StateClassificationResult,
    StatefulPolicyEngine,
)
from opensentinel.policy.protocols import (
    Decision,
    EngineResult,
    require_initialized,
)
from opensentinel.policy.registry import register_engine

logger = logging.getLogger(__name__)

@register_engine("fsm")
class FSMPolicyEngine(StatefulPolicyEngine):
    """
    Finite State Machine based policy engine.

    Uses workflow definitions with states, transitions, and LTL-lite
    constraints to enforce policies. This wraps the existing Open Sentinel
    workflow implementation as a PolicyEngine.

    Configuration:
        - workflow_path: str - Path to workflow YAML/JSON file
        OR
        - workflow: dict - Workflow definition as dictionary

    Example:
        ```python
        engine = FSMPolicyEngine()
        await engine.initialize({
            "workflow_path": "./examples/customer_support.yaml"
        })

        result = await engine.evaluate_response(
            session_id="session-123",
            response_data=llm_response,
            request_data=original_request,
        )
        ```
    """

    def __init__(self) -> None:
        self._workflow: WorkflowDefinition | None = None
        self._state_machine: WorkflowStateMachine | None = None
        self._classifier: StateClassifier | None = None
        self._constraint_evaluator: ConstraintEvaluator | None = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Unique name of this policy engine instance."""
        if self._workflow:
            return f"fsm:{self._workflow.name}"
        return "fsm:uninitialized"

    @property
    def engine_type(self) -> str:
        """Type identifier for this engine."""
        return "fsm"

    async def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize with workflow configuration.

        Detects config format automatically:
        - If config has ``steps`` key → simple format, auto-compiled
        - If config has ``states`` key → internal format, loaded directly
        - ``config_path`` / ``workflow`` → format detected from contents

        Args:
            config: Configuration dict with either:
                - config_path: str - Path to workflow YAML/JSON
                - workflow: dict - Workflow definition as dict

        Raises:
            ValueError: If neither config_path nor workflow provided
        """
        if "config_path" in config:
            self._workflow = WorkflowParser.parse_file(config["config_path"])
        elif "workflow" in config:
            workflow_data = config["workflow"]
            if isinstance(workflow_data, dict):
                self._workflow = WorkflowParser.parse_dict(workflow_data)
            else:
                self._workflow = workflow_data
        else:
            raise ValueError(
                "FSM engine requires 'config_path' or 'workflow' in config"
            )

        sm_kwargs: dict[str, Any] = {}
        if "session_ttl" in config:
            sm_kwargs["session_ttl"] = config["session_ttl"]
        if "max_sessions" in config:
            sm_kwargs["max_sessions"] = config["max_sessions"]
        self._state_machine = WorkflowStateMachine(self._workflow, **sm_kwargs)

        # Wire classifier config from engine config
        from opensentinel.config.settings import ClassifierConfig
        classifier_cfg = config.get("classifier", {})
        classifier_config = ClassifierConfig(**classifier_cfg) if classifier_cfg else None
        self._classifier = StateClassifier(self._workflow.states, config=classifier_config)

        self._constraint_evaluator = ConstraintEvaluator(self._workflow.constraints)

        self._initialized = True

        logger.info(
            f"FSMPolicyEngine initialized with workflow '{self._workflow.name}' "
            f"({len(self._workflow.states)} states, "
            f"{len(self._workflow.constraints)} constraints)"
        )

    @require_initialized
    async def evaluate_request(
        self,
        session_id: str,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        """Evaluate request — FSM evaluation happens post-call."""
        session = await self._state_machine.get_or_create_session(session_id)

        return EngineResult(
            decision=Decision.ALLOW,
            metadata={
                "current_state": session.current_state,
                "workflow": self._workflow.name,
            },
        )

    @require_initialized
    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        """Evaluate response — classify state, check constraints, transition."""
        try:
            return await self._evaluate_response_inner(
                session_id, response_data, request_data, context
            )
        except Exception as e:
            logger.exception(
                f"FSM engine error for session {session_id}: {e}"
            )
            return EngineResult(
                decision=Decision.ALLOW,
                metadata={
                    "error": str(e),
                    "engine": self.name,
                    "session_id": session_id,
                },
            )

    async def _evaluate_response_inner(
        self,
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        """Inner evaluate_response logic, separated for fail-open wrapping."""
        session = await self._state_machine.get_or_create_session(session_id)
        previous_state = session.current_state

        # Classify response to determine workflow state
        classification = self._classifier.classify(response_data, previous_state)

        logger.debug(
            f"Session {session_id}: Classified response as '{classification.state_name}' "
            f"(confidence={classification.confidence:.2f}, method={classification.method})"
        )

        # Evaluate constraints BEFORE the transition
        constraint_violations = self._constraint_evaluator.evaluate_all(
            session,
            proposed_state=classification.state_name,
        )

        # If any NEVER constraint fires, skip the transition entirely
        has_never_violation = any(
            cv.constraint_type == ConstraintType.NEVER for cv in constraint_violations
        )

        transition_result: TransitionResult
        error: str | None

        if has_never_violation:
            transition_result = TransitionResult.CONSTRAINT_VIOLATED
            error = None
        else:
            # Attempt state transition with optimistic concurrency guard
            transition_result, error = await self._state_machine.transition(
                session_id,
                classification.state_name,
                confidence=classification.confidence,
                method=classification.method,
                expected_from_state=previous_state,
            )

        decision = Decision.ALLOW
        message: str | None = None

        if transition_result == TransitionResult.INVALID_TRANSITION:
            decision = Decision.INTERVENE
            message = error or f"Invalid transition to '{classification.state_name}'"
        elif constraint_violations:
            decision = Decision.INTERVENE
            message = "\n".join(cv.message for cv in constraint_violations)

        # Session-boundary evaluation: check EVENTUALLY/RESPONSE at terminal state
        if transition_result == TransitionResult.SUCCESS:
            if await self._state_machine.is_in_terminal_state(session_id):
                boundary_violations = self._constraint_evaluator.evaluate_session_boundary(
                    session,
                )
                if boundary_violations:
                    constraint_violations.extend(boundary_violations)
                    decision = Decision.INTERVENE
                    message = "\n".join(cv.message for cv in constraint_violations)

        return EngineResult(
            decision=decision,
            message=message,
            metadata={
                "previous_state": previous_state,
                "current_state": session.current_state,
                "classification_confidence": classification.confidence,
                "classification_method": classification.method,
                "transition_result": transition_result.value,
                "transition_error": error,
                "workflow": self._workflow.name,
                "violations": [
                    {
                        "name": cv.constraint_name,
                        "message": cv.message,
                        "constraint_type": cv.constraint_type.value,
                        **cv.details,
                    }
                    for cv in constraint_violations
                ],
            },
        )

    @require_initialized
    async def classify_response(
        self,
        session_id: str,
        response_data: Any,
        current_state: str | None = None,
    ) -> StateClassificationResult:
        """
        Classify response to workflow state.

        Args:
            session_id: Unique session identifier
            response_data: The LLM response to classify
            current_state: Current state (looked up if not provided)

        Returns:
            StateClassificationResult with detected state
        """

        if current_state is None:
            session = await self._state_machine.get_or_create_session(session_id)
            current_state = session.current_state

        return self._classifier.classify(response_data, current_state)

    @require_initialized
    async def get_current_state(self, session_id: str) -> str:
        """Get current state name for session."""

        session = await self._state_machine.get_or_create_session(session_id)
        return session.current_state

    @require_initialized
    async def get_state_history(self, session_id: str) -> list[str]:
        """Get state transition history."""

        return await self._state_machine.get_state_history(session_id)

    @require_initialized
    async def get_valid_next_states(self, session_id: str) -> list[str]:
        """Get valid next states from current state."""

        valid = await self._state_machine.get_valid_transitions(session_id)
        return list(valid)

    async def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        """Get current session state for debugging/tracing."""
        if not self._initialized:
            return None

        session = await self._state_machine.get_session(session_id)
        if not session:
            return None

        return {
            "current_state": session.current_state,
            "history": session.get_state_sequence(),
            "constraint_violations": session.constraint_violations,
            "workflow": self._workflow.name,
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
        }

    async def reset_session(self, session_id: str) -> None:
        """Reset session state, evaluating boundary constraints first."""
        if not self._initialized:
            return

        # Evaluate session-boundary constraints before reset
        session = await self._state_machine.get_session(session_id)
        if session:
            boundary_violations = self._constraint_evaluator.evaluate_session_boundary(
                session,
            )
            if boundary_violations:
                logger.info(
                    f"Session {session_id} reset with {len(boundary_violations)} "
                    "pending constraint violations"
                )

        await self._state_machine.reset_session(session_id)
        logger.debug(f"Session {session_id} reset")

    def get_compiler(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> "PolicyCompiler" | None:
        """Return an FSMCompiler instance."""
        from opensentinel.policy.engines.fsm.compiler import FSMCompiler
        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return FSMCompiler(**kwargs)

    async def shutdown(self) -> None:
        """Cleanup resources."""
        # FSM engine doesn't need explicit cleanup
        self._initialized = False
        logger.info("FSMPolicyEngine shutdown")
