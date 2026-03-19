"""
Finite State Machine policy engine implementation.

Wraps the existing workflow/state machine implementation as a PolicyEngine,
enabling it to be used alongside other policy mechanisms.
"""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from opensentinel.policy.compiler.protocol import PolicyCompiler

from opensentinel.policy.protocols import (
    Decision,
    EngineResult,
    require_initialized,
)
from opensentinel.policy.engines.stateful import (
    StatefulPolicyEngine,
    StateClassificationResult,
)
from opensentinel.policy.registry import register_engine
from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition
from opensentinel.policy.engines.fsm.workflow.parser import WorkflowParser
from opensentinel.policy.engines.fsm.workflow.state_machine import WorkflowStateMachine, TransitionResult
from opensentinel.policy.engines.fsm.workflow.constraints import ConstraintEvaluator
from opensentinel.policy.engines.fsm.classifier import StateClassifier
from opensentinel.policy.engines.fsm.intervention import InterventionHandler

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

    def __init__(self):
        self._workflow: Optional[WorkflowDefinition] = None
        self._state_machine: Optional[WorkflowStateMachine] = None
        self._classifier: Optional[StateClassifier] = None
        self._constraint_evaluator: Optional[ConstraintEvaluator] = None
        self._intervention_handler: Optional[InterventionHandler] = None
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

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize with workflow configuration.

        Args:
            config: Configuration dict with either:
                - workflow_path: str - Path to workflow YAML/JSON
                - workflow: dict - Workflow definition as dict

        Raises:
            ValueError: If neither workflow_path nor workflow provided
        """
        if "config_path" in config:
            # Unified config path
            self._workflow = WorkflowParser.parse_file(config["config_path"])
        elif "workflow" in config:
            workflow_data = config["workflow"]
            if isinstance(workflow_data, dict):
                self._workflow = WorkflowParser().parse_dict(workflow_data)
            else:
                self._workflow = workflow_data
        else:
            raise ValueError(
                "FSM engine requires 'config_path' or 'workflow' in config"
            )

        self._state_machine = WorkflowStateMachine(self._workflow)

        # Wire classifier config from engine config
        from opensentinel.config.settings import ClassifierConfig
        classifier_cfg = config.get("classifier", {})
        classifier_config = ClassifierConfig(**classifier_cfg) if classifier_cfg else None
        self._classifier = StateClassifier(self._workflow.states, config=classifier_config)

        self._constraint_evaluator = ConstraintEvaluator(self._workflow.constraints)

        # Wire intervention config from engine config
        from opensentinel.core.intervention.strategies import StrategyType
        intervention_cfg = config.get("intervention", {})
        default_strategy_str = intervention_cfg.get("default_strategy", "system_prompt_append")
        default_strategy = StrategyType(default_strategy_str)
        max_intervention_attempts = intervention_cfg.get("max_intervention_attempts", 3)
        self._intervention_handler = InterventionHandler(
            self._workflow,
            default_strategy=default_strategy,
            max_intervention_attempts=max_intervention_attempts,
        )

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
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
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
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EngineResult:
        """Evaluate response — classify state, check constraints, transition."""
        session = await self._state_machine.get_or_create_session(session_id)
        previous_state = session.current_state

        # Classify response to determine workflow state
        classification = self._classifier.classify(response_data, previous_state)

        logger.debug(
            f"Session {session_id}: Classified response as '{classification.state_name}' "
            f"(confidence={classification.confidence:.2f}, method={classification.method})"
        )

        # Check constraints BEFORE transition
        constraint_violations = self._constraint_evaluator.evaluate_all(
            session,
            proposed_state=classification.state_name,
        )

        # Attempt state transition
        transition_result, error = await self._state_machine.transition(
            session_id,
            classification.state_name,
            context,
            confidence=classification.confidence,
            method=classification.method,
        )

        # Map constraint violations to Decision
        decision = Decision.ALLOW
        message: Optional[str] = None

        if constraint_violations:
            has_critical = any(cv.severity == "critical" for cv in constraint_violations)

            if has_critical:
                decision = Decision.BLOCK
                critical = next(cv for cv in constraint_violations if cv.severity == "critical")
                message = critical.message
            else:
                decision = Decision.INTERVENE
                # Build message from intervention handler template if available
                for cv in constraint_violations:
                    if cv.intervention and self._intervention_handler:
                        config = self._intervention_handler.get_config(cv.intervention)
                        if config:
                            from opensentinel.core.intervention.strategies import (
                                InterventionStrategy,
                            )
                            template_context = {
                                "constraint_type": cv.constraint_type.value,
                                **cv.details,
                            }
                            message = InterventionStrategy.format_message(
                                config.message_template, template_context
                            )
                            break
                if message is None:
                    message = constraint_violations[0].message

        return EngineResult(
            decision=decision,
            message=message,
            metadata={
                "previous_state": previous_state,
                "current_state": classification.state_name,
                "classification_confidence": classification.confidence,
                "classification_method": classification.method,
                "transition_result": transition_result.value,
                "transition_error": error,
                "workflow": self._workflow.name,
                "violations": [
                    {
                        "name": cv.constraint_name,
                        "severity": cv.severity,
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
        current_state: Optional[str] = None,
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
    async def get_state_history(self, session_id: str) -> List[str]:
        """Get state transition history."""

        return await self._state_machine.get_state_history(session_id)

    @require_initialized
    async def get_valid_next_states(self, session_id: str) -> List[str]:
        """Get valid next states from current state."""

        valid = await self._state_machine.get_valid_transitions(session_id)
        return list(valid)

    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
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
        """Reset session state."""
        if not self._initialized:
            return

        await self._state_machine.reset_session(session_id)
        logger.debug(f"Session {session_id} reset")

    def get_compiler(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Optional["PolicyCompiler"]:
        """Return an FSMCompiler instance."""
        from opensentinel.policy.engines.fsm.compiler import FSMCompiler
        kwargs: Dict[str, Any] = {}
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
