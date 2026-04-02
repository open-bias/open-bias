"""
LLM Policy Engine implementation.

The main orchestrator that uses an LLM for state classification,
drift detection, and soft constraint evaluation. Registered via
@register_engine("llm") for use with PolicyEngineRegistry.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openbias.policy.compiler.protocol import PolicyCompiler

from openbias.policy.registry import register_engine
from openbias.policy.protocols import (
    EvaluationResult,
    EvaluationStatus,
    ViolationRecord,
    require_initialized,
)
from openbias.policy.engines.stateful import (
    StatefulPolicyEngine,
    StateClassificationResult,
)
from openbias.core.session import SessionStore
from openbias.policy.engines.llm.models import SessionContext
from openbias.policy.engines.llm.llm_client import LLMClient
from openbias.policy.engines.llm.state_classifier import LLMStateClassifier
from openbias.policy.engines.llm.drift_detector import DriftDetector
from openbias.policy.engines.llm.constraint_evaluator import LLMConstraintEvaluator
from openbias.policy.engines.fsm.workflow.parser import WorkflowParser
from openbias.policy.engines.fsm.workflow.schema import WorkflowDefinition
from openbias.core.utils import extract_response_content, extract_tool_call_names

logger = logging.getLogger(__name__)


@register_engine("llm")
class LLMPolicyEngine(StatefulPolicyEngine):
    """LLM-based policy engine.
    
    Uses a lightweight LLM (e.g. gpt-4o-mini) as a reasoning backbone for:
    - State classification with confidence scoring
    - Drift detection (temporal + semantic)
    - Soft constraint evaluation
    
    Reuses the same WorkflowDefinition schema as FSM engine, so users
    can swap engines without rewriting policies.
    
    Example:
        engine = LLMPolicyEngine()
        await engine.initialize({
            "config_path": "workflow.yaml",
            "llm_model": "gpt-4o-mini",
            "temporal_weight": 0.55,
        })
        
        result = await engine.evaluate_response(
            session_id="abc123",
            response_data=llm_response,
            request_data=request,
        )
    """

    DEFAULT_SESSION_TTL = 3600  # 1 hour
    DEFAULT_MAX_SESSIONS = 10_000

    def __init__(self):
        self._workflow: WorkflowDefinition | None = None
        self._llm_client: LLMClient | None = None
        self._state_classifier: LLMStateClassifier | None = None
        self._drift_detector: DriftDetector | None = None
        self._constraint_evaluator: LLMConstraintEvaluator | None = None
        self._sessions: SessionStore[SessionContext] = SessionStore(
            ttl=self.DEFAULT_SESSION_TTL,
            max_sessions=self.DEFAULT_MAX_SESSIONS,
        )
        self._initialized = False

    @property
    def name(self) -> str:
        """Unique name of this policy engine instance."""
        if self._workflow:
            return f"llm:{self._workflow.name}"
        return "llm:uninitialized"

    @property
    def engine_type(self) -> str:
        """Type identifier for this engine."""
        return "llm"

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the engine with configuration.
        
        Args:
            config: Configuration dict with:
                - config_path: Path to workflow YAML/JSON
                - workflow: Workflow definition as dict (alternative)
                - llm_model: LLM model to use (default: gpt-4o-mini)
                - temperature: LLM temperature (default: 0.0)
                - temporal_weight: Weight for temporal drift (default: 0.55)
                - cooldown_turns: Intervention cooldown (default: 2)
        """
        # Load workflow
        workflow_path = config.get("config_path")
        workflow_dict = config.get("workflow")

        if workflow_path:
            self._workflow = WorkflowParser.parse_file(workflow_path)
        elif workflow_dict:
            self._workflow = WorkflowParser.parse_dict(workflow_dict)
        else:
            raise ValueError("Either config_path or workflow must be provided")
        
        # Create LLM client
        # model comes from config (injected by interceptor in hooks.py)
        model = config.get("llm_model")
        self._llm_client = LLMClient(
            model=model,
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 1024),
            timeout=config.get("timeout", 10.0),
        )
        logger.info(f"LLM engine using model: {self._llm_client.model}")
        
        # Create components
        self._state_classifier = LLMStateClassifier(
            self._llm_client,
            self._workflow,
            confident_threshold=config.get("confident_threshold", 0.8),
            uncertain_threshold=config.get("uncertain_threshold", 0.5),
        )
        
        self._drift_detector = DriftDetector(
            self._workflow,
            temporal_weight=config.get("temporal_weight", 0.55),
        )
        
        self._constraint_evaluator = LLMConstraintEvaluator(
            self._llm_client,
            self._workflow,
            max_constraints_per_batch=config.get("max_constraints_per_batch", 5),
        )
        
        self._initialized = True
        logger.info(f"LLMPolicyEngine initialized: {self.name}")

    @require_initialized
    async def evaluate_request(
        self,
        session_id: str,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate incoming request — pass-through, evaluation happens post-call."""
        return EvaluationResult(status=EvaluationStatus.ALLOW)

    @require_initialized
    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate LLM response — classify, detect drift, check constraints."""
        session = self._get_or_create_session(session_id)

        # Extract content from response
        message = extract_response_content(response_data)
        tool_calls = extract_tool_call_names(response_data)

        violation_records: list[ViolationRecord] = []

        try:
            # 1. Classify state (before add_turn so the current message
            #    isn't duplicated in the classifier's context window)
            classification = await self._state_classifier.classify(
                session, message, tool_calls
            )
            logger.debug(
                "State classification: state=%s, confidence=%.2f, tier=%s",
                classification.best_state,
                classification.best_confidence,
                classification.tier.value,
            )

            # Record turn after classification
            session.add_turn({
                "role": "assistant",
                "message": message,
                "tool_calls": tool_calls,
            })

            # Update confidence buffer
            session.add_confidence(classification.best_confidence)

            # Check for structural drift
            if session.is_structurally_drifting():
                violation_records.append(ViolationRecord(
                    reason="Multiple consecutive uncertain classifications",
                    severity="warning",
                    engine=self.name,
                ))

            # Check for skip violations
            for skipped in classification.skip_violations:
                violation_records.append(ViolationRecord(
                    reason=f"Skipped required state: {skipped}",
                    severity="error",
                    engine=self.name,
                    extra={"skipped_state": skipped},
                ))

            # 2. Compute drift
            expected_tools = self._get_expected_tools(classification.best_state)
            drift = self._drift_detector.compute_drift(
                session, message, tool_calls, expected_tools
            )
            logger.debug(
                "Drift computed: composite=%.3f, level=%s",
                drift.composite, drift.level.value,
            )

            # Add anomaly violations
            if drift.anomaly_flags.get("unexpected_tool_call"):
                violation_records.append(ViolationRecord(
                    reason="Unexpected tool call for current state",
                    severity="warning",
                    engine=self.name,
                ))

            if drift.anomaly_flags.get("missing_expected_tool_call"):
                violation_records.append(ViolationRecord(
                    reason="Expected tool call not made",
                    severity="warning",
                    engine=self.name,
                ))

            # 3. Evaluate constraints
            constraint_evals = await self._constraint_evaluator.evaluate(
                session, message, tool_calls
            )
            if constraint_evals:
                violated = [cv for cv in constraint_evals if cv.violated]
                logger.debug(
                    "Constraint evaluation: %d checked, %d violated",
                    len(constraint_evals), len(violated),
                )

            for cv in constraint_evals:
                if cv.violated:
                    violation_records.append(ViolationRecord(
                        reason=cv.evidence,
                        severity=cv.severity,
                        engine=self.name,
                        confidence=cv.confidence,
                    ))

            # 4. Record transition
            prev_state = session.current_state
            session.record_transition(
                from_state=prev_state,
                to_state=classification.best_state,
                confidence=classification.best_confidence,
                tier=classification.tier,
                drift_score=drift.composite,
                metadata={
                    "method": "llm",
                    "candidates": len(classification.candidates),
                },
            )

            # 5. Build result — pure evaluation, no intervention logic
            severity_order = ["critical", "error", "warning", "info"]
            max_severity = next(
                (s for s in severity_order if any(v.severity == s for v in violation_records)),
                None,
            )

            status = EvaluationStatus.VIOLATION if violation_records else EvaluationStatus.ALLOW

            return EvaluationResult(
                status=status,
                violations=violation_records,
                metadata={
                    "state": classification.best_state,
                    "confidence": classification.best_confidence,
                    "tier": classification.tier.value,
                    "drift": drift.composite,
                    "drift_level": drift.level.value,
                    "transition_legal": classification.transition_legal,
                    "max_severity": max_severity,
                },
            )
        except Exception as e:
            logger.error(f"Response evaluation failed: {e}")
            return EvaluationResult(
                status=EvaluationStatus.ALLOW,
                metadata={"error": str(e)},
            )
    @require_initialized
    async def classify_response(
        self,
        session_id: str,
        response_data: Any,
        current_state: str | None = None,
    ) -> StateClassificationResult:
        """Classify a response to a workflow state."""
        
        session = self._get_or_create_session(session_id)
        
        message = extract_response_content(response_data)
        tool_calls = extract_tool_call_names(response_data)

        result = await self._state_classifier.classify(session, message, tool_calls)
        
        return StateClassificationResult(
            state_name=result.best_state,
            confidence=result.best_confidence,
            method="llm",
            details={
                "tier": result.tier.value,
                "candidates": len(result.candidates),
                "transition_legal": result.transition_legal,
            },
        )

    @require_initialized
    async def get_current_state(self, session_id: str) -> str:
        """Get current state name for session."""
        session = self._sessions.get(session_id)
        if session is not None:
            return session.current_state

        # Return initial state
        if self._workflow:
            initial = self._workflow.get_initial_states()
            if initial:
                return initial[0].name
        return "unknown"

    @require_initialized
    async def get_state_history(self, session_id: str) -> list[str]:
        """Get state transition history."""
        session = self._sessions.get(session_id)
        if session is not None:
            return session.get_state_sequence()
        return []

    @require_initialized
    async def get_valid_next_states(self, session_id: str) -> list[str]:
        """Get valid next states from current state."""
        current = await self.get_current_state(session_id)
        if self._workflow:
            transitions = self._workflow.get_transitions_from(current)
            return [t.to_state for t in transitions]
        return []

    async def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        """Get current session state for debugging/tracing."""
        session = self._sessions.get(session_id)
        if session is not None:
            return session.to_dict()
        return None

    async def reset_session(self, session_id: str) -> None:
        """Reset session state."""
        if self._sessions.remove(session_id) is not None:
            logger.debug(f"Reset session: {session_id}")

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._sessions.clear()
        self._initialized = False
        logger.info("LLMPolicyEngine shutdown complete")

    def _get_or_create_session(self, session_id: str) -> SessionContext:
        """Get existing session or create new one."""
        session = self._sessions.get(session_id)
        if session is not None:
            self._sessions.touch(session_id)
            return session

        # Get initial state
        initial_state = "unknown"
        if self._workflow:
            initial = self._workflow.get_initial_states()
            if initial:
                initial_state = initial[0].name

        session = SessionContext(
            session_id=session_id,
            workflow_name=self._workflow.name if self._workflow else "unknown",
            current_state=initial_state,
        )
        self._sessions.put(session_id, session)
        logger.debug(f"Created session: {session_id}")
        return session

    def get_compiler(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> "PolicyCompiler | None":
        """Return an LLMCompiler instance for rules-first compilation."""
        from openbias.policy.engines.llm.compiler import LLMCompiler

        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return LLMCompiler(**kwargs)

    def _get_expected_tools(self, state_name: str) -> list[str]:
        """Get expected tool calls for a state."""
        if self._workflow:
            state = self._workflow.get_state(state_name)
            if state and state.classification.tool_calls:
                return state.classification.tool_calls
        return []

