"""
LLM Policy Engine implementation.

The main orchestrator that uses an LLM for state classification,
drift detection, and soft constraint evaluation. Registered via
@register_engine("llm") for use with PolicyEngineRegistry.
"""

import json
import logging
from typing import Any

from opensentinel.policy.registry import register_engine
from opensentinel.policy.protocols import (
    Decision,
    EngineResult,
    require_initialized,
)
from opensentinel.policy.engines.stateful import (
    StatefulPolicyEngine,
    StateClassificationResult,
)
from opensentinel.core.session import SessionStore
from opensentinel.policy.engines.llm.models import (
    SessionContext,
    ConfidenceTier,
    DriftLevel,
)
from opensentinel.policy.engines.llm.llm_client import LLMClient
from opensentinel.policy.engines.llm.state_classifier import LLMStateClassifier
from opensentinel.policy.engines.llm.drift_detector import DriftDetector
from opensentinel.policy.engines.llm.constraint_evaluator import LLMConstraintEvaluator
from opensentinel.policy.engines.llm.intervention import InterventionHandler
from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition
from opensentinel.core.utils import extract_response_content, extract_tool_call_names

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
        self._intervention_engine: InterventionHandler | None = None
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
        import yaml
        from pathlib import Path
        
        # Load workflow
        workflow_path = config.get("config_path")
        workflow_dict = config.get("workflow")
        
        if workflow_path:
            path = Path(workflow_path)
            if not path.exists():
                raise ValueError(f"Workflow file not found: {workflow_path}")
            
            with open(path) as f:
                if path.suffix in (".yaml", ".yml"):
                    workflow_dict = yaml.safe_load(f)
                else:
                    workflow_dict = json.load(f)
        
        if not workflow_dict:
            raise ValueError("Either config_path or workflow must be provided")
        
        self._workflow = WorkflowDefinition(**workflow_dict)
        
        # Create LLM client
        # model comes from config (injected by SentinelSettings.get_policy_config)
        model = config.get("llm_model") or config.get("default_model") or None
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
        
        intervention_cfg = config.get("intervention", {})
        self._intervention_engine = InterventionHandler(
            self._workflow,
            cooldown_turns=config.get("cooldown_turns", intervention_cfg.get("cooldown_turns", 2)),
            max_intervention_attempts=intervention_cfg.get("max_intervention_attempts", 3),
        )
        
        self._initialized = True
        logger.info(f"LLMPolicyEngine initialized: {self.name}")

    @require_initialized
    async def evaluate_request(
        self,
        session_id: str,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        """Evaluate incoming request — pass-through, evaluation happens post-call."""
        return EngineResult(decision=Decision.ALLOW)

    @require_initialized
    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        """Evaluate LLM response — classify, detect drift, check constraints."""
        session = self._get_or_create_session(session_id)

        # Extract content from response
        message = extract_response_content(response_data)
        tool_calls = extract_tool_call_names(response_data)

        # Add turn to session
        session.add_turn({
            "role": "assistant",
            "message": message,
            "tool_calls": tool_calls,
        })

        violations: list[dict[str, Any]] = []

        try:
            # 1. Classify state
            classification = await self._state_classifier.classify(
                session, message, tool_calls
            )

            # Update confidence buffer
            session.add_confidence(classification.best_confidence)

            # Check for structural drift
            if session.is_structurally_drifting():
                violations.append({
                    "name": "structural_drift",
                    "severity": "warning",
                    "message": "Multiple consecutive uncertain classifications",
                })

            # Check for skip violations
            for skipped in classification.skip_violations:
                violations.append({
                    "name": "skip_violation",
                    "severity": "error",
                    "message": f"Skipped required state: {skipped}",
                    "skipped_state": skipped,
                })

            # 2. Compute drift
            expected_tools = self._get_expected_tools(classification.best_state)
            drift = self._drift_detector.compute_drift(
                session, message, tool_calls, expected_tools
            )

            # Add anomaly violations
            if drift.anomaly_flags.get("unexpected_tool_call"):
                violations.append({
                    "name": "unexpected_tool_call",
                    "severity": "warning",
                    "message": "Unexpected tool call for current state",
                })

            if drift.anomaly_flags.get("missing_expected_tool_call"):
                violations.append({
                    "name": "missing_expected_tool_call",
                    "severity": "warning",
                    "message": "Expected tool call not made",
                })

            # 3. Evaluate constraints
            constraint_evals = await self._constraint_evaluator.evaluate(
                session, message, tool_calls
            )

            for cv in constraint_evals:
                if cv.violated:
                    violations.append({
                        "name": cv.constraint_id,
                        "severity": cv.severity,
                        "message": cv.evidence,
                        "confidence": cv.confidence,
                    })

            # 4. Decide intervention
            intervention_message = None
            if self._intervention_engine:
                intervention_message = self._intervention_engine.decide(
                    session, constraint_evals, drift
                )

            # 5. Record transition
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

            # 6. Determine decision and message
            decision = Decision.ALLOW
            result_message: str | None = None

            if intervention_message:
                from opensentinel.core.intervention.strategies import (
                    InterventionStrategy,
                )
                decision = Decision.INTERVENE
                template_context = {
                    "state": classification.best_state,
                    "drift": drift.composite,
                    "drift_level": drift.level.value,
                }
                result_message = InterventionStrategy.format_message(
                    intervention_message, template_context
                )

            # Critical violations override to BLOCK
            if any(v["severity"] == "critical" for v in violations):
                decision = Decision.BLOCK
                critical = next(v for v in violations if v["severity"] == "critical")
                result_message = critical["message"]

            return EngineResult(
                decision=decision,
                message=result_message,
                metadata={
                    "state": classification.best_state,
                    "confidence": classification.best_confidence,
                    "tier": classification.tier.value,
                    "drift": drift.composite,
                    "drift_level": drift.level.value,
                    "transition_legal": classification.transition_legal,
                    "violations": violations,
                },
            )

        except Exception as e:
            logger.error(f"Response evaluation failed: {e}")
            return EngineResult(
                decision=Decision.ALLOW,
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

    async def get_state_history(self, session_id: str) -> list[str]:
        """Get state transition history."""
        session = self._sessions.get(session_id)
        if session is not None:
            return session.get_state_sequence()
        return []

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

    def _get_expected_tools(self, state_name: str) -> list[str]:
        """Get expected tool calls for a state."""
        if self._workflow:
            state = self._workflow.get_state(state_name)
            if state and state.classification.tool_calls:
                return state.classification.tool_calls
        return []

