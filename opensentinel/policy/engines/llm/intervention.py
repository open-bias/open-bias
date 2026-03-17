"""
Intervention decision engine for the LLM Policy Engine.

Maps violations and drift scores to intervention configurations with
cooldown management and self-correction detection. Actual application
of interventions is handled by the interceptor layer.
"""

import logging
from typing import Optional, Dict, Any, List

from opensentinel.policy.engines.llm.models import (
    ConstraintEvaluation,
    DriftScores,
    DriftLevel,
    SessionContext,
)
from opensentinel.policy.engines.llm.templates import DEFAULT_TEMPLATES, format_template
from opensentinel.core.intervention.strategies import (
    StrategyType,
    InterventionConfig,
)
from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

logger = logging.getLogger(__name__)


class InterventionHandler:
    """Decides when and how to intervene based on violations and drift.

    Maps constraint violations and drift levels to intervention configurations,
    with cooldown to prevent intervention spam and self-correction detection.
    Actual application of interventions is handled by the interceptor.

    Example:
        engine = InterventionHandler(workflow, cooldown_turns=2)
        config = engine.decide(session, violations, drift)
        # config is returned to the interceptor for application
    """

    # Severity to strategy mapping
    SEVERITY_STRATEGY_MAP = {
        "low": StrategyType.SYSTEM_PROMPT_APPEND,
        "warning": StrategyType.SYSTEM_PROMPT_APPEND,
        "medium": StrategyType.USER_MESSAGE_INJECT,
        "error": StrategyType.USER_MESSAGE_INJECT,
        "high": StrategyType.USER_MESSAGE_INJECT,
        "critical": StrategyType.USER_MESSAGE_INJECT,
    }

    # Drift level to strategy mapping
    DRIFT_STRATEGY_MAP = {
        DriftLevel.NOMINAL: None,
        DriftLevel.WARNING: StrategyType.SYSTEM_PROMPT_APPEND,
        DriftLevel.INTERVENTION: StrategyType.USER_MESSAGE_INJECT,
        DriftLevel.CRITICAL: StrategyType.USER_MESSAGE_INJECT,
    }

    def __init__(
        self,
        workflow: WorkflowDefinition,
        cooldown_turns: int = 2,
        self_correction_margin: float = 0.1,
        max_intervention_attempts: int = 3,
    ):
        self.workflow = workflow
        self.cooldown_turns = cooldown_turns
        self.self_correction_margin = self_correction_margin
        self.max_intervention_attempts = max_intervention_attempts
        self._intervention_counts: Dict[str, int] = {}

    def decide(
        self,
        session: SessionContext,
        violations: List[ConstraintEvaluation],
        drift: DriftScores,
    ) -> Optional[InterventionConfig]:
        """Decide if intervention is needed and which strategy to use.
        
        Args:
            session: Current session context
            violations: Constraint violations detected
            drift: Drift scores computed
            
        Returns:
            InterventionConfig if intervention needed, None otherwise
        """
        # Check max intervention attempts
        session_count = self._intervention_counts.get(session.session_id, 0)
        if session_count >= self.max_intervention_attempts:
            logger.warning(
                f"Session {session.session_id} exceeded max intervention attempts "
                f"({session_count}/{self.max_intervention_attempts})"
            )
            return None

        # Check for critical violations that bypass cooldown
        has_critical = any(
            v.severity == "critical" and v.violated
            for v in violations
        )

        # Cooldown check (skip if critical)
        if not has_critical:
            turns_since_intervention = session.turn_count - session.last_intervention_turn
            if turns_since_intervention < self.cooldown_turns:
                logger.debug(
                    f"Cooldown active: {turns_since_intervention}/{self.cooldown_turns} turns"
                )
                return None
        
        # Self-correction check: if drift is decreasing, skip intervention
        if (
            session.last_intervention_turn >= 0
            and drift.composite < session.drift_score - self.self_correction_margin
        ):
            logger.info(
                f"Self-correction detected: drift {session.drift_score:.3f} → "
                f"{drift.composite:.3f}"
            )
            return None

        # Get violation-based strategy
        violation_strategy = None
        violation_severity = None
        first_violation = None
        
        for v in violations:
            if v.violated:
                first_violation = v
                strategy = self.SEVERITY_STRATEGY_MAP.get(
                    v.severity, StrategyType.SYSTEM_PROMPT_APPEND
                )
                # Take the most severe
                if violation_strategy is None or self._compare_strategies(
                    strategy, violation_strategy
                ) > 0:
                    violation_strategy = strategy
                    violation_severity = v.severity
        
        # Get drift-based strategy
        drift_strategy = self.DRIFT_STRATEGY_MAP.get(drift.level)
        
        # No intervention needed
        if not violation_strategy and not drift_strategy:
            return None
        
        # Take most severe strategy
        final_strategy = None
        if violation_strategy and drift_strategy:
            if self._compare_strategies(violation_strategy, drift_strategy) >= 0:
                final_strategy = violation_strategy
            else:
                final_strategy = drift_strategy
        else:
            final_strategy = violation_strategy or drift_strategy
        
        if final_strategy is None:
            return None
        
        # Select template
        template = self._select_template(
            first_violation,
            drift,
            session,
        )
        
        # Determine priority based on strategy severity
        priority = self._get_strategy_priority(final_strategy)
        
        return InterventionConfig(
            strategy_type=final_strategy,
            message_template=template,
            priority=priority,
        )

    def get_config(self, intervention_name: str) -> Optional[InterventionConfig]:
        """Get intervention config by name from workflow interventions."""
        template = self.workflow.interventions.get(intervention_name)
        if template is None:
            return None
        # Parse strategy prefix (same logic as FSM InterventionHandler)
        strategy = StrategyType.SYSTEM_PROMPT_APPEND
        message = template
        if template.startswith("block:"):
            strategy = StrategyType.SYSTEM_PROMPT_APPEND
            message = template[6:].strip()
        elif template.startswith("inject:"):
            strategy = StrategyType.USER_MESSAGE_INJECT
            message = template[7:].strip()
        elif template.startswith("remind:"):
            strategy = StrategyType.SYSTEM_PROMPT_APPEND
            message = template[7:].strip()
        return InterventionConfig(
            strategy_type=strategy,
            message_template=message,
            max_applications=self.max_intervention_attempts,
        )

    def list_interventions(self) -> List[str]:
        """List all available intervention names from workflow."""
        return list(self.workflow.interventions.keys())

    def should_escalate(self, drift: DriftScores) -> bool:
        """Check if situation requires escalation (human review).
        
        Args:
            drift: Current drift scores
            
        Returns:
            True if escalation is warranted
        """
        return drift.level == DriftLevel.CRITICAL

    def _compare_strategies(
        self,
        a: StrategyType,
        b: StrategyType,
    ) -> int:
        """Compare strategy severity. Returns positive if a > b."""
        order = [
            StrategyType.SYSTEM_PROMPT_APPEND,
            StrategyType.USER_MESSAGE_INJECT,
        ]
        return order.index(a) - order.index(b)

    def _get_strategy_priority(self, strategy: StrategyType) -> int:
        """Get priority value for a strategy type."""
        priority_map = {
            StrategyType.SYSTEM_PROMPT_APPEND: 1,
            StrategyType.USER_MESSAGE_INJECT: 2,
        }
        return priority_map.get(strategy, 0)

    def _select_template(
        self,
        violation: Optional[ConstraintEvaluation],
        drift: DriftScores,
        session: SessionContext,
    ) -> str:
        """Select appropriate template for intervention."""
        # Check workflow-defined interventions first
        if violation and violation.constraint_id:
            # Look up constraint to get intervention name
            for c in self.workflow.constraints:
                if c.name == violation.constraint_id and c.intervention:
                    if c.intervention in self.workflow.interventions:
                        return self.workflow.interventions[c.intervention]
        
        # Fall back to default templates
        if violation and violation.violated:
            return DEFAULT_TEMPLATES.get(
                "constraint_violation",
                DEFAULT_TEMPLATES["policy_violation"]
            )
        
        if drift.level == DriftLevel.CRITICAL:
            return DEFAULT_TEMPLATES["drift_critical"]
        elif drift.level == DriftLevel.INTERVENTION:
            return DEFAULT_TEMPLATES["drift_intervention"]
        elif drift.level == DriftLevel.WARNING:
            return DEFAULT_TEMPLATES["drift_warning"]
        
        # Structural drift (multiple uncertain classifications)
        if session.is_structurally_drifting():
            return DEFAULT_TEMPLATES["structural_drift"]
        
        return DEFAULT_TEMPLATES["policy_violation"]

