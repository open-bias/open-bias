"""
Intervention decision engine for the LLM Policy Engine.

Decides whether to intervene based on violations and drift, with
cooldown management and self-correction detection. Returns a message
template string; strategy selection is handled by the interceptor layer.
"""

import logging
from typing import Optional, Dict, List

from opensentinel.policy.engines.llm.models import (
    ConstraintEvaluation,
    DriftScores,
    DriftLevel,
    SessionContext,
)
from opensentinel.policy.engines.llm.templates import DEFAULT_TEMPLATES
from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

logger = logging.getLogger(__name__)


class InterventionHandler:
    """Decides whether to intervene based on violations and drift.

    Handles cooldown to prevent intervention spam, self-correction detection,
    and max attempt caps. Returns a message template string for the interceptor
    to apply using its configured strategy.

    Example:
        handler = InterventionHandler(workflow, cooldown_turns=2)
        message = handler.decide(session, violations, drift)
        # message is returned via EngineResult for the interceptor to apply
    """

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
    ) -> Optional[str]:
        """Decide if intervention is needed and select a message template.

        Args:
            session: Current session context
            violations: Constraint violations detected
            drift: Drift scores computed

        Returns:
            Message template string if intervention needed, None otherwise
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

        # Check whether any violation or drift warrants intervention
        has_violation = any(v.violated for v in violations)
        has_drift = drift.level not in (DriftLevel.NOMINAL, None)

        if not has_violation and not has_drift:
            return None

        first_violation = next((v for v in violations if v.violated), None)

        return self._select_template(first_violation, drift, session)

    def get_template(self, constraint_name: str) -> Optional[str]:
        """Get intervention message by constraint name from workflow constraints."""
        for c in self.workflow.constraints:
            if c.name == constraint_name:
                return c.message if c.message else None
        return None

    def list_interventions(self) -> List[str]:
        """List all constraint names from workflow."""
        return [c.name for c in self.workflow.constraints]

    def should_escalate(self, drift: DriftScores) -> bool:
        """Check if situation requires escalation (human review).

        Args:
            drift: Current drift scores

        Returns:
            True if escalation is warranted
        """
        return drift.level == DriftLevel.CRITICAL

    def _select_template(
        self,
        violation: Optional[ConstraintEvaluation],
        drift: DriftScores,
        session: SessionContext,
    ) -> str:
        """Select appropriate template for intervention."""
        # Check workflow-defined constraint messages first
        if violation and violation.constraint_id:
            # Look up constraint to get its message
            for c in self.workflow.constraints:
                if c.name == violation.constraint_id and c.message:
                    return c.message

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

