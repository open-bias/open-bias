"""
Constraint evaluator for workflow enforcement.

Implements runtime verification for 4 constraint types:
- PRECEDENCE: Target must occur before trigger
- NEVER: Target state must never occur
- EVENTUALLY: Must eventually reach target state
- RESPONSE: If trigger occurs, target must eventually follow
"""

import logging
from dataclasses import dataclass
from enum import Enum

from opensentinel.policy.engines.fsm.workflow.schema import Constraint, ConstraintType
from opensentinel.policy.engines.fsm.workflow.state_machine import SessionState

logger = logging.getLogger(__name__)


class EvaluationResult(Enum):
    """Constraint evaluation outcome."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    PENDING = "pending"


@dataclass
class ConstraintViolation:
    """Details of a constraint violation."""

    constraint_name: str
    constraint_type: ConstraintType
    message: str
    details: dict


class ConstraintEvaluator:
    """
    Evaluates constraints against workflow execution.

    Example:
        ```python
        evaluator = ConstraintEvaluator(workflow.constraints)
        violations = evaluator.evaluate_all(session_state)
        violations = evaluator.evaluate_all(session_state, proposed_state="resolution")
        ```
    """

    def __init__(self, constraints: list[Constraint]):
        self.constraints = constraints
        logger.debug(
            f"ConstraintEvaluator initialized with {len(constraints)} constraints"
        )

    def evaluate_all(
        self,
        session: SessionState,
        proposed_state: str | None = None,
    ) -> list[ConstraintViolation]:
        """
        Evaluate all constraints for a session.

        Args:
            session: Current session state
            proposed_state: If given, evaluates as if that transition happened

        Returns:
            List of constraint violations (empty if all satisfied)
        """
        violations = []
        history = session.get_state_sequence()

        if proposed_state:
            history = history + [proposed_state]

        for constraint in self.constraints:
            result = self._evaluate_constraint(constraint, history, proposed_state)

            if result == EvaluationResult.VIOLATED:
                violation = ConstraintViolation(
                    constraint_name=constraint.name,
                    constraint_type=constraint.type,
                    message=constraint.message
                    or self._format_violation_message(constraint, history),
                    details={
                        "history": history[-5:],
                        "current_state": session.current_state,
                        "proposed_state": proposed_state,
                    },
                )
                violations.append(violation)
                logger.info(
                    f"Constraint violated: {constraint.name} ({constraint.type.value})"
                )

        return violations

    def evaluate_session_boundary(
        self,
        session: SessionState,
    ) -> list[ConstraintViolation]:
        """Evaluate EVENTUALLY and RESPONSE constraints at session boundary.

        Called when a terminal state is reached or session is reset.
        PENDING EVENTUALLY/RESPONSE constraints that haven't been satisfied
        are now considered violated.
        """
        violations = []
        history = session.get_state_sequence()

        for constraint in self.constraints:
            if constraint.type not in (ConstraintType.EVENTUALLY, ConstraintType.RESPONSE):
                continue

            result = self._evaluate_constraint(constraint, history)

            if result == EvaluationResult.PENDING:
                violation = ConstraintViolation(
                    constraint_name=constraint.name,
                    constraint_type=constraint.type,
                    message=constraint.message
                    or self._format_violation_message(constraint, history),
                    details={
                        "history": history[-5:],
                        "current_state": session.current_state,
                        "proposed_state": None,
                        "reason": "session_boundary",
                    },
                )
                violations.append(violation)
                logger.info(
                    f"Constraint violated at session boundary: "
                    f"{constraint.name} ({constraint.type.value})"
                )

        return violations

    def evaluate_transition(
        self,
        session: SessionState,
        from_state: str,
        to_state: str,
    ) -> list[ConstraintViolation]:
        """Evaluate constraints for a specific transition."""
        return self.evaluate_all(session, proposed_state=to_state)

    def _evaluate_constraint(
        self,
        constraint: Constraint,
        history: list[str],
        proposed_state: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a single constraint."""
        match constraint.type:
            case ConstraintType.EVENTUALLY:
                return self._eval_eventually(constraint.target, history)

            case ConstraintType.NEVER:
                return self._eval_never(constraint.target, proposed_state)

            case ConstraintType.RESPONSE:
                return self._eval_response(
                    constraint.trigger, constraint.target, history
                )

            case ConstraintType.PRECEDENCE:
                return self._eval_precedence(
                    constraint.trigger, constraint.target, history
                )

        return EvaluationResult.PENDING

    def _eval_eventually(
        self, target: str | None, history: list[str]
    ) -> EvaluationResult:
        """F(target): Must eventually reach target state."""
        if not target:
            return EvaluationResult.PENDING

        if target in history:
            return EvaluationResult.SATISFIED

        return EvaluationResult.PENDING

    def _eval_never(
        self, target: str | None, proposed_state: str | None
    ) -> EvaluationResult:
        """G(!target): Target state must never occur."""
        if not target:
            return EvaluationResult.PENDING

        if proposed_state == target:
            return EvaluationResult.VIOLATED

        return EvaluationResult.SATISFIED

    def _eval_response(
        self,
        trigger: str | None,
        target: str | None,
        history: list[str],
    ) -> EvaluationResult:
        """G(trigger -> F(target)): If trigger occurs, target must eventually follow."""
        if not trigger or not target:
            return EvaluationResult.PENDING

        trigger_indices = [i for i, s in enumerate(history) if s == trigger]

        if not trigger_indices:
            return EvaluationResult.SATISFIED

        for trigger_idx in trigger_indices:
            found_target = False
            for j in range(trigger_idx + 1, len(history)):
                if history[j] == target:
                    found_target = True
                    break

            if not found_target:
                return EvaluationResult.PENDING

        return EvaluationResult.SATISFIED

    def _eval_precedence(
        self,
        trigger: str | None,
        target: str | None,
        history: list[str],
    ) -> EvaluationResult:
        """Target must precede trigger."""
        if not trigger or not target:
            return EvaluationResult.PENDING

        target_seen = False
        for state in history:
            if state == target:
                target_seen = True
            if state == trigger and not target_seen:
                return EvaluationResult.VIOLATED

        return EvaluationResult.SATISFIED

    def _format_violation_message(
        self,
        constraint: Constraint,
        history: list[str],
    ) -> str:
        """Format a human-readable violation message."""
        recent_history = " -> ".join(history[-5:]) if history else "(empty)"

        match constraint.type:
            case ConstraintType.EVENTUALLY:
                return (
                    f"Constraint '{constraint.name}': Must eventually reach "
                    f"'{constraint.target}'. History: {recent_history}"
                )

            case ConstraintType.NEVER:
                return (
                    f"Constraint '{constraint.name}': State '{constraint.target}' "
                    f"must never occur. History: {recent_history}"
                )

            case ConstraintType.RESPONSE:
                return (
                    f"Constraint '{constraint.name}': After '{constraint.trigger}', "
                    f"must eventually reach '{constraint.target}'. History: {recent_history}"
                )

            case ConstraintType.PRECEDENCE:
                return (
                    f"Constraint '{constraint.name}': '{constraint.trigger}' cannot occur "
                    f"before '{constraint.target}'. History: {recent_history}"
                )

        return f"Constraint '{constraint.name}' violated. History: {recent_history}"
