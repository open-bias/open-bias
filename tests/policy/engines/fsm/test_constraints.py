"""Tests for constraint evaluation."""

import pytest
from datetime import datetime, timezone

from opensentinel.policy.engines.fsm.workflow.constraints import (
    ConstraintEvaluator,
    ConstraintViolation,
    EvaluationResult,
)
from opensentinel.policy.engines.fsm.workflow.schema import Constraint, ConstraintType
from opensentinel.policy.engines.fsm.workflow.state_machine import SessionState, StateHistoryEntry


def make_session(states: list[str]) -> SessionState:
    """Helper to create a session with given state history."""
    history = [
        StateHistoryEntry(state_name=s, entered_at=datetime.now(timezone.utc)) for s in states
    ]
    return SessionState(
        session_id="test",
        workflow_name="test",
        current_state=states[-1] if states else "unknown",
        history=history,
    )


class TestConstraintTypes:
    """Tests for the 4 constraint types: PRECEDENCE, NEVER, EVENTUALLY, RESPONSE."""

    def test_constraint_type_values(self):
        """Only 4 constraint types exist."""
        assert set(ct.value for ct in ConstraintType) == {
            "precedence", "never", "eventually", "response",
        }

    # --- EVENTUALLY ---

    def test_eventually_satisfied(self):
        constraints = [
            Constraint(name="test", type=ConstraintType.EVENTUALLY, target="goal")
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "middle", "goal"])

        assert evaluator.evaluate_all(session) == []

    def test_eventually_pending(self):
        """EVENTUALLY returns no violations mid-session (PENDING, not VIOLATED)."""
        constraints = [
            Constraint(name="test", type=ConstraintType.EVENTUALLY, target="goal")
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "middle"])

        assert evaluator.evaluate_all(session) == []

    # --- NEVER ---

    def test_never_satisfied(self):
        constraints = [
            Constraint(name="test", type=ConstraintType.NEVER, target="forbidden")
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "middle", "end"])

        assert evaluator.evaluate_all(session) == []

    def test_never_violated(self):
        """NEVER fires when the proposed state matches the forbidden target."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.NEVER,
                target="forbidden",
                message="Do not enter forbidden state",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start"])

        violations = evaluator.evaluate_all(session, proposed_state="forbidden")
        assert len(violations) == 1
        assert violations[0].constraint_name == "test"
        assert violations[0].constraint_type == ConstraintType.NEVER
        assert violations[0].message == "Do not enter forbidden state"

    def test_never_with_proposed_state(self):
        """NEVER fires when the proposed state is the forbidden one."""
        constraints = [
            Constraint(name="test", type=ConstraintType.NEVER, target="forbidden")
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start"])

        violations = evaluator.evaluate_all(session, proposed_state="forbidden")
        assert len(violations) == 1

    # --- PRECEDENCE ---

    def test_precedence_satisfied(self):
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.PRECEDENCE,
                trigger="action",
                target="verify",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "verify", "action", "end"])

        assert evaluator.evaluate_all(session) == []

    def test_precedence_violated(self):
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.PRECEDENCE,
                trigger="action",
                target="verify",
                message="Must verify first",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "action", "verify", "end"])

        violations = evaluator.evaluate_all(session)
        assert len(violations) == 1
        assert violations[0].constraint_name == "test"

    def test_precedence_with_proposed_state(self):
        """PRECEDENCE fires when proposing trigger without prior target."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.PRECEDENCE,
                trigger="action",
                target="verify",
                message="Must verify first",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start"])

        violations = evaluator.evaluate_all(session, proposed_state="action")
        assert len(violations) == 1

    def test_precedence_vacuously_satisfied(self):
        """If trigger never appears, PRECEDENCE is satisfied."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.PRECEDENCE,
                trigger="action",
                target="verify",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "middle", "end"])

        assert evaluator.evaluate_all(session) == []

    # --- RESPONSE ---

    def test_response_satisfied(self):
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.RESPONSE,
                trigger="request",
                target="acknowledge",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "request", "acknowledge", "end"])

        assert evaluator.evaluate_all(session) == []

    def test_response_no_trigger(self):
        """Vacuously satisfied when trigger never occurs."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.RESPONSE,
                trigger="request",
                target="acknowledge",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "middle", "end"])

        assert evaluator.evaluate_all(session) == []

    def test_response_pending(self):
        """RESPONSE is PENDING when trigger seen but target not yet."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.RESPONSE,
                trigger="request",
                target="acknowledge",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "request", "middle"])

        # Mid-session: PENDING, no violation
        assert evaluator.evaluate_all(session) == []

    # --- Multiple constraints ---

    def test_multiple_constraints_all_satisfied(self):
        constraints = [
            Constraint(name="never_bad", type=ConstraintType.NEVER, target="bad"),
            Constraint(
                name="verify_first",
                type=ConstraintType.PRECEDENCE,
                trigger="action",
                target="verify",
            ),
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "verify", "action", "end"])

        assert evaluator.evaluate_all(session) == []

    def test_multiple_violations(self):
        constraints = [
            Constraint(name="never_bad", type=ConstraintType.NEVER, target="bad"),
            Constraint(
                name="verify_first",
                type=ConstraintType.PRECEDENCE,
                trigger="action",
                target="verify",
            ),
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "action"])

        violations = evaluator.evaluate_all(session, proposed_state="bad")
        assert len(violations) == 2

    # --- Violation details ---

    def test_violation_fields(self):
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.NEVER,
                target="forbidden",
                message="Do not go there",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start"])

        violations = evaluator.evaluate_all(session, proposed_state="forbidden")
        assert len(violations) == 1
        v = violations[0]
        assert v.constraint_name == "test"
        assert v.constraint_type == ConstraintType.NEVER
        assert v.message == "Do not go there"
        assert "forbidden" in str(v.details)

    def test_violation_auto_message(self):
        """When no message is set, a descriptive one is auto-generated."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.NEVER,
                target="forbidden",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start"])

        violations = evaluator.evaluate_all(session, proposed_state="forbidden")
        assert len(violations) == 1
        assert "forbidden" in violations[0].message


class TestSessionBoundaryEvaluation:
    """Tests for evaluate_session_boundary()."""

    def test_eventually_violated_at_boundary(self):
        """EVENTUALLY that never reached target fires at session boundary."""
        constraints = [
            Constraint(
                name="must_resolve",
                type=ConstraintType.EVENTUALLY,
                target="resolve",
                message="Must resolve",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "middle", "end"])

        violations = evaluator.evaluate_session_boundary(session)
        assert len(violations) == 1
        assert violations[0].constraint_name == "must_resolve"

    def test_eventually_satisfied_no_boundary_violation(self):
        """EVENTUALLY already satisfied does not fire at boundary."""
        constraints = [
            Constraint(
                name="must_resolve",
                type=ConstraintType.EVENTUALLY,
                target="resolve",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "resolve", "end"])

        assert evaluator.evaluate_session_boundary(session) == []

    def test_response_pending_at_boundary(self):
        """RESPONSE with unresolved trigger fires at boundary."""
        constraints = [
            Constraint(
                name="if_escalate_notify",
                type=ConstraintType.RESPONSE,
                trigger="escalate",
                target="notify",
                message="Must notify after escalation",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "escalate", "end"])

        violations = evaluator.evaluate_session_boundary(session)
        assert len(violations) == 1
        assert violations[0].constraint_name == "if_escalate_notify"

    def test_response_satisfied_no_boundary_violation(self):
        constraints = [
            Constraint(
                name="if_escalate_notify",
                type=ConstraintType.RESPONSE,
                trigger="escalate",
                target="notify",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "escalate", "notify", "end"])

        assert evaluator.evaluate_session_boundary(session) == []

    def test_response_no_trigger_no_boundary_violation(self):
        """If trigger never occurred, RESPONSE is vacuously satisfied at boundary."""
        constraints = [
            Constraint(
                name="if_escalate_notify",
                type=ConstraintType.RESPONSE,
                trigger="escalate",
                target="notify",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "middle", "end"])

        assert evaluator.evaluate_session_boundary(session) == []

    def test_boundary_ignores_precedence_and_never(self):
        """Session boundary only checks EVENTUALLY and RESPONSE."""
        constraints = [
            Constraint(
                name="never_bad",
                type=ConstraintType.NEVER,
                target="bad",
            ),
            Constraint(
                name="verify_first",
                type=ConstraintType.PRECEDENCE,
                trigger="action",
                target="verify",
            ),
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "end"])

        assert evaluator.evaluate_session_boundary(session) == []


class TestNeverNoPoisoning:
    """NEVER constraints must not permanently poison session history."""

    def test_never_does_not_fire_on_past_history(self):
        """NEVER should not fire when forbidden state is only in history, not proposed."""
        constraints = [
            Constraint(name="no_forbidden", type=ConstraintType.NEVER, target="forbidden")
        ]
        evaluator = ConstraintEvaluator(constraints)
        # Even if forbidden is in history, without proposing it again, no violation
        session = make_session(["start", "forbidden", "safe"])

        violations = evaluator.evaluate_all(session)
        assert violations == []

    def test_never_fires_only_on_proposed_state(self):
        """NEVER fires when forbidden state is proposed, regardless of history."""
        constraints = [
            Constraint(name="no_forbidden", type=ConstraintType.NEVER, target="forbidden")
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "safe"])

        violations = evaluator.evaluate_all(session, proposed_state="forbidden")
        assert len(violations) == 1

    def test_never_does_not_fire_on_unrelated_proposed_state(self):
        """NEVER does not fire when a different state is proposed."""
        constraints = [
            Constraint(name="no_forbidden", type=ConstraintType.NEVER, target="forbidden")
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start"])

        violations = evaluator.evaluate_all(session, proposed_state="safe")
        assert violations == []

    def test_never_no_permanent_poisoning(self):
        """After a forbidden state was proposed and blocked, subsequent safe states pass."""
        constraints = [
            Constraint(name="no_forbidden", type=ConstraintType.NEVER, target="forbidden")
        ]
        evaluator = ConstraintEvaluator(constraints)

        # First: propose forbidden -> violation
        session = make_session(["start"])
        violations = evaluator.evaluate_all(session, proposed_state="forbidden")
        assert len(violations) == 1

        # Simulate: transition was blocked, session stays at "start"
        # Now propose a safe state -> no violation
        violations = evaluator.evaluate_all(session, proposed_state="safe")
        assert violations == []

    def test_never_without_proposed_state_always_satisfied(self):
        """Without a proposed state, NEVER is always satisfied (nothing to check)."""
        constraints = [
            Constraint(name="no_forbidden", type=ConstraintType.NEVER, target="forbidden")
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "middle", "end"])

        violations = evaluator.evaluate_all(session)
        assert violations == []


class TestNullTargetTriggerSatisfied:
    """Task 86: Null target/trigger returns SATISFIED, not PENDING.

    Schema validation normally prevents None target/trigger, so we test
    the internal evaluator methods directly as a defensive-coding check.
    """

    def test_eventually_none_target_satisfied(self):
        """EVENTUALLY with no target is trivially satisfied (not PENDING)."""
        evaluator = ConstraintEvaluator([])
        result = evaluator._eval_eventually(None, ["start", "middle"])
        assert result == EvaluationResult.SATISFIED

    def test_never_none_target_satisfied(self):
        """NEVER with no target is trivially satisfied."""
        evaluator = ConstraintEvaluator([])
        result = evaluator._eval_never(None, "anything")
        assert result == EvaluationResult.SATISFIED

    def test_response_none_trigger_satisfied(self):
        """RESPONSE with no trigger is trivially satisfied."""
        evaluator = ConstraintEvaluator([])
        result = evaluator._eval_response(None, "ack", ["start"])
        assert result == EvaluationResult.SATISFIED

    def test_response_none_target_satisfied(self):
        """RESPONSE with no target is trivially satisfied."""
        evaluator = ConstraintEvaluator([])
        result = evaluator._eval_response("req", None, ["start", "req"])
        assert result == EvaluationResult.SATISFIED

    def test_precedence_none_trigger_satisfied(self):
        """PRECEDENCE with no trigger is trivially satisfied."""
        evaluator = ConstraintEvaluator([])
        result = evaluator._eval_precedence(None, "v", ["start"])
        assert result == EvaluationResult.SATISFIED

    def test_precedence_none_target_satisfied(self):
        """PRECEDENCE with no target is trivially satisfied."""
        evaluator = ConstraintEvaluator([])
        result = evaluator._eval_precedence("a", None, ["start", "a"])
        assert result == EvaluationResult.SATISFIED

    def test_none_target_not_violated_at_boundary(self):
        """EVENTUALLY with None target should not fire at session boundary.

        Uses model_construct to bypass pydantic validation for this edge case.
        """
        constraint = Constraint.model_construct(
            name="test", type=ConstraintType.EVENTUALLY, target=None, message=""
        )
        evaluator = ConstraintEvaluator([constraint])
        session = make_session(["start", "end"])

        # Previously this returned PENDING which boundary treated as violated
        assert evaluator.evaluate_session_boundary(session) == []


class TestEvaluateTransitionSignature:
    """Task 59: evaluate_transition no longer accepts from_state."""

    def test_evaluate_transition_without_from_state(self):
        """evaluate_transition works with just session and to_state."""
        constraints = [
            Constraint(name="no_bad", type=ConstraintType.NEVER, target="bad")
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start"])

        # Should work without from_state
        violations = evaluator.evaluate_transition(session, "safe")
        assert violations == []

        violations = evaluator.evaluate_transition(session, "bad")
        assert len(violations) == 1

    def test_evaluate_transition_delegates_to_evaluate_all(self):
        """evaluate_transition delegates to evaluate_all with proposed_state."""
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.PRECEDENCE,
                trigger="action",
                target="verify",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start"])

        violations = evaluator.evaluate_transition(session, "action")
        assert len(violations) == 1
