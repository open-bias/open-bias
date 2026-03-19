"""Tests for constraint evaluation."""

import pytest
from datetime import datetime, timezone

from opensentinel.policy.engines.fsm.workflow.constraints import (
    ConstraintEvaluator,
    ConstraintViolation,
    EvaluationResult,
    get_decision,
)
from opensentinel.policy.engines.fsm.workflow.schema import Constraint, ConstraintType
from opensentinel.policy.engines.fsm.workflow.state_machine import SessionState, StateHistoryEntry
from opensentinel.policy.protocols import Decision


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
        constraints = [
            Constraint(
                name="test",
                type=ConstraintType.NEVER,
                target="forbidden",
                message="Do not enter forbidden state",
            )
        ]
        evaluator = ConstraintEvaluator(constraints)
        session = make_session(["start", "forbidden", "end"])

        violations = evaluator.evaluate_all(session)
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
        session = make_session(["start", "action", "bad"])

        violations = evaluator.evaluate_all(session)
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
        session = make_session(["start", "forbidden"])

        violations = evaluator.evaluate_all(session)
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
        session = make_session(["start", "forbidden"])

        violations = evaluator.evaluate_all(session)
        assert len(violations) == 1
        assert "forbidden" in violations[0].message


class TestModeAwareDecisions:
    """Tests for get_decision() — mode × constraint_type → Decision."""

    def test_guide_always_intervenes(self):
        for ct in ConstraintType:
            assert get_decision("guide", ct) == Decision.INTERVENE

    def test_enforce_precedence_blocks(self):
        assert get_decision("enforce", ConstraintType.PRECEDENCE) == Decision.BLOCK

    def test_enforce_never_blocks(self):
        assert get_decision("enforce", ConstraintType.NEVER) == Decision.BLOCK

    def test_enforce_eventually_intervenes(self):
        assert get_decision("enforce", ConstraintType.EVENTUALLY) == Decision.INTERVENE

    def test_enforce_response_intervenes(self):
        assert get_decision("enforce", ConstraintType.RESPONSE) == Decision.INTERVENE


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
