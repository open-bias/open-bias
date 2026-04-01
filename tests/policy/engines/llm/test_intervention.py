"""
Tests for InterventionHandler.
"""

import pytest
from openbias.policy.engines.llm.intervention import InterventionHandler
from openbias.policy.engines.llm.models import (
    SessionContext,
    ConstraintEvaluation,
    DriftScores,
    DriftLevel,
)
from openbias.policy.engines.fsm.workflow.schema import WorkflowDefinition


@pytest.fixture
def sample_workflow():
    """Sample workflow for testing."""
    return WorkflowDefinition(
        name="test-workflow",
        states=[
            {"name": "greeting", "is_initial": True, "description": "Initial greeting"},
            {"name": "resolution", "is_terminal": True},
        ],
        transitions=[
            {"from_state": "greeting", "to_state": "resolution"},
        ],
        constraints=[
            {
                "name": "no_rude_behavior",
                "type": "never",
                "target": "rude_behavior",
                "message": "Please maintain a polite and professional tone.",
            }
        ],
    )


@pytest.fixture
def engine(sample_workflow):
    """Create an InterventionHandler."""
    return InterventionHandler(sample_workflow, cooldown_turns=2)


@pytest.fixture
def session():
    """Create a test session."""
    return SessionContext(
        session_id="test-session",
        workflow_name="test-workflow",
        current_state="greeting",
        turn_count=5,
        last_intervention_turn=0,
    )


@pytest.fixture
def nominal_drift():
    """Create nominal drift scores."""
    return DriftScores(
        temporal=0.1,
        semantic=0.1,
        composite=0.1,
        level=DriftLevel.NOMINAL,
    )


class TestNoIntervention:
    """Tests for when no intervention is needed."""

    def test_no_violations_nominal_drift(self, engine, session, nominal_drift):
        """No intervention when no violations and nominal drift."""
        result = engine.decide(session, [], nominal_drift)
        assert result is None

    def test_cooldown_blocks_intervention(self, engine, session):
        """Cooldown should block non-critical interventions."""
        session.turn_count = 3
        session.last_intervention_turn = 2  # Only 1 turn since last

        violations = [
            ConstraintEvaluation(
                constraint_id="test",
                violated=True,
                confidence=0.9,
                evidence="Minor issue",
                severity="warning",
            )
        ]

        warning_drift = DriftScores(
            temporal=0.4, semantic=0.4, composite=0.4, level=DriftLevel.WARNING
        )

        result = engine.decide(session, violations, warning_drift)
        assert result is None

    def test_cooldown_blocks_at_exact_boundary(self, engine, session):
        """Cooldown should block when turns_since == cooldown_turns."""
        session.turn_count = 4
        session.last_intervention_turn = 2  # Exactly 2 turns since last (== cooldown_turns)

        violations = [
            ConstraintEvaluation(
                constraint_id="test",
                violated=True,
                confidence=0.9,
                evidence="Minor issue",
                severity="warning",
            )
        ]

        warning_drift = DriftScores(
            temporal=0.4, semantic=0.4, composite=0.4, level=DriftLevel.WARNING
        )

        result = engine.decide(session, violations, warning_drift)
        assert result is None

    def test_cooldown_expires_after_boundary(self, engine, session):
        """Cooldown should expire when turns_since > cooldown_turns."""
        session.turn_count = 5
        session.last_intervention_turn = 2  # 3 turns since last (> cooldown_turns=2)

        violations = [
            ConstraintEvaluation(
                constraint_id="test",
                violated=True,
                confidence=0.9,
                evidence="Minor issue",
                severity="warning",
            )
        ]

        warning_drift = DriftScores(
            temporal=0.4, semantic=0.4, composite=0.4, level=DriftLevel.WARNING
        )

        result = engine.decide(session, violations, warning_drift)
        assert result is not None


class TestViolationIntervention:
    """Tests for violation-triggered interventions."""

    def test_warning_returns_message(self, engine, session, nominal_drift):
        """Warning violation returns a message template."""
        violations = [
            ConstraintEvaluation(
                constraint_id="test",
                violated=True,
                confidence=0.9,
                evidence="Minor issue",
                severity="warning",
            )
        ]

        result = engine.decide(session, violations, nominal_drift)

        assert result is not None
        assert isinstance(result, str)

    def test_error_returns_message(self, engine, session, nominal_drift):
        """Error violation returns a message template."""
        violations = [
            ConstraintEvaluation(
                constraint_id="test",
                violated=True,
                confidence=0.9,
                evidence="Significant issue",
                severity="error",
            )
        ]

        result = engine.decide(session, violations, nominal_drift)

        assert result is not None
        assert isinstance(result, str)

    def test_critical_returns_message(self, engine, session, nominal_drift):
        """Critical violation returns a message template."""
        violations = [
            ConstraintEvaluation(
                constraint_id="test",
                violated=True,
                confidence=0.95,
                evidence="Critical issue",
                severity="critical",
            )
        ]

        result = engine.decide(session, violations, nominal_drift)

        assert result is not None
        assert isinstance(result, str)


class TestDriftIntervention:
    """Tests for drift-triggered interventions."""

    def test_warning_drift(self, engine, session):
        """Warning drift returns a message template."""
        drift = DriftScores(
            temporal=0.4, semantic=0.5, composite=0.45, level=DriftLevel.WARNING
        )

        result = engine.decide(session, [], drift)

        assert result is not None
        assert isinstance(result, str)

    def test_intervention_drift(self, engine, session):
        """Intervention drift returns a message template."""
        drift = DriftScores(
            temporal=0.7, semantic=0.7, composite=0.7, level=DriftLevel.INTERVENTION
        )

        result = engine.decide(session, [], drift)

        assert result is not None
        assert isinstance(result, str)

    def test_critical_drift(self, engine, session):
        """Critical drift returns a message template."""
        drift = DriftScores(
            temporal=0.9, semantic=0.9, composite=0.9, level=DriftLevel.CRITICAL
        )

        result = engine.decide(session, [], drift)

        assert result is not None
        assert isinstance(result, str)


class TestCriticalBypassCooldown:
    """Tests for critical violations bypassing cooldown."""

    def test_critical_bypasses_cooldown(self, engine, session):
        """Critical violations should bypass cooldown."""
        session.turn_count = 3
        session.last_intervention_turn = 2  # Cooldown active

        violations = [
            ConstraintEvaluation(
                constraint_id="test",
                violated=True,
                confidence=0.95,
                evidence="Critical!",
                severity="critical",
            )
        ]

        result = engine.decide(session, violations, DriftScores.from_scores(0.1, 0.1))

        # Should NOT be None despite cooldown
        assert result is not None
        assert isinstance(result, str)


class TestSelfCorrection:
    """Tests for self-correction detection."""

    def test_self_correction_skips_intervention(self, engine, session):
        """Decreasing drift should skip intervention when no violations."""
        session.last_intervention_turn = 0  # Previous intervention happened
        session.drift_score = 0.6  # Previous drift
        session.drift_at_last_intervention = 0.6
        session.turn_count = 5  # Past cooldown

        # Current drift is lower (agent self-correcting)
        current_drift = DriftScores(
            temporal=0.2, semantic=0.2, composite=0.2, level=DriftLevel.NOMINAL
        )

        result = engine.decide(session, [], current_drift)

        # Should return None (intervention skipped due to self-correction)
        assert result is None

    def test_self_correction_does_not_suppress_critical(self, engine, session):
        """Critical violations must not be suppressed by self-correction (Task 51)."""
        session.last_intervention_turn = 0
        session.drift_at_last_intervention = 0.6
        session.turn_count = 5  # Past cooldown

        violations = [
            ConstraintEvaluation(
                constraint_id="test",
                violated=True,
                confidence=0.95,
                evidence="Critical security breach",
                severity="critical",
            )
        ]

        # Drift is decreasing — self-correction would normally fire
        current_drift = DriftScores(
            temporal=0.1, semantic=0.1, composite=0.1, level=DriftLevel.NOMINAL
        )

        result = engine.decide(session, violations, current_drift)

        # Critical violation must NOT be suppressed
        assert result is not None

    def test_self_correction_does_not_suppress_active_violation(self, engine, session):
        """Active constraint violations must not be suppressed by self-correction (Task 79)."""
        session.last_intervention_turn = 0
        session.drift_at_last_intervention = 0.6
        session.turn_count = 5  # Past cooldown

        violations = [
            ConstraintEvaluation(
                constraint_id="no_rude_behavior",
                violated=True,
                confidence=0.9,
                evidence="Agent used rude language",
                severity="warning",
            )
        ]

        # Drift is decreasing — self-correction would normally fire
        current_drift = DriftScores(
            temporal=0.1, semantic=0.1, composite=0.1, level=DriftLevel.NOMINAL
        )

        result = engine.decide(session, violations, current_drift)

        # Active violation must NOT be suppressed by self-correction
        assert result is not None


class TestEscalation:
    """Tests for escalation checks."""

    def test_should_escalate_critical(self, engine):
        """Critical drift should trigger escalation."""
        drift = DriftScores(
            temporal=0.9, semantic=0.9, composite=0.9, level=DriftLevel.CRITICAL
        )

        assert engine.should_escalate(drift) is True

    def test_should_not_escalate_lower(self, engine):
        """Non-critical drift should not escalate."""
        drift = DriftScores(
            temporal=0.5, semantic=0.5, composite=0.5, level=DriftLevel.WARNING
        )

        assert engine.should_escalate(drift) is False
