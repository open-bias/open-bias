"""
Tests for InterventionHandler.
"""

import pytest
from opensentinel.policy.engines.llm.intervention import InterventionHandler
from opensentinel.policy.engines.llm.models import (
    SessionContext,
    ConstraintEvaluation,
    DriftScores,
    DriftLevel,
)
from opensentinel.core.intervention.strategies import StrategyType
from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition


@pytest.fixture
def sample_workflow():
    """Sample workflow for testing."""
    return WorkflowDefinition(
        name="test-workflow",
        version="1.0",
        states=[
            {"name": "greeting", "is_initial": True, "description": "Initial greeting"},
            {"name": "resolution", "is_terminal": True},
        ],
        transitions=[
            {"from_state": "greeting", "to_state": "resolution"},
        ],
        constraints=[
            {
                "name": "be_polite",
                "type": "always",
                "condition": "agent is polite",
                "severity": "warning",
                "intervention": "politeness_reminder",
            }
        ],
        interventions={
            "politeness_reminder": "Please maintain a polite and professional tone.",
        },
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


class TestViolationMapping:
    """Tests for severity to strategy mapping."""

    def test_warning_maps_to_system_prompt(self, engine, session, nominal_drift):
        """Warning severity should map to SYSTEM_PROMPT_APPEND."""
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
        assert result.strategy_type == StrategyType.SYSTEM_PROMPT_APPEND

    def test_error_maps_to_user_message(self, engine, session, nominal_drift):
        """Error severity should map to USER_MESSAGE_INJECT."""
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
        assert result.strategy_type == StrategyType.USER_MESSAGE_INJECT

    def test_critical_maps_to_user_message_inject(self, engine, session, nominal_drift):
        """Critical severity should map to USER_MESSAGE_INJECT."""
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
        assert result.strategy_type == StrategyType.USER_MESSAGE_INJECT


class TestDriftMapping:
    """Tests for drift level to strategy mapping."""

    def test_warning_drift(self, engine, session):
        """Warning drift should map to SYSTEM_PROMPT_APPEND."""
        drift = DriftScores(
            temporal=0.4, semantic=0.5, composite=0.45, level=DriftLevel.WARNING
        )
        
        result = engine.decide(session, [], drift)
        
        assert result is not None
        assert result.strategy_type == StrategyType.SYSTEM_PROMPT_APPEND

    def test_intervention_drift(self, engine, session):
        """Intervention drift should map to USER_MESSAGE_INJECT."""
        drift = DriftScores(
            temporal=0.7, semantic=0.7, composite=0.7, level=DriftLevel.INTERVENTION
        )

        result = engine.decide(session, [], drift)

        assert result is not None
        assert result.strategy_type == StrategyType.USER_MESSAGE_INJECT

    def test_critical_drift(self, engine, session):
        """Critical drift should map to USER_MESSAGE_INJECT."""
        drift = DriftScores(
            temporal=0.9, semantic=0.9, composite=0.9, level=DriftLevel.CRITICAL
        )
        
        result = engine.decide(session, [], drift)

        assert result is not None
        assert result.strategy_type == StrategyType.USER_MESSAGE_INJECT


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
        assert result.strategy_type == StrategyType.USER_MESSAGE_INJECT


class TestSelfCorrection:
    """Tests for self-correction detection."""

    def test_self_correction_skips_intervention(self, engine, session):
        """Decreasing drift should skip intervention."""
        session.last_intervention_turn = 0  # Previous intervention happened
        session.drift_score = 0.6  # Previous drift
        session.turn_count = 5  # Past cooldown

        # Current drift is lower (agent self-correcting)
        current_drift = DriftScores(
            temporal=0.2, semantic=0.2, composite=0.2, level=DriftLevel.NOMINAL
        )

        result = engine.decide(session, [], current_drift)

        # Should return None (intervention skipped due to self-correction)
        assert result is None


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
