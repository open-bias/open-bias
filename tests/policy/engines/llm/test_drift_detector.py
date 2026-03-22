"""
Tests for DriftDetector.
"""

import pytest
from opensentinel.policy.engines.llm.drift_detector import DriftDetector
from opensentinel.policy.engines.llm.models import SessionContext, DriftLevel
from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition


@pytest.fixture
def sample_workflow():
    """Sample workflow for testing."""
    return WorkflowDefinition(
        name="test-workflow",
        states=[
            {
                "name": "greeting",
                "is_initial": True,
                "description": "Initial greeting",
            },
            {
                "name": "identify_issue",
                "description": "Identifying the problem",
            },
            {
                "name": "resolution",
                "is_terminal": True,
                "description": "Resolving the issue",
            },
        ],
        transitions=[
            {"from_state": "greeting", "to_state": "identify_issue", "priority": 1},
            {"from_state": "identify_issue", "to_state": "resolution", "priority": 1},
        ],
    )


@pytest.fixture
def detector(sample_workflow):
    """Create a DriftDetector."""
    return DriftDetector(sample_workflow)


@pytest.fixture
def session():
    """Create a test session."""
    return SessionContext(
        session_id="test-session",
        workflow_name="test-workflow",
        current_state="greeting",
    )


class TestTemporalDrift:
    """Tests for temporal drift computation."""

    def test_identical_sequences_no_drift(self, detector):
        """Identical sequences should have zero drift."""
        drift = detector.compute_temporal_drift(
            ["greeting", "identify_issue", "resolution"],
            ["greeting", "identify_issue", "resolution"],
        )
        assert drift == 0.0

    def test_completely_different_sequences(self, detector):
        """Completely different sequences should have high drift."""
        drift = detector.compute_temporal_drift(
            ["greeting", "identify_issue", "resolution"],
            ["resolution", "greeting", "identify_issue"],
        )
        assert drift > 0.5

    def test_partial_divergence(self, detector):
        """Partial divergence should have moderate drift."""
        drift = detector.compute_temporal_drift(
            ["greeting", "identify_issue", "resolution"],
            ["greeting", "resolution", "identify_issue"],
        )
        assert 0.0 < drift < 1.0

    def test_empty_sequences(self, detector):
        """Empty sequences should have zero drift unless only expected is empty."""
        assert detector.compute_temporal_drift([], []) == 0.0
        assert detector.compute_temporal_drift(["a"], []) == 0.0
        assert detector.compute_temporal_drift([], ["a"]) == 0.5

    def test_decay_weighting(self, detector):
        """Recent deviations should be weighted more heavily."""
        # Early deviation
        drift_early = detector.compute_temporal_drift(
            ["A", "B", "C", "D"],
            ["X", "B", "C", "D"],
        )
        
        # Late deviation
        drift_late = detector.compute_temporal_drift(
            ["A", "B", "C", "D"],
            ["A", "B", "C", "X"],
        )
        
        # Late deviation should contribute more to drift
        assert drift_late > drift_early


class TestExpectedSequence:
    """Tests for expected sequence computation."""

    def test_expected_sequence_from_workflow(self, detector):
        """Test expected sequence matches workflow transitions."""
        seq = detector.expected_sequence
        assert seq == ["greeting", "identify_issue", "resolution"]

    def test_expected_sequence_cached(self, detector):
        """Test expected sequence is cached."""
        seq1 = detector.expected_sequence
        seq2 = detector.expected_sequence
        assert seq1 is seq2


class TestCompositeDrift:
    """Tests for composite drift computation."""

    def test_nominal_drift(self, detector, session):
        """Test nominal drift level."""
        # On-track session
        session.turn_window = [
            {"role": "assistant", "message": "Hello! How can I help?"},
        ]
        
        drift = detector.compute_drift(session, "Let me help you", [])
        
        # Without embeddings, should just be temporal drift
        assert drift.level in [DriftLevel.NOMINAL, DriftLevel.WARNING]

    def test_composite_weights(self, detector, session):
        """Test that composite respects configured weights."""
        # Set a specific temporal weight
        detector.temporal_weight = 0.7
        
        drift = detector.compute_drift(session, "Test message", [])
        
        # Composite should be weighted sum
        expected = drift.temporal * 0.7 + drift.semantic * 0.3
        assert abs(drift.composite - expected) < 0.001

    def test_anomaly_flag_unexpected_tool(self, detector, session):
        """Test anomaly flag for unexpected tool call."""
        drift = detector.compute_drift(
            session,
            "Searching...",
            tool_calls=["unexpected_tool"],
            expected_tool_calls=["search_kb"],
        )
        
        assert drift.anomaly_flags.get("unexpected_tool_call") is True

    def test_anomaly_flag_missing_tool(self, detector, session):
        """Test anomaly flag for missing expected tool call."""
        drift = detector.compute_drift(
            session,
            "I did something",
            tool_calls=["other_tool"],
            expected_tool_calls=["search_kb"],
        )
        
        assert drift.anomaly_flags.get("missing_expected_tool_call") is True


class TestDriftLevels:
    """Tests for drift level thresholds."""

    def test_drift_level_from_scores(self):
        """Test drift level determination from composite score."""
        from opensentinel.policy.engines.llm.models import DriftScores
        
        # Nominal
        d1 = DriftScores.from_scores(0.1, 0.2)
        assert d1.level == DriftLevel.NOMINAL
        
        # Warning
        d2 = DriftScores.from_scores(0.4, 0.5)
        assert d2.level == DriftLevel.WARNING
        
        # Intervention
        d3 = DriftScores.from_scores(0.7, 0.7)
        assert d3.level == DriftLevel.INTERVENTION
        
        # Critical
        d4 = DriftScores.from_scores(0.9, 0.9)
        assert d4.level == DriftLevel.CRITICAL
