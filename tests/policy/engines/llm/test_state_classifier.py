"""
Tests for LLMStateClassifier.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openbias.policy.engines.llm.state_classifier import LLMStateClassifier
from openbias.policy.engines.llm.llm_client import LLMClient, LLMClientError
from openbias.policy.engines.llm.models import (
    SessionContext,
    ConfidenceTier,
)
from openbias.policy.engines.fsm.workflow.schema import WorkflowDefinition


@pytest.fixture
def sample_workflow():
    """Sample workflow for testing."""
    return WorkflowDefinition(
        name="test-workflow",
        states=[
            {
                "name": "greeting",
                "is_initial": True,
                "classification": {"patterns": ["hello", "hi"]},
            },
            {
                "name": "identify_issue",
                "classification": {"tool_calls": ["search_kb"]},
            },
            {
                "name": "resolution",
                "is_terminal": True,
                "classification": {"patterns": ["resolved"]},
            },
        ],
        transitions=[
            {"from_state": "greeting", "to_state": "identify_issue"},
            {"from_state": "identify_issue", "to_state": "resolution"},
        ],
    )


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock(spec=LLMClient)
    client.complete_json = AsyncMock()
    return client


@pytest.fixture
def session():
    """Create a test session."""
    return SessionContext(
        session_id="test-session",
        workflow_name="test-workflow",
        current_state="greeting",
    )


class TestClassification:
    """Tests for state classification."""

    async def test_confident_classification(
        self, sample_workflow, mock_llm_client, session
    ):
        """Test classification with high confidence."""
        mock_llm_client.complete_json.return_value = [
            {"state_id": "identify_issue", "confidence": 0.95, "reasoning": "Tool call detected"}
        ]
        
        classifier = LLMStateClassifier(mock_llm_client, sample_workflow)
        result = await classifier.classify(session, "Let me search", ["search_kb"])
        
        assert result.best_state == "identify_issue"
        assert result.best_confidence == 0.95
        assert result.tier == ConfidenceTier.CONFIDENT

    async def test_uncertain_classification(
        self, sample_workflow, mock_llm_client, session
    ):
        """Test classification with medium confidence."""
        mock_llm_client.complete_json.return_value = [
            {"state_id": "identify_issue", "confidence": 0.65, "reasoning": "Maybe searching"}
        ]
        
        classifier = LLMStateClassifier(mock_llm_client, sample_workflow)
        result = await classifier.classify(session, "I might look into that", [])
        
        assert result.tier == ConfidenceTier.UNCERTAIN

    async def test_lost_classification(
        self, sample_workflow, mock_llm_client, session
    ):
        """Test classification with low confidence."""
        mock_llm_client.complete_json.return_value = [
            {"state_id": "greeting", "confidence": 0.3, "reasoning": "Unclear response"}
        ]
        
        classifier = LLMStateClassifier(mock_llm_client, sample_workflow)
        result = await classifier.classify(session, "Hmm...", [])
        
        assert result.tier == ConfidenceTier.LOST


class TestTransitionLegality:
    """Tests for transition legality checking."""

    async def test_legal_transition(
        self, sample_workflow, mock_llm_client, session
    ):
        """Test a legal transition."""
        mock_llm_client.complete_json.return_value = [
            {"state_id": "identify_issue", "confidence": 0.9, "reasoning": ""}
        ]
        
        classifier = LLMStateClassifier(mock_llm_client, sample_workflow)
        result = await classifier.classify(session, "Searching...", ["search_kb"])
        
        assert result.transition_legal is True

    async def test_illegal_transition(
        self, sample_workflow, mock_llm_client, session
    ):
        """Test an illegal transition (skipping states)."""
        mock_llm_client.complete_json.return_value = [
            {"state_id": "resolution", "confidence": 0.85, "reasoning": ""}
        ]
        
        classifier = LLMStateClassifier(mock_llm_client, sample_workflow)
        result = await classifier.classify(session, "All done!", [])
        
        # greeting -> resolution is not defined
        assert result.transition_legal is False


class TestSkipViolations:
    """Tests for skip violation detection."""

    async def test_detect_skipped_states(
        self, sample_workflow, mock_llm_client, session
    ):
        """Test detection of skipped intermediate states."""
        mock_llm_client.complete_json.return_value = [
            {"state_id": "resolution", "confidence": 0.85, "reasoning": ""}
        ]
        
        classifier = LLMStateClassifier(mock_llm_client, sample_workflow)
        result = await classifier.classify(session, "All done!", [])
        
        # Should detect that identify_issue was skipped
        assert "identify_issue" in result.skip_violations


class TestErrorHandling:
    """Tests for error handling."""

    async def test_llm_error_fallback(
        self, sample_workflow, mock_llm_client, session
    ):
        """Test fallback when LLM call fails."""
        mock_llm_client.complete_json.side_effect = LLMClientError("API error")
        
        classifier = LLMStateClassifier(mock_llm_client, sample_workflow)
        result = await classifier.classify(session, "Hello", [])
        
        # Should fallback to current state with lost confidence
        assert result.best_state == "greeting"
        assert result.tier == ConfidenceTier.LOST

    async def test_invalid_state_id_ignored(
        self, sample_workflow, mock_llm_client, session
    ):
        """Test that invalid state IDs are ignored."""
        mock_llm_client.complete_json.return_value = [
            {"state_id": "nonexistent", "confidence": 0.9, "reasoning": ""},
            {"state_id": "greeting", "confidence": 0.7, "reasoning": ""},
        ]
        
        classifier = LLMStateClassifier(mock_llm_client, sample_workflow)
        result = await classifier.classify(session, "Hello", [])
        
        # Should pick the valid state
        assert result.best_state == "greeting"
