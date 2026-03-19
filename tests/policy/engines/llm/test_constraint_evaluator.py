"""
Tests for LLMConstraintEvaluator.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from opensentinel.policy.engines.llm.constraint_evaluator import LLMConstraintEvaluator
from opensentinel.policy.engines.llm.llm_client import LLMClient, LLMClientError
from opensentinel.policy.engines.llm.models import SessionContext, ConstraintEvaluation
from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition


@pytest.fixture
def sample_workflow():
    """Sample workflow with constraints."""
    return WorkflowDefinition(
        name="test-workflow",
        states=[
            {"name": "greeting", "is_initial": True},
            {"name": "verify_identity"},
            {"name": "account_action"},
            {"name": "resolution", "is_terminal": True},
        ],
        transitions=[
            {"from_state": "greeting", "to_state": "verify_identity"},
            {"from_state": "verify_identity", "to_state": "account_action"},
            {"from_state": "account_action", "to_state": "resolution"},
        ],
        constraints=[
            {
                "name": "must_verify",
                "type": "precedence",
                "trigger": "account_action",
                "target": "verify_identity",
                "message": "You must verify identity before account actions.",
            },
            {
                "name": "no_share_password",
                "type": "never",
                "target": "share_credentials",
                "message": "Never share credentials with the customer.",
            },
            {
                "name": "no_rude_behavior",
                "type": "never",
                "target": "rude_behavior",
                "message": "The agent must always remain polite and professional.",
            },
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


class TestConstraintSelection:
    """Tests for constraint selection logic."""

    def test_never_constraints_always_active(self, sample_workflow, mock_llm_client, session):
        """NEVER constraints should always be active."""
        evaluator = LLMConstraintEvaluator(mock_llm_client, sample_workflow)
        active = evaluator._select_active_constraints(session)
        
        never_constraint = next(c for c in active if c.name == "no_share_password")
        assert never_constraint is not None

    def test_never_constraints_always_active_for_any_state(self, sample_workflow, mock_llm_client, session):
        """NEVER constraints should always be active regardless of current state."""
        evaluator = LLMConstraintEvaluator(mock_llm_client, sample_workflow)
        active = evaluator._select_active_constraints(session)

        never_constraint = next(c for c in active if c.name == "no_rude_behavior")
        assert never_constraint is not None

    def test_precedence_active_near_trigger(self, sample_workflow, mock_llm_client, session):
        """PRECEDENCE constraints should be active when trigger is current."""
        evaluator = LLMConstraintEvaluator(mock_llm_client, sample_workflow)
        
        # Set current state to trigger
        session.current_state = "account_action"
        active = evaluator._select_active_constraints(session)
        
        precedence_constraint = next(
            (c for c in active if c.name == "must_verify"), None
        )
        assert precedence_constraint is not None


class TestEvaluation:
    """Tests for constraint evaluation."""

    @pytest.mark.asyncio
    async def test_no_violations(self, sample_workflow, mock_llm_client, session):
        """Test evaluation with no violations."""
        mock_llm_client.complete_json.return_value = [
            {"constraint_id": "no_rude_behavior", "violated": False, "confidence": 0.9, "evidence": "", "severity": "warning"},
            {"constraint_id": "no_share_password", "violated": False, "confidence": 0.95, "evidence": "", "severity": "critical"},
        ]

        evaluator = LLMConstraintEvaluator(mock_llm_client, sample_workflow)
        evals = await evaluator.evaluate(session, "Hello, how are you?", [])

        assert all(not e.violated for e in evals)

    @pytest.mark.asyncio
    async def test_violation_detected(self, sample_workflow, mock_llm_client, session):
        """Test evaluation with a violation."""
        mock_llm_client.complete_json.return_value = [
            {
                "constraint_id": "no_rude_behavior",
                "violated": True,
                "confidence": 0.85,
                "evidence": "Rude language detected",
                "severity": "warning",
            },
        ]

        evaluator = LLMConstraintEvaluator(mock_llm_client, sample_workflow)
        evals = await evaluator.evaluate(session, "Get lost!", [])

        violations = [e for e in evals if e.violated]
        assert len(violations) > 0
        assert violations[0].evidence == "Rude language detected"


class TestEvidenceMemory:
    """Tests for evidence memory accumulation."""

    @pytest.mark.asyncio
    async def test_evidence_accumulated(self, sample_workflow, mock_llm_client, session):
        """Test that evidence is accumulated in session memory."""
        mock_llm_client.complete_json.return_value = [
            {
                "constraint_id": "no_rude_behavior",
                "violated": False,
                "confidence": 0.8,
                "evidence": "Agent greeted politely",
                "severity": "warning",
            },
        ]

        evaluator = LLMConstraintEvaluator(mock_llm_client, sample_workflow)
        await evaluator.evaluate(session, "Hello!", [])

        # Evidence should be stored
        assert "no_rude_behavior" in session.constraint_memory
        assert "Agent greeted politely" in session.constraint_memory["no_rude_behavior"]

    @pytest.mark.asyncio
    async def test_low_confidence_not_stored(self, sample_workflow, mock_llm_client, session):
        """Test that low-confidence evidence is not stored."""
        mock_llm_client.complete_json.return_value = [
            {
                "constraint_id": "no_rude_behavior",
                "violated": False,
                "confidence": 0.2,  # Below threshold
                "evidence": "Unclear",
                "severity": "warning",
            },
        ]

        evaluator = LLMConstraintEvaluator(mock_llm_client, sample_workflow)
        await evaluator.evaluate(session, "...", [])

        # Evidence should not be stored
        assert "no_rude_behavior" not in session.constraint_memory or len(session.constraint_memory["no_rude_behavior"]) == 0


class TestBatching:
    """Tests for constraint batching."""

    @pytest.mark.asyncio
    async def test_batching_large_constraint_set(self, mock_llm_client, session):
        """Test that many constraints are batched."""
        # Create workflow with many constraints
        workflow = WorkflowDefinition(
            name="test",
            states=[{"name": "initial", "is_initial": True}],
            constraints=[
                {
                    "name": f"constraint_{i}",
                    "type": "never",
                    "target": f"forbidden_state_{i}",
                    "message": f"Condition {i} must never be violated.",
                }
                for i in range(10)
            ],
        )
        
        # Return empty results for each batch
        mock_llm_client.complete_json.return_value = []
        
        evaluator = LLMConstraintEvaluator(
            mock_llm_client, workflow, max_constraints_per_batch=3
        )
        await evaluator.evaluate(session, "Test", [])
        
        # Should have made multiple calls (10 constraints / 3 per batch = 4 batches)
        assert mock_llm_client.complete_json.call_count >= 3


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_llm_error_continues(self, sample_workflow, mock_llm_client, session):
        """Test that LLM errors don't crash evaluation."""
        mock_llm_client.complete_json.side_effect = LLMClientError("API error")
        
        evaluator = LLMConstraintEvaluator(mock_llm_client, sample_workflow)
        evals = await evaluator.evaluate(session, "Hello", [])
        
        # Should return empty list, not crash
        assert evals == []
