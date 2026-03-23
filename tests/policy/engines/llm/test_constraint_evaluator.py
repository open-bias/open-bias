"""
Tests for LLMConstraintEvaluator.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openbias.policy.engines.llm.constraint_evaluator import LLMConstraintEvaluator
from openbias.policy.engines.llm.llm_client import LLMClient, LLMClientError
from openbias.policy.engines.llm.models import SessionContext, ConstraintEvaluation
from openbias.policy.engines.fsm.workflow.schema import WorkflowDefinition


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

    async def test_no_violations(self, sample_workflow, mock_llm_client, session):
        """Test evaluation with no violations."""
        mock_llm_client.complete_json.return_value = [
            {"constraint_id": "no_rude_behavior", "violated": False, "confidence": 0.9, "evidence": "", "severity": "warning"},
            {"constraint_id": "no_share_password", "violated": False, "confidence": 0.95, "evidence": "", "severity": "critical"},
        ]

        evaluator = LLMConstraintEvaluator(mock_llm_client, sample_workflow)
        evals = await evaluator.evaluate(session, "Hello, how are you?", [])

        assert all(not e.violated for e in evals)

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


class TestResponseConstraintSelection:
    """Tests for RESPONSE constraint selection logic (Tasks 54 and 55)."""

    @pytest.fixture
    def response_workflow(self):
        """Workflow with a RESPONSE constraint: after 'trigger_state', must reach 'target_state'."""
        return WorkflowDefinition(
            name="response-test-workflow",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "trigger_state"},
                {"name": "target_state"},
                {"name": "other_state"},
            ],
            transitions=[
                {"from_state": "start", "to_state": "trigger_state"},
                {"from_state": "trigger_state", "to_state": "target_state"},
                {"from_state": "trigger_state", "to_state": "other_state"},
                {"from_state": "other_state", "to_state": "trigger_state"},
            ],
            constraints=[
                {
                    "name": "must_reach_target",
                    "type": "response",
                    "trigger": "trigger_state",
                    "target": "target_state",
                    "message": "After trigger_state, target_state must follow.",
                },
            ],
        )

    def _make_session(self, states: list[str]) -> SessionContext:
        """Build a session with the given ordered list of to_states as history."""
        from openbias.policy.engines.llm.models import ConfidenceTier
        session = SessionContext(
            session_id="test",
            workflow_name="response-test-workflow",
            current_state=states[-1] if states else "start",
        )
        prev = "start"
        for state in states:
            session.record_transition(prev, state, 0.9, ConfidenceTier.CONFIDENT, 0.0)
            prev = state
        return session

    def test_response_active_when_trigger_occurred_no_target(self, response_workflow, mock_llm_client):
        """RESPONSE constraint is active when trigger has occurred and target has not."""
        session = self._make_session(["trigger_state", "other_state"])
        evaluator = LLMConstraintEvaluator(mock_llm_client, response_workflow)
        active = evaluator._select_active_constraints(session)
        assert any(c.name == "must_reach_target" for c in active)

    def test_response_inactive_when_target_followed_trigger(self, response_workflow, mock_llm_client):
        """RESPONSE constraint is inactive when target appeared after trigger."""
        session = self._make_session(["trigger_state", "target_state"])
        evaluator = LLMConstraintEvaluator(mock_llm_client, response_workflow)
        active = evaluator._select_active_constraints(session)
        assert not any(c.name == "must_reach_target" for c in active)

    def test_response_active_when_trigger_fires_again_after_target(self, response_workflow, mock_llm_client):
        """RESPONSE constraint is active when trigger fires again after a previous target — last trigger has no following target."""
        # History: trigger → target → other → trigger  (last trigger has no target yet)
        session = self._make_session(["trigger_state", "target_state", "other_state", "trigger_state"])
        evaluator = LLMConstraintEvaluator(mock_llm_client, response_workflow)
        active = evaluator._select_active_constraints(session)
        assert any(c.name == "must_reach_target" for c in active)

    def test_response_inactive_when_last_trigger_has_target(self, response_workflow, mock_llm_client):
        """RESPONSE constraint is inactive when the last trigger occurrence is followed by a target."""
        # History: trigger → other → trigger → target  (last trigger is satisfied)
        session = self._make_session(["trigger_state", "other_state", "trigger_state", "target_state"])
        evaluator = LLMConstraintEvaluator(mock_llm_client, response_workflow)
        active = evaluator._select_active_constraints(session)
        assert not any(c.name == "must_reach_target" for c in active)

    def test_response_self_match_not_satisfied(self, mock_llm_client):
        """RESPONSE constraint where trigger == target is never self-satisfied."""
        from openbias.policy.engines.llm.models import ConfidenceTier
        workflow = WorkflowDefinition(
            name="self-match-workflow",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "loop_state"},
            ],
            transitions=[
                {"from_state": "start", "to_state": "loop_state"},
                {"from_state": "loop_state", "to_state": "loop_state"},
            ],
            constraints=[
                {
                    "name": "self_response",
                    "type": "response",
                    "trigger": "loop_state",
                    "target": "loop_state",
                    "message": "trigger == target edge case",
                },
            ],
        )
        session = SessionContext(
            session_id="self-match-test",
            workflow_name="self-match-workflow",
            current_state="start",
        )
        session.record_transition("start", "loop_state", 0.9, ConfidenceTier.CONFIDENT, 0.0)

        evaluator = LLMConstraintEvaluator(mock_llm_client, workflow)
        active = evaluator._select_active_constraints(session)
        # Constraint must remain active — the trigger itself doesn't count as the target
        assert any(c.name == "self_response" for c in active)


class TestErrorHandling:
    """Tests for error handling."""

    async def test_llm_error_continues(self, sample_workflow, mock_llm_client, session):
        """Test that LLM errors don't crash evaluation."""
        mock_llm_client.complete_json.side_effect = LLMClientError("API error")
        
        evaluator = LLMConstraintEvaluator(mock_llm_client, sample_workflow)
        evals = await evaluator.evaluate(session, "Hello", [])
        
        # Should return empty list, not crash
        assert evals == []
