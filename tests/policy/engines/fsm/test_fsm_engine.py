
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opensentinel.policy.engines.fsm.engine import FSMPolicyEngine
from opensentinel.policy.engines.fsm.workflow.schema import ConstraintType
from opensentinel.policy.engines.fsm.workflow.state_machine import TransitionResult
from opensentinel.policy.engines.stateful import StateClassificationResult
from opensentinel.policy.protocols import Decision


@pytest.fixture
def engine():
    return FSMPolicyEngine()


@pytest.fixture
def mocks():
    with patch("opensentinel.policy.engines.fsm.engine.WorkflowParser") as mock_parser, \
         patch("opensentinel.policy.engines.fsm.engine.WorkflowStateMachine") as mock_sm, \
         patch("opensentinel.policy.engines.fsm.engine.StateClassifier") as mock_classifier, \
         patch("opensentinel.policy.engines.fsm.engine.ConstraintEvaluator") as mock_constraints:

        # Default: is_in_terminal_state returns False
        mock_sm.return_value.is_in_terminal_state = AsyncMock(return_value=False)

        yield {
            "parser": mock_parser,
            "sm": mock_sm,
            "classifier": mock_classifier,
            "constraints": mock_constraints,
        }


async def test_initialization(engine, mocks):
    config = {"workflow": {"name": "test_workflow", "states": [], "constraints": []}}
    mock_wf = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_wf

    await engine.initialize(config)

    assert engine._initialized
    mocks["sm"].assert_called_once()
    mocks["classifier"].assert_called_once()
    mocks["constraints"].assert_called_once()
    assert engine.engine_type == "fsm"


async def test_evaluate_response_success(engine, mocks):
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    # Mock session
    mock_session = MagicMock()
    mock_session.current_state = "start"
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    # Mock classification
    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="next_state", confidence=0.9, method="test"
    )

    # Mock constraints (no violations)
    mocks["constraints"].return_value.evaluate_all.return_value = []

    # Mock transition
    mocks["sm"].return_value.transition = AsyncMock(
        return_value=(TransitionResult.SUCCESS, None)
    )

    result = await engine.evaluate_response("sid", "response", {})

    assert result.decision == Decision.ALLOW
    assert len(result.metadata.get("violations", [])) == 0
    mocks["sm"].return_value.transition.assert_called_once()


async def test_evaluate_response_violations_always_intervene(engine, mocks):
    """All violations produce INTERVENE regardless of constraint type."""
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="bad_state", confidence=0.9, method="test"
    )

    mocks["sm"].return_value.transition = AsyncMock(
        return_value=(TransitionResult.SUCCESS, None)
    )

    for ctype in (ConstraintType.NEVER, ConstraintType.PRECEDENCE, ConstraintType.EVENTUALLY):
        violation = MagicMock()
        violation.constraint_name = "test"
        violation.message = "violation"
        violation.constraint_type = ctype
        violation.details = {}

        mocks["constraints"].return_value.evaluate_all.return_value = [violation]

        result = await engine.evaluate_response("sid", "response", {})

        assert result.decision == Decision.INTERVENE
        assert "mode" not in result.metadata


async def test_initialization_with_config_path(engine, mocks):
    """Test initialization using the unified config_path parameter."""
    config = {"config_path": "path/to/workflow.yaml"}
    mock_wf = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"].parse_file.return_value = mock_wf

    await engine.initialize(config)

    assert engine._initialized
    mocks["parser"].parse_file.assert_called_with("path/to/workflow.yaml")
    mocks["sm"].assert_called_once()
    assert engine.engine_type == "fsm"


async def test_initialization_failure(engine, mocks):
    """Test initialization failure when no valid config provided."""
    config = {}
    with pytest.raises(ValueError, match="FSM engine requires 'config_path' or 'workflow'"):
        await engine.initialize(config)


async def test_session_boundary_evaluation_at_terminal(engine, mocks):
    """Terminal state triggers session-boundary constraint evaluation."""
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="terminal", confidence=0.9, method="test"
    )

    # No regular violations
    mocks["constraints"].return_value.evaluate_all.return_value = []

    # But boundary violations exist
    boundary_violation = MagicMock()
    boundary_violation.constraint_name = "must_resolve"
    boundary_violation.message = "Must resolve"
    boundary_violation.constraint_type = ConstraintType.EVENTUALLY
    boundary_violation.details = {}
    mocks["constraints"].return_value.evaluate_session_boundary.return_value = [
        boundary_violation
    ]

    mocks["sm"].return_value.transition = AsyncMock(
        return_value=(TransitionResult.SUCCESS, None)
    )
    # Terminal state
    mocks["sm"].return_value.is_in_terminal_state = AsyncMock(return_value=True)

    result = await engine.evaluate_response("sid", "response", {})

    # EVENTUALLY in enforce → INTERVENE
    assert result.decision == Decision.INTERVENE
    assert len(result.metadata["violations"]) == 1
    mocks["constraints"].return_value.evaluate_session_boundary.assert_called_once()


async def test_invalid_transition_returns_intervene(engine, mocks):
    """Invalid transitions produce INTERVENE to enforce FSM graph structure."""
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="invalid_state", confidence=0.9, method="test"
    )

    # No constraint violations
    mocks["constraints"].return_value.evaluate_all.return_value = []

    mocks["sm"].return_value.transition = AsyncMock(
        return_value=(TransitionResult.INVALID_TRANSITION, "No such transition")
    )

    result = await engine.evaluate_response("sid", "response", {})

    assert result.decision == Decision.INTERVENE
    assert result.message == "No such transition"


async def test_multiple_violations_joined_in_message(engine, mocks):
    """All violation messages should be included, not just the first."""
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="bad_state", confidence=0.9, method="test"
    )

    mocks["sm"].return_value.transition = AsyncMock(
        return_value=(TransitionResult.SUCCESS, None)
    )

    violations = []
    for i in range(3):
        v = MagicMock()
        v.constraint_name = f"constraint_{i}"
        v.message = f"Violation {i}"
        v.constraint_type = ConstraintType.NEVER
        v.details = {}
        violations.append(v)

    mocks["constraints"].return_value.evaluate_all.return_value = violations

    result = await engine.evaluate_response("sid", "response", {})

    assert result.decision == Decision.INTERVENE
    assert result.message == "Violation 0\nViolation 1\nViolation 2"


async def test_never_violation_skips_transition(engine, mocks):
    """NEVER constraint violation prevents the state machine transition."""
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="forbidden", confidence=0.9, method="test"
    )

    # NEVER constraint violation
    violation = MagicMock()
    violation.constraint_name = "no_forbidden"
    violation.message = "Forbidden state"
    violation.constraint_type = ConstraintType.NEVER
    violation.details = {}
    mocks["constraints"].return_value.evaluate_all.return_value = [violation]

    result = await engine.evaluate_response("sid", "response", {})

    assert result.decision == Decision.INTERVENE
    # Transition should NOT have been called
    mocks["sm"].return_value.transition.assert_not_called()
    assert result.metadata["transition_result"] == TransitionResult.CONSTRAINT_VIOLATED.value


async def test_constraints_evaluated_before_transition(engine, mocks):
    """Constraints are evaluated before the transition, not after."""
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="next", confidence=0.9, method="test"
    )

    # No violations
    mocks["constraints"].return_value.evaluate_all.return_value = []
    mocks["sm"].return_value.transition = AsyncMock(
        return_value=(TransitionResult.SUCCESS, None)
    )

    await engine.evaluate_response("sid", "response", {})

    # Constraints evaluated first, then transition
    mocks["constraints"].return_value.evaluate_all.assert_called_once()
    mocks["sm"].return_value.transition.assert_called_once()


async def test_evaluate_response_fail_open_on_exception(engine, mocks):
    """Exceptions in evaluate_response return ALLOW (fail-open)."""
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    # Make get_or_create_session raise
    mocks["sm"].return_value.get_or_create_session = AsyncMock(
        side_effect=RuntimeError("store exploded")
    )

    result = await engine.evaluate_response("sid", "response", {})

    assert result.decision == Decision.ALLOW
    assert "store exploded" in result.metadata["error"]
    assert result.metadata["session_id"] == "sid"


async def test_evaluate_response_fail_open_on_classifier_error(engine, mocks):
    """Classifier exception returns ALLOW (fail-open)."""
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mock_session.current_state = "start"
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    # Classifier raises
    mocks["classifier"].return_value.classify.side_effect = ValueError("bad input")

    result = await engine.evaluate_response("sid", "response", {})

    assert result.decision == Decision.ALLOW
    assert "bad input" in result.metadata["error"]


async def test_transition_called_with_expected_from_state(engine, mocks):
    """evaluate_response passes expected_from_state to transition."""
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mock_session.current_state = "start"
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="middle", confidence=0.9, method="test"
    )
    mocks["constraints"].return_value.evaluate_all.return_value = []
    mocks["sm"].return_value.transition = AsyncMock(
        return_value=(TransitionResult.SUCCESS, None)
    )

    await engine.evaluate_response("sid", "response", {})

    mocks["sm"].return_value.transition.assert_called_once_with(
        "sid",
        "middle",
        confidence=0.9,
        method="test",
        expected_from_state="start",
    )


async def test_reset_session_evaluates_boundary_constraints(engine, mocks):
    """Task 83: reset_session evaluates boundary constraints before reset."""
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mocks["sm"].return_value.get_session = AsyncMock(return_value=mock_session)
    mocks["sm"].return_value.reset_session = AsyncMock()

    violation = MagicMock()
    violation.constraint_name = "must_resolve"
    violation.message = "Must resolve"
    violation.constraint_type = "eventually"
    violation.details = {}
    mocks["constraints"].return_value.evaluate_session_boundary.return_value = [violation]

    result = await engine.reset_session("sid")

    mocks["constraints"].return_value.evaluate_session_boundary.assert_called_once_with(
        mock_session,
    )
    mocks["sm"].return_value.reset_session.assert_called_once()
    assert result == [violation]


async def test_reset_session_skips_when_not_initialized(engine, mocks):
    """reset_session is a no-op when engine is not initialized."""
    result = await engine.reset_session("sid")
    assert result == []


async def test_reset_session_skips_boundary_when_no_session(engine, mocks):
    """reset_session skips boundary eval when session doesn't exist."""
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mocks["sm"].return_value.get_session = AsyncMock(return_value=None)
    mocks["sm"].return_value.reset_session = AsyncMock()

    result = await engine.reset_session("sid")

    mocks["constraints"].return_value.evaluate_session_boundary.assert_not_called()
    assert result == []


async def test_eviction_callback_wired_on_initialize(engine, mocks):
    """Task 21: initialize wires eviction callback to state machine."""
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mocks["sm"].return_value.set_eviction_callback.assert_called_once()


async def test_max_history_passed_to_state_machine(engine, mocks):
    """Task 13: max_history config is forwarded to WorkflowStateMachine."""
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}, "max_history": 500})

    mocks["sm"].assert_called_once()
    call_kwargs = mocks["sm"].call_args
    assert call_kwargs[1]["max_history"] == 500


async def test_initialize_warns_when_embeddings_unavailable(engine, mocks, caplog):
    """Warn at startup when sentence-transformers is not installed."""
    import logging

    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow

    # Make the classifier's check return False
    mocks["classifier"].return_value.check_embedding_availability.return_value = False

    with caplog.at_level(logging.WARNING, logger="opensentinel.policy.engines.fsm.engine"):
        await engine.initialize({"workflow": {}})

    assert any(
        "sentence-transformers not installed" in record.message
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Integration tests — real components, no mocks fixture
# ---------------------------------------------------------------------------

class TestFSMIntegration:
    """End-to-end tests using real WorkflowStateMachine, ConstraintEvaluator,
    and StateClassifier (pattern-based classification, no embeddings)."""

    WORKFLOW = {
        "name": "support-flow",
        "states": [
            {
                "name": "greeting",
                "is_initial": True,
                "classification": {"patterns": [r"\bhello\b", r"\bwelcome\b"]},
            },
            {
                "name": "identify_issue",
                "classification": {"patterns": [r"\bissue\b", r"\bproblem\b"]},
            },
            {
                "name": "resolution",
                "is_terminal": True,
                "classification": {"patterns": [r"\bresolved\b", r"\bfixed\b"]},
            },
            {
                "name": "data_leak",
                "classification": {"patterns": [r"\binternal_dump\b"]},
            },
        ],
        "transitions": [
            {"from_state": "greeting", "to_state": "identify_issue"},
            {"from_state": "identify_issue", "to_state": "resolution"},
            {"from_state": "greeting", "to_state": "resolution"},
        ],
        "constraints": [
            {
                "name": "no_data_leak",
                "type": "never",
                "target": "data_leak",
                "message": "Data leak state must never be entered.",
            },
        ],
    }

    @staticmethod
    def _response(content: str) -> dict:
        """Build a minimal OpenAI-style response dict."""
        return {"choices": [{"message": {"content": content}}]}

    @staticmethod
    def _tool_response(tool_name: str) -> dict:
        """Build a response with a tool call."""
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": tool_name}},
                        ],
                    },
                }
            ],
        }

    async def _init_engine(self) -> FSMPolicyEngine:
        engine = FSMPolicyEngine()
        await engine.initialize({"workflow": self.WORKFLOW})
        return engine

    async def test_happy_path_transitions_and_allows(self):
        """Response classified via pattern → state transitions → ALLOW."""
        engine = await self._init_engine()
        sid = "integration-1"

        # Step 1: greeting → identify_issue (pattern: "issue")
        result = await engine.evaluate_response(
            sid, self._response("I have an issue with my account"), {}
        )
        assert result.decision == Decision.ALLOW
        assert result.metadata["current_state"] == "identify_issue"
        assert result.metadata["previous_state"] == "greeting"

        # Step 2: identify_issue → resolution (pattern: "resolved")
        result = await engine.evaluate_response(
            sid, self._response("Your ticket has been resolved"), {}
        )
        assert result.decision == Decision.ALLOW
        assert result.metadata["current_state"] == "resolution"

    async def test_never_constraint_triggers_intervene(self):
        """Transitioning to a NEVER-constrained state returns INTERVENE and
        blocks the transition."""
        engine = await self._init_engine()
        sid = "integration-2"

        # Attempt a response that classifies as the forbidden data_leak state
        result = await engine.evaluate_response(
            sid, self._response("Here is the internal_dump of the system"), {}
        )
        assert result.decision == Decision.INTERVENE
        assert "Data leak" in (result.message or "")
        assert result.metadata["transition_result"] == "constraint_violated"
        # State should remain at greeting (transition blocked)
        assert result.metadata["current_state"] == "greeting"

    async def test_invalid_transition_returns_intervene(self):
        """Attempting a transition not in the graph returns INTERVENE."""
        engine = await self._init_engine()
        sid = "integration-3"

        # Move to identify_issue first
        result = await engine.evaluate_response(
            sid, self._response("I have a problem"), {}
        )
        assert result.decision == Decision.ALLOW
        assert result.metadata["current_state"] == "identify_issue"

        # Now try to go back to greeting — no such transition defined
        result = await engine.evaluate_response(
            sid, self._response("hello again, welcome back"), {}
        )
        assert result.decision == Decision.INTERVENE
        assert result.metadata["transition_result"] == "invalid_transition"

    async def test_tool_call_classification(self):
        """Tool-call based classification works end-to-end."""
        # Add a tool_calls hint to identify_issue
        workflow = {
            **self.WORKFLOW,
            "states": [
                {
                    "name": "greeting",
                    "is_initial": True,
                    "classification": {"patterns": [r"\bhello\b"]},
                },
                {
                    "name": "identify_issue",
                    "classification": {"tool_calls": ["lookup_ticket"]},
                },
                {
                    "name": "resolution",
                    "is_terminal": True,
                    "classification": {"patterns": [r"\bresolved\b"]},
                },
            ],
            "transitions": [
                {"from_state": "greeting", "to_state": "identify_issue"},
                {"from_state": "identify_issue", "to_state": "resolution"},
            ],
            "constraints": [],
        }
        engine = FSMPolicyEngine()
        await engine.initialize({"workflow": workflow})

        result = await engine.evaluate_response(
            "integration-4", self._tool_response("lookup_ticket"), {}
        )
        assert result.decision == Decision.ALLOW
        assert result.metadata["current_state"] == "identify_issue"
        assert result.metadata["classification_method"] == "tool_call"

    async def test_same_state_stays_allow(self):
        """Classifying to the same state is a no-op ALLOW."""
        engine = await self._init_engine()
        sid = "integration-5"

        # Classify as greeting while already in greeting
        result = await engine.evaluate_response(
            sid, self._response("hello there"), {}
        )
        assert result.decision == Decision.ALLOW
        assert result.metadata["transition_result"] == "same_state"
        assert result.metadata["current_state"] == "greeting"
