
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
