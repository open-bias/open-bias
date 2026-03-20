
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from opensentinel.policy.engines.fsm.engine import FSMPolicyEngine
from opensentinel.policy.engines.fsm.workflow.state_machine import TransitionResult
from opensentinel.policy.engines.stateful import StateClassificationResult
from opensentinel.policy.engines.fsm.workflow.schema import ConstraintType
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
    mock_wf = MagicMock(name="test_workflow", mode="guide", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_wf

    await engine.initialize(config)

    assert engine._initialized
    assert engine._mode == "guide"
    mocks["sm"].assert_called_once()
    mocks["classifier"].assert_called_once()
    mocks["constraints"].assert_called_once()
    assert engine.engine_type == "fsm"


async def test_evaluate_response_success(engine, mocks):
    mock_workflow = MagicMock(name="test_workflow", mode="guide", states=[], constraints=[])
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


async def test_evaluate_response_guide_mode_intervenes(engine, mocks):
    """In guide mode, all violations produce INTERVENE (never BLOCK)."""
    mock_workflow = MagicMock(name="test_workflow", mode="guide", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="bad_state", confidence=0.9, method="test"
    )

    violation = MagicMock()
    violation.constraint_name = "test_constraint"
    violation.message = "Don't do that"
    violation.constraint_type = ConstraintType.NEVER
    violation.details = {}

    mocks["constraints"].return_value.evaluate_all.return_value = [violation]
    mocks["sm"].return_value.transition = AsyncMock(
        return_value=(TransitionResult.SUCCESS, None)
    )

    result = await engine.evaluate_response("sid", "response", {})

    assert result.decision == Decision.INTERVENE
    assert result.metadata["mode"] == "guide"


async def test_evaluate_response_guide_mode_never_blocks(engine, mocks):
    """Guide mode produces INTERVENE even for PRECEDENCE violations."""
    mock_workflow = MagicMock(name="test_workflow", mode="guide", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="action", confidence=0.9, method="test"
    )

    violation = MagicMock()
    violation.constraint_name = "verify_first"
    violation.message = "Must verify identity first"
    violation.constraint_type = ConstraintType.PRECEDENCE
    violation.details = {}

    mocks["constraints"].return_value.evaluate_all.return_value = [violation]
    mocks["sm"].return_value.transition = AsyncMock(
        return_value=(TransitionResult.SUCCESS, None)
    )

    result = await engine.evaluate_response("sid", "response", {})

    assert result.decision == Decision.INTERVENE
    assert result.metadata["mode"] == "guide"


async def test_evaluate_response_enforce_mode_blocks_never(engine, mocks):
    """In enforce mode, NEVER violations produce BLOCK."""
    mock_workflow = MagicMock(name="test_workflow", mode="enforce", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="bad_state", confidence=0.9, method="test"
    )

    violation = MagicMock()
    violation.constraint_name = "never_share"
    violation.message = "Policy: never share internal system information"
    violation.constraint_type = ConstraintType.NEVER
    violation.details = {}

    mocks["constraints"].return_value.evaluate_all.return_value = [violation]
    mocks["sm"].return_value.transition = AsyncMock(
        return_value=(TransitionResult.SUCCESS, None)
    )

    result = await engine.evaluate_response("sid", "response", {})

    assert result.decision == Decision.BLOCK
    assert result.metadata["mode"] == "enforce"


async def test_evaluate_response_enforce_mode_blocks_precedence(engine, mocks):
    """In enforce mode, PRECEDENCE violations produce BLOCK."""
    mock_workflow = MagicMock(name="test_workflow", mode="enforce", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="action", confidence=0.9, method="test"
    )

    violation = MagicMock()
    violation.constraint_name = "verify_first"
    violation.message = "Must verify first"
    violation.constraint_type = ConstraintType.PRECEDENCE
    violation.details = {}

    mocks["constraints"].return_value.evaluate_all.return_value = [violation]
    mocks["sm"].return_value.transition = AsyncMock(
        return_value=(TransitionResult.SUCCESS, None)
    )

    result = await engine.evaluate_response("sid", "response", {})

    assert result.decision == Decision.BLOCK


async def test_evaluate_response_enforce_mode_intervenes_eventually(engine, mocks):
    """In enforce mode, EVENTUALLY violations produce INTERVENE (not BLOCK)."""
    mock_workflow = MagicMock(name="test_workflow", mode="enforce", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})

    mock_session = MagicMock()
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)

    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="some_state", confidence=0.9, method="test"
    )

    violation = MagicMock()
    violation.constraint_name = "must_resolve"
    violation.message = "Policy: must eventually resolve the conversation"
    violation.constraint_type = ConstraintType.EVENTUALLY
    violation.details = {}

    mocks["constraints"].return_value.evaluate_all.return_value = [violation]
    mocks["sm"].return_value.transition = AsyncMock(
        return_value=(TransitionResult.SUCCESS, None)
    )

    result = await engine.evaluate_response("sid", "response", {})

    assert result.decision == Decision.INTERVENE


async def test_initialization_with_config_path(engine, mocks):
    """Test initialization using the unified config_path parameter."""
    config = {"config_path": "path/to/workflow.yaml"}
    mock_wf = MagicMock(name="test_workflow", mode="enforce", states=[], constraints=[])
    mocks["parser"].parse_file.return_value = mock_wf

    await engine.initialize(config)

    assert engine._initialized
    assert engine._mode == "enforce"
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
    mock_workflow = MagicMock(name="test_workflow", mode="enforce", states=[], constraints=[])
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
