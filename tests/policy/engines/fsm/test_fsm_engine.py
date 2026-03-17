
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from opensentinel.policy.engines.fsm.engine import FSMPolicyEngine, StateClassificationResult, TransitionResult
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
        
        yield {
            "parser": mock_parser,
            "sm": mock_sm,
            "classifier": mock_classifier,
            "constraints": mock_constraints
        }

@pytest.mark.asyncio
async def test_initialization(engine, mocks):
    config = {"workflow": {"name": "test_workflow", "states": [], "constraints": []}}
    mocks["parser"]().parse_dict.return_value = MagicMock(name="test_workflow", states=[], constraints=[])
    
    await engine.initialize(config)
    
    assert engine._initialized
    mocks["sm"].assert_called_once()
    mocks["classifier"].assert_called_once()
    mocks["constraints"].assert_called_once()
    assert engine.engine_type == "fsm"

@pytest.mark.asyncio
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
    mocks["sm"].return_value.transition = AsyncMock(return_value=(TransitionResult.SUCCESS, None))
    
    result = await engine.evaluate_response("sid", "response", {})
    
    assert result.decision == Decision.ALLOW
    assert len(result.metadata.get("violations", [])) == 0
    mocks["sm"].return_value.transition.assert_called_once()

@pytest.mark.asyncio
async def test_evaluate_response_with_violations(engine, mocks):
    mock_workflow = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"]().parse_dict.return_value = mock_workflow
    await engine.initialize({"workflow": {}})
    
    mock_session = MagicMock()
    mocks["sm"].return_value.get_or_create_session = AsyncMock(return_value=mock_session)
    
    mocks["classifier"].return_value.classify.return_value = StateClassificationResult(
        state_name="bad_state", confidence=0.9, method="test"
    )
    
    # Mock constraint violation
    violation = MagicMock()
    violation.constraint_name = "test_constraint"
    violation.severity = "critical"
    violation.message = "Don't do that"
    violation.intervention = "block"
    violation.constraint_type = MagicMock()
    violation.details = {}
    
    mocks["constraints"].return_value.evaluate_all.return_value = [violation]
    
    mocks["sm"].return_value.transition = AsyncMock(return_value=(TransitionResult.SUCCESS, None))

    result = await engine.evaluate_response("sid", "response", {})

    assert result.decision == Decision.BLOCK
    assert len(result.metadata.get("violations", [])) == 1

@pytest.mark.asyncio
async def test_initialization_with_config_path(engine, mocks):
    """Test initialization using the unified config_path parameter."""
    config = {"config_path": "path/to/workflow.yaml"}
    mocks["parser"]().parse_dict.return_value = MagicMock(name="test_workflow", states=[], constraints=[])
    mocks["parser"].parse_file.return_value = MagicMock(name="test_workflow", states=[], constraints=[])
    
    await engine.initialize(config)
    
    assert engine._initialized
    mocks["parser"].parse_file.assert_called_with("path/to/workflow.yaml")
    mocks["sm"].assert_called_once()
    assert engine.engine_type == "fsm"

@pytest.mark.asyncio
async def test_initialization_failure(engine, mocks):
    """Test initialization failure when no valid config provided."""
    config = {}
    with pytest.raises(ValueError, match="FSM engine requires 'config_path' or 'workflow'"):
        await engine.initialize(config)
