
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from opensentinel.policy.engines.composite.engine import CompositePolicyEngine
from opensentinel.policy.protocols import Decision, EngineResult


@pytest.fixture
def engine():
    return CompositePolicyEngine()

@pytest.fixture
def mock_registry():
    with patch("opensentinel.policy.engines.composite.engine.PolicyEngineRegistry") as mock:
        yield mock

@pytest.mark.asyncio
async def test_initialization(engine, mock_registry):
    mock_engine1 = MagicMock()
    mock_engine1.configure_mock(name="mock1", engine_type="type1")
    mock_engine1.initialize = AsyncMock()

    mock_engine2 = MagicMock()
    mock_engine2.configure_mock(name="mock2", engine_type="type2")
    mock_engine2.initialize = AsyncMock()

    mock_registry.create.side_effect = [mock_engine1, mock_engine2]

    config = {
        "engines": [
            {"type": "type1", "config": {"a": 1}},
            {"type": "type2", "config": {"b": 2}},
        ]
    }

    await engine.initialize(config)

    assert engine._initialized
    assert len(engine._engines) == 2
    mock_registry.create.assert_any_call("type1")
    mock_registry.create.assert_any_call("type2")
    mock_engine1.initialize.assert_awaited_with({"a": 1})

@pytest.mark.asyncio
async def test_evaluate_request_all_allow(engine, mock_registry):
    mock_engine1 = MagicMock()
    mock_engine1.initialize = AsyncMock()
    mock_engine1.evaluate_request = AsyncMock(return_value=EngineResult(decision=Decision.ALLOW))

    mock_engine2 = MagicMock()
    mock_engine2.initialize = AsyncMock()
    mock_engine2.evaluate_request = AsyncMock(return_value=EngineResult(decision=Decision.ALLOW))

    mock_registry.create.side_effect = [mock_engine1, mock_engine2]

    await engine.initialize({
        "engines": [{"type": "t1"}, {"type": "t2"}]
    })

    result = await engine.evaluate_request("sid", {})

    assert result.decision == Decision.ALLOW

@pytest.mark.asyncio
async def test_evaluate_request_one_block(engine, mock_registry):
    mock_engine1 = MagicMock()
    mock_engine1.name = "e1"
    mock_engine1.initialize = AsyncMock()
    mock_engine1.evaluate_request = AsyncMock(return_value=EngineResult(decision=Decision.ALLOW))

    mock_engine2 = MagicMock()
    mock_engine2.name = "e2"
    mock_engine2.initialize = AsyncMock()
    mock_engine2.evaluate_request = AsyncMock(return_value=EngineResult(
        decision=Decision.BLOCK, message="blocked by e2"
    ))

    mock_registry.create.side_effect = [mock_engine1, mock_engine2]

    await engine.initialize({
        "engines": [{"type": "t1"}, {"type": "t2"}]
    })

    result = await engine.evaluate_request("sid", {})

    assert result.decision == Decision.BLOCK
    assert result.message == "blocked by e2"

@pytest.mark.asyncio
async def test_evaluate_request_block_beats_intervene(engine, mock_registry):
    mock_engine1 = MagicMock()
    mock_engine1.name = "e1"
    mock_engine1.initialize = AsyncMock()
    mock_engine1.evaluate_request = AsyncMock(return_value=EngineResult(
        decision=Decision.INTERVENE, message="intervene msg"
    ))

    mock_engine2 = MagicMock()
    mock_engine2.name = "e2"
    mock_engine2.initialize = AsyncMock()
    mock_engine2.evaluate_request = AsyncMock(return_value=EngineResult(
        decision=Decision.BLOCK, message="block msg"
    ))

    mock_registry.create.side_effect = [mock_engine1, mock_engine2]

    await engine.initialize({
        "engines": [{"type": "t1"}, {"type": "t2"}]
    })

    result = await engine.evaluate_request("sid", {})

    assert result.decision == Decision.BLOCK
    assert result.message == "block msg"

@pytest.mark.asyncio
async def test_evaluate_request_intervene_message_first_wins(engine, mock_registry):
    mock_engine1 = MagicMock()
    mock_engine1.name = "e1"
    mock_engine1.initialize = AsyncMock()
    mock_engine1.evaluate_request = AsyncMock(return_value=EngineResult(
        decision=Decision.INTERVENE, message="first intervene"
    ))

    mock_engine2 = MagicMock()
    mock_engine2.name = "e2"
    mock_engine2.initialize = AsyncMock()
    mock_engine2.evaluate_request = AsyncMock(return_value=EngineResult(
        decision=Decision.INTERVENE, message="second intervene"
    ))

    mock_registry.create.side_effect = [mock_engine1, mock_engine2]

    await engine.initialize({
        "engines": [{"type": "t1"}, {"type": "t2"}]
    })

    result = await engine.evaluate_request("sid", {})

    assert result.decision == Decision.INTERVENE
    assert result.message == "first intervene"

@pytest.mark.asyncio
async def test_evaluate_response_parallel_execution(engine, mock_registry):
    mock_engine1 = MagicMock()
    mock_engine1.name = "e1"
    mock_engine1.initialize = AsyncMock()
    mock_engine1.evaluate_response = AsyncMock(return_value=EngineResult(
        decision=Decision.ALLOW
    ))

    mock_engine2 = MagicMock()
    mock_engine2.name = "e2"
    mock_engine2.initialize = AsyncMock()
    mock_engine2.evaluate_response = AsyncMock(return_value=EngineResult(
        decision=Decision.INTERVENE, message="intervene msg"
    ))

    mock_registry.create.side_effect = [mock_engine1, mock_engine2]

    await engine.initialize({
        "engines": [{"type": "t1"}, {"type": "t2"}],
        "parallel": True
    })

    result = await engine.evaluate_response("sid", "data", {})

    assert result.decision == Decision.INTERVENE
    assert result.message == "intervene msg"

@pytest.mark.asyncio
async def test_evaluate_request_engine_error_fail_open(engine, mock_registry):
    mock_engine1 = MagicMock()
    mock_engine1.name = "e1"
    mock_engine1.initialize = AsyncMock()
    mock_engine1.evaluate_request = AsyncMock(side_effect=Exception("Engine failure"))

    mock_registry.create.return_value = mock_engine1

    await engine.initialize({
        "engines": [{"type": "t1"}]
    })

    result = await engine.evaluate_request("sid", {})

    # Engine errors are fail-open
    assert result.decision == Decision.ALLOW
    assert "error" in result.metadata.get("engines", {}).get("e1", {})

@pytest.mark.asyncio
async def test_metadata_merged_under_engine_names(engine, mock_registry):
    mock_engine1 = MagicMock()
    mock_engine1.name = "e1"
    mock_engine1.initialize = AsyncMock()
    mock_engine1.evaluate_request = AsyncMock(return_value=EngineResult(
        decision=Decision.ALLOW, metadata={"key1": "val1"}
    ))

    mock_engine2 = MagicMock()
    mock_engine2.name = "e2"
    mock_engine2.initialize = AsyncMock()
    mock_engine2.evaluate_request = AsyncMock(return_value=EngineResult(
        decision=Decision.ALLOW, metadata={"key2": "val2"}
    ))

    mock_registry.create.side_effect = [mock_engine1, mock_engine2]

    await engine.initialize({
        "engines": [{"type": "t1"}, {"type": "t2"}]
    })

    result = await engine.evaluate_request("sid", {})

    assert result.metadata["engines"]["e1"] == {"key1": "val1"}
    assert result.metadata["engines"]["e2"] == {"key2": "val2"}
