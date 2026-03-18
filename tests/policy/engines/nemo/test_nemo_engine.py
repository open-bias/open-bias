
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys

# Mock nemoguardrails before importing engine
mock_nemo = MagicMock()
sys.modules["nemoguardrails"] = mock_nemo

from opensentinel.policy.engines.nemo.engine import NemoGuardrailsPolicyEngine
from opensentinel.policy.protocols import Decision

@pytest.fixture(autouse=True)
def _reset_nemo_mock():
    """Reset module-level mock state between tests to avoid cross-test pollution."""
    mock_nemo.reset_mock()
    yield

@pytest.fixture
def engine():
    return NemoGuardrailsPolicyEngine()

@pytest.fixture
def mock_rails():
    mock_nemo.LLMRails.return_value = MagicMock()
    rails = mock_nemo.LLMRails.return_value
    rails.generate_async = AsyncMock()
    rails.register_action = MagicMock()
    return rails

@pytest.fixture
def mock_config():
    mock_nemo.RailsConfig.from_path.return_value = MagicMock()
    mock_nemo.RailsConfig.from_content.return_value = MagicMock()
    return mock_nemo.RailsConfig

@pytest.mark.asyncio
async def test_initialization(engine, mock_config, mock_rails):
    config = {"config_path": "dummy/path"}

    await engine.initialize(config)

    assert engine._initialized
    mock_config.from_path.assert_called_with("dummy/path")
    mock_nemo.LLMRails.assert_called_once()
    assert engine.name == "nemo:guardrails"
    assert engine.engine_type == "nemo"

@pytest.mark.asyncio
async def test_evaluate_request_allow(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.return_value = "OK"

    result = await engine.evaluate_request(
        session_id="test-session",
        request_data={"messages": [{"role": "user", "content": "hello"}]}
    )

    assert result.decision == Decision.ALLOW

@pytest.mark.asyncio
async def test_evaluate_request_blocked(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    # Mock blocked response
    mock_rails.generate_async.return_value = "I cannot fulfill this request."

    result = await engine.evaluate_request(
        session_id="test-session",
        request_data={"messages": [{"role": "user", "content": "bad request"}]}
    )

    assert result.decision == Decision.BLOCK
    assert "blocked" in result.message.lower()

@pytest.mark.asyncio
async def test_evaluate_response_allow(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.return_value = "Safe response"

    result = await engine.evaluate_response(
        session_id="test-session",
        response_data="Safe response",
        request_data={"messages": []}
    )

    assert result.decision == Decision.ALLOW

@pytest.mark.asyncio
async def test_evaluate_response_blocked(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.return_value = "I cannot provide this info [blocked]"

    result = await engine.evaluate_response(
        session_id="test-session",
        response_data="Unsafe response",
        request_data={"messages": []}
    )

    assert result.decision == Decision.BLOCK
    assert result.message is not None

@pytest.mark.asyncio
async def test_evaluate_request_error_fail_open(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.side_effect = Exception("NeMo error")

    result = await engine.evaluate_request(
        session_id="test-session",
        request_data={"messages": [{"role": "user", "content": "hi"}]}
    )

    # Default is fail open
    assert result.decision == Decision.ALLOW
    assert "error" in result.metadata

@pytest.mark.asyncio
async def test_evaluate_request_error_fail_closed(engine, mock_rails):
    await engine.initialize({
        "config_path": "dummy",
        "fail_closed": True
    })

    mock_rails.generate_async.side_effect = Exception("NeMo error")

    result = await engine.evaluate_request(
        session_id="test-session",
        request_data={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert result.decision == Decision.BLOCK
    assert result.message is not None
