
import pytest
from unittest.mock import AsyncMock, MagicMock
import sys

# Mock nemoguardrails before importing engine
mock_nemo = MagicMock()
sys.modules["nemoguardrails"] = mock_nemo

from opensentinel.policy.engines.nemo.engine import NemoGuardrailsPolicyEngine
from opensentinel.policy.protocols import Decision


def _make_result(activated_rails: list | None = None) -> MagicMock:
    """Build a mock NeMo result object with .log.activated_rails."""
    result = MagicMock(spec=["log"])
    result.log = MagicMock(spec=["activated_rails"])
    result.log.activated_rails = activated_rails if activated_rails is not None else []
    return result


def _make_plain_string_result(text: str) -> str:
    """Simulate older NeMo that returns a plain string."""
    return text


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


async def test_initialization(engine, mock_config, mock_rails):
    config = {"config_path": "dummy/path"}

    await engine.initialize(config)

    assert engine._initialized
    mock_config.from_path.assert_called_with("dummy/path")
    mock_nemo.LLMRails.assert_called_once()
    assert engine.name == "nemo:guardrails"
    assert engine.engine_type == "nemo"


async def test_evaluate_request_allow(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    # No activated rails — request is allowed
    mock_rails.generate_async.return_value = _make_result(activated_rails=[])

    result = await engine.evaluate_request(
        session_id="test-session",
        request_data={"messages": [{"role": "user", "content": "hello"}]}
    )

    assert result.decision == Decision.ALLOW


async def test_evaluate_request_blocked(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    # Rail was activated
    mock_rails.generate_async.return_value = _make_result(
        activated_rails=[{"type": "input", "name": "block jailbreak"}]
    )

    result = await engine.evaluate_request(
        session_id="test-session",
        request_data={"messages": [{"role": "user", "content": "bad request"}]}
    )

    assert result.decision == Decision.INTERVENE
    assert "intercepted" in result.message.lower()


async def test_evaluate_request_blocked_violations_metadata(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.return_value = _make_result(
        activated_rails=[{"type": "input", "name": "block jailbreak"}]
    )

    result = await engine.evaluate_request(
        session_id="test-session",
        request_data={"messages": [{"role": "user", "content": "bad request"}]}
    )

    assert result.decision == Decision.INTERVENE
    violations = result.metadata.get("violations", [])
    assert len(violations) == 1
    assert violations[0]["name"] == "nemo_input_blocked"
    assert violations[0]["message"] == "block jailbreak"


async def test_evaluate_response_allow(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    # No activated rails
    mock_rails.generate_async.return_value = _make_result(activated_rails=[])

    result = await engine.evaluate_response(
        session_id="test-session",
        response_data="Safe response",
        request_data={"messages": []}
    )

    assert result.decision == Decision.ALLOW


async def test_evaluate_response_blocked(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.return_value = _make_result(
        activated_rails=[{"type": "output", "name": "block unsafe content"}]
    )

    result = await engine.evaluate_response(
        session_id="test-session",
        response_data="Unsafe response",
        request_data={"messages": []}
    )

    assert result.decision == Decision.INTERVENE
    assert result.message is not None


async def test_evaluate_response_blocked_violations_metadata(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.return_value = _make_result(
        activated_rails=[{"type": "output", "name": "block unsafe content"}]
    )

    result = await engine.evaluate_response(
        session_id="test-session",
        response_data="Unsafe response",
        request_data={"messages": []}
    )

    assert result.decision == Decision.INTERVENE
    violations = result.metadata.get("violations", [])
    assert len(violations) == 1
    assert violations[0]["name"] == "nemo_output_blocked"
    assert violations[0]["message"] == "block unsafe content"


async def test_evaluate_request_plain_string_result_no_false_positive(engine, mock_rails):
    """Older NeMo versions return a plain string — should not false-positive."""
    await engine.initialize({"config_path": "dummy"})

    # Plain string with text that used to trigger false positives
    mock_rails.generate_async.return_value = "The character says 'I cannot believe it'"

    result = await engine.evaluate_request(
        session_id="test-session",
        request_data={"messages": [{"role": "user", "content": "tell me a story"}]}
    )

    assert result.decision == Decision.ALLOW


async def test_evaluate_response_plain_string_result_no_false_positive(engine, mock_rails):
    """Plain string result should not trigger a block."""
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.return_value = "I cannot believe how great this is!"

    result = await engine.evaluate_response(
        session_id="test-session",
        response_data="I cannot believe how great this is!",
        request_data={"messages": []}
    )

    assert result.decision == Decision.ALLOW


async def test_check_rail_activations_no_log_attr():
    """_check_rail_activations handles result with no .log attribute."""
    engine = NemoGuardrailsPolicyEngine()
    result = _make_plain_string_result("some text")
    activations = engine._check_rail_activations(result)
    assert activations == []


async def test_check_rail_activations_empty_list():
    engine = NemoGuardrailsPolicyEngine()
    result = _make_result(activated_rails=[])
    activations = engine._check_rail_activations(result)
    assert activations == []


async def test_check_rail_activations_with_entries():
    engine = NemoGuardrailsPolicyEngine()
    rails = [{"type": "input", "name": "rail1"}, {"type": "output", "name": "rail2"}]
    result = _make_result(activated_rails=rails)
    activations = engine._check_rail_activations(result)
    assert len(activations) == 2
    assert activations[0]["name"] == "rail1"


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


async def test_violations_use_fallback_type_when_missing(engine, mock_rails):
    """If activation dict has no 'type' key, name falls back to 'input'/'output'."""
    await engine.initialize({"config_path": "dummy"})

    # Activation with no 'type' key
    mock_rails.generate_async.return_value = _make_result(
        activated_rails=[{"name": "some-rail"}]
    )

    result = await engine.evaluate_request(
        session_id="test-session",
        request_data={"messages": [{"role": "user", "content": "test"}]}
    )

    assert result.decision == Decision.INTERVENE
    violations = result.metadata.get("violations", [])
    assert violations[0]["name"] == "nemo_input_blocked"


async def test_multiple_rail_activations_produce_multiple_violations(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.return_value = _make_result(
        activated_rails=[
            {"type": "input", "name": "jailbreak-rail"},
            {"type": "input", "name": "pii-rail"},
        ]
    )

    result = await engine.evaluate_request(
        session_id="test-session",
        request_data={"messages": [{"role": "user", "content": "test"}]}
    )

    assert result.decision == Decision.INTERVENE
    violations = result.metadata.get("violations", [])
    assert len(violations) == 2


async def test_get_session_state_returns_none(engine):
    """NeMo is stateless — get_session_state should always return None."""
    await engine.initialize({"config_path": "dummy"})

    result = await engine.get_session_state("any-session-id")

    assert result is None
