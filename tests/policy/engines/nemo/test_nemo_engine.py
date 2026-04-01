
import pytest
from unittest.mock import AsyncMock, MagicMock
import sys

# Mock nemoguardrails before importing engine
mock_nemo = MagicMock()
sys.modules["nemoguardrails"] = mock_nemo

from openbias.policy.engines.nemo.engine import NemoGuardrailsPolicyEngine
from openbias.policy.protocols import EvaluationStatus


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

    assert result.status == EvaluationStatus.ALLOW


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

    assert result.status == EvaluationStatus.VIOLATION
    assert len(result.violations) > 0


async def test_evaluate_request_blocked_violations_metadata(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.return_value = _make_result(
        activated_rails=[{"type": "input", "name": "block jailbreak"}]
    )

    result = await engine.evaluate_request(
        session_id="test-session",
        request_data={"messages": [{"role": "user", "content": "bad request"}]}
    )

    assert result.status == EvaluationStatus.VIOLATION
    assert len(result.violations) == 1
    assert result.violations[0].rule_name == "nemo_input_blocked"
    assert result.violations[0].reason == "block jailbreak"


async def test_evaluate_response_allow(engine, mock_rails):
    await engine.initialize({"config_path": "dummy"})

    # No activated rails
    mock_rails.generate_async.return_value = _make_result(activated_rails=[])

    result = await engine.evaluate_response(
        session_id="test-session",
        response_data="Safe response",
        request_data={"messages": []}
    )

    assert result.status == EvaluationStatus.ALLOW


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

    assert result.status == EvaluationStatus.VIOLATION
    assert len(result.violations) > 0


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

    assert result.status == EvaluationStatus.VIOLATION
    assert len(result.violations) == 1
    assert result.violations[0].rule_name == "nemo_output_blocked"
    assert result.violations[0].reason == "block unsafe content"


async def test_evaluate_request_plain_string_result_no_false_positive(engine, mock_rails):
    """Older NeMo versions return a plain string — should not false-positive."""
    await engine.initialize({"config_path": "dummy"})

    # Plain string with text that used to trigger false positives
    mock_rails.generate_async.return_value = "The character says 'I cannot believe it'"

    result = await engine.evaluate_request(
        session_id="test-session",
        request_data={"messages": [{"role": "user", "content": "tell me a story"}]}
    )

    assert result.status == EvaluationStatus.ALLOW


async def test_evaluate_response_plain_string_result_no_false_positive(engine, mock_rails):
    """Plain string result should not trigger a block."""
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.return_value = "I cannot believe how great this is!"

    result = await engine.evaluate_response(
        session_id="test-session",
        response_data="I cannot believe how great this is!",
        request_data={"messages": []}
    )

    assert result.status == EvaluationStatus.ALLOW


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
    assert result.status == EvaluationStatus.ALLOW
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

    # Engine is a pure evaluator: reports VIOLATION, not BLOCK.
    # The interceptor maps violations to block/intervene via fail_action.
    assert result.status == EvaluationStatus.VIOLATION
    assert len(result.violations) > 0
    # Provider decision metadata is preserved for interceptor to inspect
    assert any(v.extra.get("provider_decision") == "block" for v in result.violations)


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

    assert result.status == EvaluationStatus.VIOLATION
    assert result.violations[0].rule_name == "nemo_input_blocked"


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

    assert result.status == EvaluationStatus.VIOLATION
    assert len(result.violations) == 2


async def test_get_session_state_returns_none(engine):
    """NeMo is stateless — get_session_state should always return None."""
    await engine.initialize({"config_path": "dummy"})

    result = await engine.get_session_state("any-session-id")

    assert result is None


# ---------------------------------------------------------------------------
# Parameterized edge-case coverage for _check_rail_activations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "activated_rails, expected_len",
    [
        pytest.param([{"type": "input", "name": ""}], 1, id="empty-string-name"),
        pytest.param([{"type": "input", "name": None}], 1, id="none-name"),
        pytest.param([{"type": "input"}], 1, id="missing-name-key"),
        pytest.param([{"name": "block something"}], 1, id="missing-type-key"),
        pytest.param([{"type": "dialog"}], 1, id="only-type-no-name"),
        pytest.param([{}], 1, id="empty-dict"),
        pytest.param(None, 0, id="none-activated-rails"),
        pytest.param([], 0, id="empty-list"),
    ],
)
async def test_check_rail_activations_edge_cases(activated_rails, expected_len):
    """_check_rail_activations handles malformed activation dicts gracefully."""
    engine = NemoGuardrailsPolicyEngine()
    result = _make_result(activated_rails=activated_rails)
    activations = engine._check_rail_activations(result)
    assert len(activations) == expected_len


@pytest.mark.parametrize(
    "activated_rails",
    [
        pytest.param(
            [{"type": "input", "name": "block jailbreak", "score": 0.95}],
            id="extra-score-key",
        ),
        pytest.param(
            [{"type": "output", "name": "pii filter", "model": "bert-base", "latency_ms": 12}],
            id="extra-model-and-latency-keys",
        ),
        pytest.param(
            [{"type": "execution", "name": "sandbox escape", "custom": {"nested": True}}],
            id="extra-nested-dict-key",
        ),
    ],
)
async def test_check_rail_activations_extra_keys(activated_rails):
    """Activation dicts with unexpected extra keys are passed through."""
    engine = NemoGuardrailsPolicyEngine()
    result = _make_result(activated_rails=activated_rails)
    activations = engine._check_rail_activations(result)
    assert len(activations) == 1
    assert activations[0] is activated_rails[0]


async def test_check_rail_activations_log_is_none():
    """Result whose .log attribute is None should return empty list."""
    engine = NemoGuardrailsPolicyEngine()
    result = MagicMock(spec=["log"])
    result.log = None
    assert engine._check_rail_activations(result) == []


# ---------------------------------------------------------------------------
# Parameterized evaluate_request / evaluate_response with varied block patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rail_type, rail_name, expected_violation_name",
    [
        pytest.param("input", "block jailbreak", "nemo_input_blocked", id="input-jailbreak"),
        pytest.param("dialog", "topic guard", "nemo_dialog_blocked", id="dialog-topic-guard"),
        pytest.param("execution", "sandbox escape", "nemo_execution_blocked", id="execution-rail"),
        pytest.param("custom_type", "org policy", "nemo_custom_type_blocked", id="custom-type"),
    ],
)
async def test_evaluate_request_varied_rail_types(
    engine, mock_rails, rail_type, rail_name, expected_violation_name
):
    """evaluate_request produces correct violation name for varied rail types."""
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.return_value = _make_result(
        activated_rails=[{"type": rail_type, "name": rail_name}]
    )

    result = await engine.evaluate_request(
        session_id="s1",
        request_data={"messages": [{"role": "user", "content": "test"}]},
    )

    assert result.status == EvaluationStatus.VIOLATION
    assert result.violations[0].rule_name == expected_violation_name
    assert result.violations[0].reason == rail_name


@pytest.mark.parametrize(
    "rail_type, rail_name, expected_violation_name",
    [
        pytest.param("output", "block unsafe content", "nemo_output_blocked", id="output-unsafe"),
        pytest.param("dialog", "off-topic", "nemo_dialog_blocked", id="dialog-off-topic"),
        pytest.param("retrieval", "hallucination", "nemo_retrieval_blocked", id="retrieval-rail"),
    ],
)
async def test_evaluate_response_varied_rail_types(
    engine, mock_rails, rail_type, rail_name, expected_violation_name
):
    """evaluate_response produces correct violation name for varied rail types."""
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.return_value = _make_result(
        activated_rails=[{"type": rail_type, "name": rail_name}]
    )

    result = await engine.evaluate_response(
        session_id="s1",
        response_data="some response",
        request_data={"messages": [{"role": "user", "content": "q"}]},
    )

    assert result.status == EvaluationStatus.VIOLATION
    assert result.violations[0].rule_name == expected_violation_name
    assert result.violations[0].reason == rail_name


@pytest.mark.parametrize(
    "activation, expected_msg",
    [
        pytest.param(
            {"type": "input"},
            "NeMo input rail triggered",
            id="missing-name-uses-fallback-message",
        ),
        pytest.param(
            {"type": "input", "name": ""},
            "",
            id="empty-name-kept-as-is",
        ),
        pytest.param(
            {"name": "some rail"},
            "some rail",
            id="missing-type-falls-back-to-input",
        ),
        pytest.param(
            {},
            "NeMo input rail triggered",
            id="empty-dict-uses-all-fallbacks",
        ),
    ],
)
async def test_evaluate_request_violation_fallback_values(
    engine, mock_rails, activation, expected_msg
):
    """Verify fallback values when activation dict is missing keys."""
    await engine.initialize({"config_path": "dummy"})

    mock_rails.generate_async.return_value = _make_result(activated_rails=[activation])

    result = await engine.evaluate_request(
        session_id="s1",
        request_data={"messages": [{"role": "user", "content": "x"}]},
    )

    assert result.status == EvaluationStatus.VIOLATION
    assert result.violations[0].reason == expected_msg


# ---------------------------------------------------------------------------
# Parameterized false-positive resilience
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("The character says 'I cannot believe it'", id="fiction-dialogue"),
        pytest.param("I'm sorry, but that movie was terrible!", id="apology-phrase"),
        pytest.param("block of cheese on the counter", id="word-block-in-context"),
        pytest.param("unsafe at any speed is a great book", id="word-unsafe-in-context"),
        pytest.param("", id="empty-string"),
        pytest.param("jailbreak your phone to install custom firmware", id="jailbreak-benign"),
        pytest.param("The policy was updated yesterday.", id="word-policy-in-context"),
    ],
)
async def test_plain_string_result_no_false_positive_varied(engine, mock_rails, text):
    """Plain string results (older NeMo) with varied text should never trigger blocks."""
    await engine.initialize({"config_path": "dummy"})
    mock_rails.generate_async.return_value = text

    result = await engine.evaluate_request(
        session_id="s1",
        request_data={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert result.status == EvaluationStatus.ALLOW
