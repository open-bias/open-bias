"""Evaluation tests for the NeMo Guardrails policy engine using conversation scenarios."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock nemoguardrails before importing engine
if "nemoguardrails" not in sys.modules:
    mock_nemo = MagicMock()
    sys.modules["nemoguardrails"] = mock_nemo
else:
    mock_nemo = sys.modules["nemoguardrails"]

from openbias.eval.runner import EvalRunner
from openbias.policy.engines.nemo.engine import NemoGuardrailsPolicyEngine
from openbias.policy.protocols import Decision

EVALS_DIR = Path(__file__).resolve().parent.parent.parent / "evals" / "nemo"


@pytest.fixture
async def engine():
    eng = NemoGuardrailsPolicyEngine()
    mock_nemo.RailsConfig.from_path.return_value = MagicMock()
    mock_rails = MagicMock()
    mock_rails.generate_async = AsyncMock()
    mock_rails.register_action = MagicMock()
    mock_nemo.LLMRails.return_value = mock_rails
    await eng.initialize({"config_path": "dummy/path"})
    yield eng
    await eng.shutdown()


@pytest.fixture
def runner() -> EvalRunner:
    return EvalRunner()


def _make_nemo_result(text: str, activated_rails: list[dict[str, Any]] | None = None) -> Any:
    """Build a mock NeMo result object with .log.activated_rails."""
    log = MagicMock()
    log.activated_rails = activated_rails or []
    result = MagicMock()
    result.log = log
    result.content = text
    return result


def _patch_nemo(
    engine: Any,
    responses: list[str | dict[str, Any]],
) -> None:
    """Patch engine._rails.generate_async with sequential canned NeMo result objects.

    Each entry in *responses* is either:
    - A plain str  → clean result (no rails fired, empty activated_rails)
    - A dict with keys "text" and "activated_rails" → rails-fired result
    """
    call_count = 0

    async def mock_generate_async(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        spec = responses[call_count % len(responses)]
        call_count += 1
        if isinstance(spec, dict):
            return _make_nemo_result(spec["text"], spec.get("activated_rails", []))
        return _make_nemo_result(spec)

    engine._rails.generate_async = mock_generate_async


async def test_clean_support_no_violations(engine, runner):
    """Clean support conversation: all turns should ALLOW with no violations."""
    messages = json.loads((EVALS_DIR / "clean_support.json").read_text())

    # 2 turns × 2 calls (input rail + output rail) = 4 responses, all clean
    _patch_nemo(engine, [
        "I can help you with logging into your account.",
        "Let me help you with your login issue.",
        "Sure, I can help with that.",
        "Try the Forgot Password option on the login page.",
    ])

    result = await runner.run(engine, messages)

    assert result.error is None
    assert len(result.turns) > 0

    for turn in result.turns:
        assert turn.response_eval.decision == Decision.ALLOW
        assert len(turn.response_eval.metadata.get("violations", [])) == 0


async def test_financial_advice_blocked(engine, runner):
    """Financial advice: NeMo should block the response via output rail."""
    messages = json.loads((EVALS_DIR / "financial_advice.json").read_text())

    # 2 turns × 2 calls = 4 responses
    # Input rails pass (clean), output rails fire for financial advice
    _patch_nemo(engine, [
        "Let me check on that for you.",
        {
            "text": "I am a customer support agent and cannot provide financial advice.",
            "activated_rails": [{"type": "output", "name": "financial_advice_rail"}],
        },
        "I understand your question.",
        {
            "text": "I cannot provide investment recommendations.",
            "activated_rails": [{"type": "output", "name": "financial_advice_rail"}],
        },
    ])

    result = await runner.run(engine, messages)

    assert result.error is None
    assert len(result.turns) > 0

    violation_turns = [
        t
        for t in result.turns
        if t.response_eval.decision in (Decision.INTERVENE, Decision.BLOCK)
    ]
    assert len(violation_turns) > 0, "Expected at least one turn with a violation decision"


async def test_security_bypass_blocked(engine, runner):
    """Security bypass: NeMo should block the response via output rail."""
    messages = json.loads((EVALS_DIR / "security_bypass.json").read_text())

    # 2 turns × 2 calls = 4 responses
    # Input rails pass (clean), output rails fire for security bypass attempts
    _patch_nemo(engine, [
        "Let me look into that.",
        {
            "text": "I cannot help with bypassing security measures.",
            "activated_rails": [{"type": "output", "name": "security_bypass_rail"}],
        },
        "I see what you're asking about.",
        {
            "text": "I'm not able to disable security features.",
            "activated_rails": [{"type": "output", "name": "security_bypass_rail"}],
        },
    ])

    result = await runner.run(engine, messages)

    assert result.error is None
    assert len(result.turns) > 0

    violation_turns = [
        t
        for t in result.turns
        if t.response_eval.decision in (Decision.INTERVENE, Decision.BLOCK)
    ]
    assert len(violation_turns) > 0, "Expected at least one turn with a violation decision"
