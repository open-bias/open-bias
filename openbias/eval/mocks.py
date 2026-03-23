"""Mock provider for engines that make internal LLM calls."""

from __future__ import annotations

import logging
from typing import Any

from openbias.policy.protocols import PolicyEngine

logger = logging.getLogger(__name__)


class MockResponseSequence:
    """Returns canned responses in sequence, cycling if exhausted."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self._index = 0

    def next(self) -> Any:
        if not self._responses:
            return {}
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response


def apply_mock_provider(
    engine: PolicyEngine,
    engine_type: str,
    responses: list[str] | None = None,
) -> None:
    """Patch internal LLM clients on an engine with canned responses.

    Args:
        engine: An initialized policy engine instance.
        engine_type: The engine type string (e.g., "judge", "nemo").
        responses: List of JSON string responses to return in sequence.
    """
    if responses is None:
        return

    import json

    parsed: list[Any] = []
    for r in responses:
        try:
            parsed.append(json.loads(r))
        except (json.JSONDecodeError, TypeError):
            parsed.append(r)

    seq = MockResponseSequence(parsed)

    if engine_type == "judge":
        _mock_judge(engine, seq)
    elif engine_type == "nemo":
        _mock_nemo(engine, seq)
    else:
        logger.debug(f"No mock provider needed for engine type: {engine_type}")


def _mock_judge(engine: PolicyEngine, seq: MockResponseSequence) -> None:
    """Patch JudgeClient.call_judge with canned responses."""
    client = getattr(engine, "_client", None)
    if client is None:
        logger.warning("Judge engine has no _client attribute; cannot apply mock")
        return

    async def mock_call_judge(
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = seq.next()
        return result

    client.call_judge = mock_call_judge
    logger.debug("Mocked JudgeClient.call_judge")


def _mock_nemo(engine: PolicyEngine, seq: MockResponseSequence) -> None:
    """Patch NeMo rails generate_async with canned responses."""
    rails = getattr(engine, "_rails", None)
    if rails is None:
        logger.warning("Nemo engine has no _rails attribute; cannot apply mock")
        return

    async def mock_generate_async(*args: Any, **kwargs: Any) -> Any:
        return seq.next()

    rails.generate_async = mock_generate_async
    logger.debug("Mocked NeMo generate_async")
