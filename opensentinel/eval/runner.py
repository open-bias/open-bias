"""EvalRunner: replays conversations through policy engines."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opensentinel.policy.protocols import PolicyEngine, EngineResult

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Result from one turn of evaluation."""

    turn_index: int
    request_data: dict[str, Any]
    response_data: dict[str, Any]
    request_eval: EngineResult
    response_eval: EngineResult


@dataclass
class EvalResult:
    """Result from replaying a full conversation."""

    scenario_path: str
    session_id: str
    turns: list[TurnResult]
    engine_type: str
    error: str | None = None


def _split_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split a flat message array into turns.

    Each turn contains:
      - "messages_so_far": all messages up to (not including) the assistant message
      - "assistant_message": the assistant response for this turn
      - "tool_messages": any tool messages immediately following the assistant message

    Walk the array: accumulate messages. When we hit an assistant message,
    that's the response for this turn. Tool messages following it are part
    of the same turn.
    """
    turns: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if msg["role"] == "assistant":
            # Collect any tool messages that follow
            tool_messages: list[dict[str, Any]] = []
            j = i + 1
            while j < len(messages) and messages[j]["role"] == "tool":
                tool_messages.append(messages[j])
                j += 1

            turns.append(
                {
                    "messages_so_far": list(buffer),
                    "assistant_message": msg,
                    "tool_messages": tool_messages,
                }
            )

            # Add assistant + tool messages to the running buffer for next turn
            buffer.append(msg)
            buffer.extend(tool_messages)
            i = j
        else:
            buffer.append(msg)
            i += 1

    return turns


class EvalRunner:
    """Replays conversations through a policy engine turn-by-turn."""

    async def run(
        self,
        engine: PolicyEngine,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> EvalResult:
        """Replay a conversation through an engine.

        Splits messages into turns, calls evaluate_request then
        evaluate_response for each turn, collects results.
        """
        sid = session_id or f"eval-{uuid.uuid4().hex[:12]}"
        turn_results: list[TurnResult] = []

        try:
            turns = _split_turns(messages)

            for idx, turn in enumerate(turns):
                # Build request_data from messages so far
                request_data: dict[str, Any] = {
                    "messages": turn["messages_so_far"],
                    "model": "eval-mock",
                }

                # Build response_data wrapping the assistant message
                assistant_msg = turn["assistant_message"]
                response_data: dict[str, Any] = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": assistant_msg.get("content", ""),
                                "tool_calls": assistant_msg.get("tool_calls", []),
                            },
                        }
                    ],
                    "model": "eval-mock",
                }

                request_eval = await engine.evaluate_request(
                    session_id=sid,
                    request_data=request_data,
                )
                response_eval = await engine.evaluate_response(
                    session_id=sid,
                    response_data=response_data,
                    request_data=request_data,
                )

                turn_results.append(
                    TurnResult(
                        turn_index=idx,
                        request_data=request_data,
                        response_data=response_data,
                        request_eval=request_eval,
                        response_eval=response_eval,
                    )
                )

        except Exception as e:
            logger.exception(f"Error during eval replay: {e}")
            return EvalResult(
                scenario_path="",
                session_id=sid,
                turns=turn_results,
                engine_type=engine.engine_type,
                error=str(e),
            )

        return EvalResult(
            scenario_path="",
            session_id=sid,
            turns=turn_results,
            engine_type=engine.engine_type,
        )

    async def run_suite(
        self,
        engine: PolicyEngine,
        scenario_paths: list[Path],
    ) -> list[EvalResult]:
        """Run multiple conversations through the same engine."""
        results: list[EvalResult] = []

        for path in scenario_paths:
            messages = json.loads(path.read_text())
            result = await self.run(engine, messages)
            result.scenario_path = str(path)
            results.append(result)

        return results
