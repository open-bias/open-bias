"""Shared fixtures for the rebuilt eval harness tests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pytest

from openbias.policy.protocols import (
    EvaluationResult,
    EvaluationStatus,
    PolicyEngine,
    ViolationRecord,
)


@dataclass
class KeywordEngine(PolicyEngine):
    """Deterministic fake engine for eval harness tests."""

    request_keywords: set[str] = field(default_factory=set)
    response_keywords: set[str] = field(default_factory=set)
    false_positive_keywords: set[str] = field(default_factory=set)
    repair_phrase: str = "corrected answer"

    def __post_init__(self) -> None:
        self._initialized = True
        self._sessions: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "keyword-engine"

    @property
    def engine_type(self) -> str:
        return "keyword"

    async def initialize(self, config: dict[str, Any]) -> None:
        del config
        self._initialized = True

    async def evaluate_request(
        self,
        session_id: str,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        del context
        latest_user = _latest_message_content(request_data.get("messages", []), role="user")
        if any(keyword in latest_user for keyword in self.request_keywords):
            return _violation("request keyword violation", scope="request")
        if any(keyword in latest_user for keyword in self.false_positive_keywords):
            return _violation("request false positive", scope="request")
        return _allow()

    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        del context
        state = self._sessions.setdefault(session_id, {})
        assistant_content = _response_content(response_data)
        request_text = "\n".join(
            message.get("content", "")
            for message in request_data.get("messages", [])
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        )
        state["last_request_text"] = request_text
        state["last_response_text"] = assistant_content

        if any(keyword in assistant_content for keyword in self.response_keywords):
            return _violation("response keyword violation", scope="response")
        if any(keyword in assistant_content for keyword in self.false_positive_keywords):
            return _violation("response false positive", scope="response")
        if self.repair_phrase in assistant_content:
            if "I think I made a mistake before; here's what I mean:" in request_text:
                return _allow()
            return _violation("repair answer arrived without intervention context", scope="response")
        return _allow()

    async def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        return self._sessions.get(session_id)

    async def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


@pytest.fixture
def keyword_engine() -> KeywordEngine:
    return KeywordEngine(
        request_keywords={"request-risk"},
        response_keywords={"unsafe answer", "still unsafe"},
        false_positive_keywords={"false-positive trigger"},
    )


def _latest_message_content(messages: Iterable[dict[str, Any]], *, role: str) -> str:
    latest = ""
    for message in messages:
        if message.get("role") == role and isinstance(message.get("content"), str):
            latest = message["content"]
    return latest


def _response_content(response_data: Any) -> str:
    if isinstance(response_data, dict):
        choices = response_data.get("choices", [])
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
    return ""


def _allow() -> EvaluationResult:
    return EvaluationResult(status=EvaluationStatus.ALLOW)


def _violation(reason: str, *, scope: str) -> EvaluationResult:
    return EvaluationResult(
        status=EvaluationStatus.VIOLATION,
        violations=[ViolationRecord(reason=reason, scope=scope, engine="keyword")],
        metadata={"violations": [{"reason": reason, "scope": scope, "engine": "keyword"}]},
    )
