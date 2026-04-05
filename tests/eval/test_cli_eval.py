"""CLI eval runner tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openbias.cli_eval import run_eval
from openbias.policy.protocols import (
    EvaluationResult,
    EvaluationStatus,
    PolicyEngine,
    ViolationRecord,
)


@dataclass
class KeywordEngine(PolicyEngine):
    request_keywords: set[str] = field(default_factory=set)
    response_keywords: set[str] = field(default_factory=set)
    false_positive_keywords: set[str] = field(default_factory=set)
    repair_phrase: str = "corrected answer"

    def __post_init__(self) -> None:
        self._initialized = True
        self._sessions: dict[str, dict[str, str]] = {}

    @property
    def name(self) -> str:
        return "keyword-engine"

    @property
    def engine_type(self) -> str:
        return "keyword"

    async def initialize(self, config: dict[str, object]) -> None:
        del config

    async def evaluate_request(self, session_id: str, request_data: dict[str, object], context=None) -> EvaluationResult:
        del session_id, context
        messages = request_data.get("messages", [])
        latest_user = ""
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                latest_user = str(message.get("content", ""))
        if any(keyword in latest_user for keyword in self.request_keywords):
            return EvaluationResult(
                status=EvaluationStatus.VIOLATION,
                violations=[ViolationRecord(reason="request keyword violation", scope="request", engine="keyword")],
            )
        return EvaluationResult(status=EvaluationStatus.ALLOW)

    async def evaluate_response(self, session_id: str, response_data, request_data: dict[str, object], context=None) -> EvaluationResult:
        del session_id, request_data, context
        content = ""
        if isinstance(response_data, dict):
            choices = response_data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    content = str(message.get("content", ""))
        if any(keyword in content for keyword in self.response_keywords):
            return EvaluationResult(
                status=EvaluationStatus.VIOLATION,
                violations=[ViolationRecord(reason="response keyword violation", scope="response", engine="keyword")],
            )
        return EvaluationResult(status=EvaluationStatus.ALLOW)

    async def get_session_state(self, session_id: str):
        return self._sessions.get(session_id)

    async def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def _write_repo_suite(tmp_path: Path) -> None:
    (tmp_path / "openbias.yaml").write_text(
        "model: gpt-4o-mini\n"
        "evaluators:\n"
        "  - name: behavior\n"
        "    type: judge\n"
        "    phase: post_call\n",
        encoding="utf-8",
    )
    (tmp_path / "RULES.md").write_text("- Be safe\n", encoding="utf-8")
    suite_dir = tmp_path / "evals" / "suites"
    suite_dir.mkdir(parents=True, exist_ok=True)
    fixture = Path("tests/eval/fixtures/native_suite.yaml").read_text(encoding="utf-8")
    (suite_dir / "native_suite.yaml").write_text(fixture, encoding="utf-8")


def _mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.validate = MagicMock()
    settings.eval.suites = []
    return settings


def test_run_eval_writes_json_output_for_discovered_suites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_repo_suite(tmp_path)
    settings = _mock_settings()
    monkeypatch.setattr("openbias.cli_eval.Settings", MagicMock(return_value=settings))

    engine = KeywordEngine(
        request_keywords={"request-risk"},
        response_keywords={"unsafe answer", "still unsafe"},
        false_positive_keywords={"false-positive trigger"},
    )

    async def _fake_build_engine_for_policy(*, settings, config_path, rules_path):
        del settings, config_path, rules_path
        return engine

    monkeypatch.setattr("openbias.cli_eval.build_engine_for_policy", _fake_build_engine_for_policy)

    json_output = tmp_path / "artifacts" / "eval.json"
    results = run_eval(
        config=tmp_path / "openbias.yaml",
        suite_paths=(),
        json_output=json_output,
        verbose=False,
    )

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert len(results) == 1
    assert payload["suites"][0]["suite_name"] == "native-smoke"
    assert payload["suites"][0]["summary"]["exact_case_pass_rate"] == pytest.approx(1.0)


def test_run_eval_exits_nonzero_when_cases_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_repo_suite(tmp_path)
    settings = _mock_settings()
    monkeypatch.setattr("openbias.cli_eval.Settings", MagicMock(return_value=settings))

    engine = KeywordEngine(
        request_keywords=set(),
        response_keywords={"unsafe answer", "still unsafe"},
        false_positive_keywords={"false-positive trigger"},
    )

    async def _fake_build_engine_for_policy(*, settings, config_path, rules_path):
        del settings, config_path, rules_path
        return engine

    monkeypatch.setattr("openbias.cli_eval.build_engine_for_policy", _fake_build_engine_for_policy)

    with pytest.raises(SystemExit) as exc_info:
        run_eval(
            config=tmp_path / "openbias.yaml",
            suite_paths=(),
            json_output=None,
            verbose=False,
        )
    assert exc_info.value.code == 1
