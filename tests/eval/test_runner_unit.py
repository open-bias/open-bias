"""Unit tests for openbias.eval.runner (_split_turns, EvalRunner)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from openbias.eval.runner import EvalRunner, _split_turns
from openbias.policy.protocols import Decision, EngineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allow() -> EngineResult:
    return EngineResult(decision=Decision.ALLOW)


def _block() -> EngineResult:
    return EngineResult(decision=Decision.BLOCK)


def _mock_engine(
    request_result: EngineResult | None = None,
    response_result: EngineResult | None = None,
    engine_type: str = "mock",
) -> MagicMock:
    engine = MagicMock()
    engine.engine_type = engine_type
    engine.evaluate_request = AsyncMock(return_value=request_result or _allow())
    engine.evaluate_response = AsyncMock(return_value=response_result or _allow())
    return engine


# ---------------------------------------------------------------------------
# _split_turns
# ---------------------------------------------------------------------------

class TestSplitTurns:
    def test_empty(self):
        assert _split_turns([]) == []

    def test_only_system_message_no_turns(self):
        messages = [{"role": "system", "content": "sys"}]
        assert _split_turns(messages) == []

    def test_single_turn(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        turns = _split_turns(messages)
        assert len(turns) == 1
        assert turns[0]["assistant_message"]["content"] == "Hi there"
        assert turns[0]["messages_so_far"] == [{"role": "user", "content": "Hello"}]
        assert turns[0]["tool_messages"] == []

    def test_two_turns(self):
        messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "resp2"},
        ]
        turns = _split_turns(messages)
        assert len(turns) == 2
        assert turns[0]["assistant_message"]["content"] == "resp1"
        assert turns[1]["assistant_message"]["content"] == "resp2"
        # Second turn's messages_so_far includes first assistant message
        assert len(turns[1]["messages_so_far"]) == 3

    def test_turn_with_tool_messages(self):
        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "calling tool", "tool_calls": [{}]},
            {"role": "tool", "content": "tool result"},
            {"role": "user", "content": "next user"},
            {"role": "assistant", "content": "final response"},
        ]
        turns = _split_turns(messages)
        assert len(turns) == 2
        assert turns[0]["tool_messages"] == [{"role": "tool", "content": "tool result"}]
        assert turns[1]["tool_messages"] == []

    def test_multiple_tool_messages(self):
        messages = [
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "calling"},
            {"role": "tool", "content": "result1"},
            {"role": "tool", "content": "result2"},
            {"role": "assistant", "content": "done"},
        ]
        turns = _split_turns(messages)
        assert len(turns) == 2
        assert len(turns[0]["tool_messages"]) == 2

    def test_system_message_included_in_messages_so_far(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        turns = _split_turns(messages)
        assert len(turns) == 1
        # system + user should be in messages_so_far
        assert len(turns[0]["messages_so_far"]) == 2
        assert turns[0]["messages_so_far"][0]["role"] == "system"

    def test_buffer_carries_over_between_turns(self):
        messages = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "resp2"},
        ]
        turns = _split_turns(messages)
        # Turn 2's buffer should include turn1 user + turn1 assistant
        assert any(m["content"] == "resp1" for m in turns[1]["messages_so_far"])


# ---------------------------------------------------------------------------
# EvalRunner.run
# ---------------------------------------------------------------------------

class TestEvalRunnerRun:
    @pytest.fixture
    def runner(self):
        return EvalRunner()

    async def test_empty_messages(self, runner):
        engine = _mock_engine()
        result = await runner.run(engine, [])
        assert result.error is None
        assert result.turns == []
        assert result.engine_type == "mock"

    async def test_single_turn(self, runner):
        engine = _mock_engine()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = await runner.run(engine, messages)

        assert result.error is None
        assert len(result.turns) == 1
        assert result.turns[0].turn_index == 0
        assert result.turns[0].request_eval.decision == Decision.ALLOW
        assert result.turns[0].response_eval.decision == Decision.ALLOW

    async def test_uses_provided_session_id(self, runner):
        engine = _mock_engine()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = await runner.run(engine, messages, session_id="my-session")
        assert result.session_id == "my-session"

    async def test_auto_generates_session_id(self, runner):
        engine = _mock_engine()
        result = await runner.run(engine, [])
        assert result.session_id.startswith("eval-")

    async def test_engine_error_captured(self, runner):
        engine = _mock_engine()
        engine.evaluate_request = AsyncMock(side_effect=RuntimeError("engine broke"))
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = await runner.run(engine, messages)
        assert result.error == "engine broke"

    async def test_response_data_built_from_assistant_message(self, runner):
        engine = _mock_engine()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "my answer", "tool_calls": []},
        ]
        result = await runner.run(engine, messages)
        # Check that response_data passed to engine contains the assistant content
        call_args = engine.evaluate_response.call_args
        response_data = call_args.kwargs["response_data"]
        assert response_data["choices"][0]["message"]["content"] == "my answer"

    async def test_tool_calls_absent_yields_none(self, runner):
        engine = _mock_engine()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "no tools here"},
        ]
        await runner.run(engine, messages)
        call_args = engine.evaluate_response.call_args
        response_data = call_args.kwargs["response_data"]
        assert response_data["choices"][0]["message"]["tool_calls"] is None

    async def test_tool_calls_present_passed_through(self, runner):
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]
        engine = _mock_engine()
        messages = [
            {"role": "user", "content": "search for something"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ]
        await runner.run(engine, messages)
        call_args = engine.evaluate_response.call_args
        response_data = call_args.kwargs["response_data"]
        assert response_data["choices"][0]["message"]["tool_calls"] == tool_calls

    async def test_multiple_turns(self, runner):
        engine = _mock_engine()
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = await runner.run(engine, messages)
        assert len(result.turns) == 2
        assert result.turns[0].turn_index == 0
        assert result.turns[1].turn_index == 1


# ---------------------------------------------------------------------------
# EvalRunner.run_suite
# ---------------------------------------------------------------------------

class TestEvalRunnerRunSuite:
    @pytest.fixture
    def runner(self):
        return EvalRunner()

    async def test_run_suite_multiple_files(self, runner, tmp_path):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        f1 = tmp_path / "scenario1.json"
        f2 = tmp_path / "scenario2.json"
        f1.write_text(json.dumps(messages))
        f2.write_text(json.dumps(messages))

        engine = _mock_engine()
        results = await runner.run_suite(engine, [f1, f2])

        assert len(results) == 2
        assert results[0].scenario_path == str(f1)
        assert results[1].scenario_path == str(f2)

    async def test_run_suite_empty(self, runner):
        engine = _mock_engine()
        results = await runner.run_suite(engine, [])
        assert results == []

    async def test_run_suite_sets_scenario_path(self, runner, tmp_path):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        f = tmp_path / "test.json"
        f.write_text(json.dumps(messages))
        engine = _mock_engine()
        results = await runner.run_suite(engine, [f])
        assert results[0].scenario_path == str(f)

    async def test_run_suite_bad_json_records_error(self, runner, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json}")
        engine = _mock_engine()
        results = await runner.run_suite(engine, [bad])
        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].scenario_path == str(bad)
        assert results[0].turns == []
        assert results[0].engine_type == "mock"

    async def test_run_suite_permission_error_records_error(self, runner, tmp_path):
        no_access = tmp_path / "noaccess.json"
        no_access.write_text("[]")
        no_access.chmod(0o000)
        engine = _mock_engine()
        try:
            results = await runner.run_suite(engine, [no_access])
            assert len(results) == 1
            assert results[0].error is not None
            assert results[0].scenario_path == str(no_access)
            assert results[0].turns == []
        finally:
            no_access.chmod(0o644)

    async def test_run_suite_bad_file_does_not_lose_good_results(self, runner, tmp_path):
        good_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        good = tmp_path / "good.json"
        good.write_text(json.dumps(good_messages))
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json}")

        engine = _mock_engine()
        results = await runner.run_suite(engine, [good, bad, good])

        assert len(results) == 3
        assert results[0].error is None
        assert results[0].scenario_path == str(good)
        assert len(results[0].turns) == 1
        assert results[1].error is not None
        assert results[1].scenario_path == str(bad)
        assert results[2].error is None
        assert results[2].scenario_path == str(good)
