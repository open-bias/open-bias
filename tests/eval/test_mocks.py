"""Unit tests for opensentinel.eval.mocks."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

from opensentinel.eval.mocks import MockResponseSequence, apply_mock_provider

# ---------------------------------------------------------------------------
# MockResponseSequence
# ---------------------------------------------------------------------------

class TestMockResponseSequence:
    def test_empty_returns_empty_dict(self):
        seq = MockResponseSequence([])
        assert seq.next() == {}

    def test_single_response_cycles(self):
        seq = MockResponseSequence([{"a": 1}])
        assert seq.next() == {"a": 1}
        assert seq.next() == {"a": 1}
        assert seq.next() == {"a": 1}

    def test_two_responses_cycle(self):
        seq = MockResponseSequence([{"x": 1}, {"x": 2}])
        assert seq.next() == {"x": 1}
        assert seq.next() == {"x": 2}
        assert seq.next() == {"x": 1}

    def test_index_increments(self):
        items = [1, 2, 3]
        seq = MockResponseSequence(items)
        results = [seq.next() for _ in range(6)]
        assert results == [1, 2, 3, 1, 2, 3]

    def test_accepts_any_type(self):
        seq = MockResponseSequence(["string", 42, None, []])
        assert seq.next() == "string"
        assert seq.next() == 42
        assert seq.next() is None
        assert seq.next() == []


# ---------------------------------------------------------------------------
# apply_mock_provider
# ---------------------------------------------------------------------------

class TestApplyMockProvider:
    def test_no_responses_does_nothing(self):
        engine = MagicMock()
        # Should not raise, no attributes touched
        apply_mock_provider(engine, "judge", responses=None)

    def test_unknown_engine_type_logs_debug(self):
        engine = MagicMock()
        # Should not raise for unknown engine types
        apply_mock_provider(engine, "unknown_type", responses=['{"key": "val"}'])

    def test_judge_patches_client_call_judge(self):
        client = MagicMock()
        engine = MagicMock()
        engine._client = client

        responses = [json.dumps({"score": 0.9, "decision": "allow"})]
        apply_mock_provider(engine, "judge", responses=responses)

        # The call_judge attribute should now be a coroutine function
        import asyncio
        assert asyncio.iscoroutinefunction(client.call_judge)

    async def test_judge_mock_returns_parsed_json(self):
        client = MagicMock()
        engine = MagicMock()
        engine._client = client

        payload = {"score": 0.8, "label": "ok"}
        responses = [json.dumps(payload)]
        apply_mock_provider(engine, "judge", responses=responses)

        result = await client.call_judge(
            model_name="m",
            system_prompt="sp",
            user_prompt="up",
        )
        assert result == payload

    async def test_judge_mock_cycles_responses(self):
        client = MagicMock()
        engine = MagicMock()
        engine._client = client

        r1 = {"n": 1}
        r2 = {"n": 2}
        responses = [json.dumps(r1), json.dumps(r2)]
        apply_mock_provider(engine, "judge", responses=responses)

        first = await client.call_judge("m", "sp", "up")
        second = await client.call_judge("m", "sp", "up")
        third = await client.call_judge("m", "sp", "up")
        assert first == r1
        assert second == r2
        assert third == r1

    def test_judge_engine_without_client_logs_warning(self, caplog):
        engine = MagicMock(spec=[])  # no _client attribute
        with caplog.at_level(logging.WARNING, logger="opensentinel.eval.mocks"):
            apply_mock_provider(engine, "judge", responses=['{}'])
        assert "cannot apply mock" in caplog.text.lower()

    async def test_nemo_patches_rails_generate_async(self):
        rails = MagicMock()
        engine = MagicMock()
        engine._rails = rails

        response = {"output": "hello"}
        responses = [json.dumps(response)]
        apply_mock_provider(engine, "nemo", responses=responses)

        import asyncio
        assert asyncio.iscoroutinefunction(rails.generate_async)
        result = await rails.generate_async()
        assert result == response

    async def test_nemo_mock_cycles_responses(self):
        rails = MagicMock()
        engine = MagicMock()
        engine._rails = rails

        r1 = {"n": 1}
        r2 = {"n": 2}
        apply_mock_provider(engine, "nemo", responses=[json.dumps(r1), json.dumps(r2)])

        assert await rails.generate_async() == r1
        assert await rails.generate_async() == r2
        assert await rails.generate_async() == r1

    async def test_invalid_json_stored_as_string(self):
        client = MagicMock()
        engine = MagicMock()
        engine._client = client

        apply_mock_provider(engine, "judge", responses=["not-json"])
        # Should not raise; non-JSON stored as raw string
        result = await client.call_judge("m", "sp", "up")
        assert result == "not-json"
