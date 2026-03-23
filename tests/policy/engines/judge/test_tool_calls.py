"""
Tests for tool call awareness in the judge engine.
"""

import pytest

from openbias.core.utils import extract_tool_calls
from openbias.policy.engines.judge import JudgePolicyEngine
from openbias.policy.engines.judge.prompts import (
    format_conversation_block,
    format_tool_calls_block,
    _classify_tool_operation,
)
from openbias.policy.protocols import Decision


class TestExtractToolCalls:
    """Tests for JudgePolicyEngine._extract_tool_calls()."""

    @pytest.fixture
    def engine(self):
        return JudgePolicyEngine()

    def test_extract_from_openai_dict(self, engine):
        response = {
            "choices": [{
                "message": {
                    "content": "Sure, I'll delete that.",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "function": {
                                "name": "delete_all_records",
                                "arguments": '{"table": "users"}',
                            },
                        },
                    ],
                },
            }],
        }
        result = extract_tool_calls(response)
        assert len(result) == 1
        assert result[0]["id"] == "call_abc123"
        assert result[0]["function_name"] == "delete_all_records"
        assert result[0]["arguments"] == '{"table": "users"}'

    def test_extract_multiple_tool_calls(self, engine):
        response = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                        },
                        {
                            "id": "call_2",
                            "function": {"name": "write_file", "arguments": '{"path": "b.txt"}'},
                        },
                    ],
                },
            }],
        }
        result = extract_tool_calls(response)
        assert len(result) == 2
        assert result[0]["function_name"] == "read_file"
        assert result[1]["function_name"] == "write_file"

    def test_extract_no_tool_calls(self, engine):
        response = {
            "choices": [{"message": {"content": "Hello!"}}],
        }
        result = extract_tool_calls(response)
        assert result == []

    def test_extract_from_string_response(self, engine):
        result = extract_tool_calls("just a string")
        assert result == []

    def test_extract_from_dict_without_choices(self, engine):
        response = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "search", "arguments": '{"q": "test"}'},
                },
            ],
        }
        result = extract_tool_calls(response)
        assert len(result) == 1
        assert result[0]["function_name"] == "search"

    def test_extract_from_object_format(self, engine):
        """Test extraction from SDK-style response objects with attributes."""

        class FunctionObj:
            name = "delete_database"
            arguments = '{"confirm": true}'

        class ToolCallObj:
            id = "call_xyz"
            function = FunctionObj()

        class MessageObj:
            content = "Deleting now."
            tool_calls = [ToolCallObj()]

        class ChoiceObj:
            message = MessageObj()

        class ResponseObj:
            choices = [ChoiceObj()]

        result = extract_tool_calls(ResponseObj())
        assert len(result) == 1
        assert result[0]["id"] == "call_xyz"
        assert result[0]["function_name"] == "delete_database"
        assert result[0]["arguments"] == '{"confirm": true}'

    def test_extract_empty_tool_calls_list(self, engine):
        response = {
            "choices": [{"message": {"content": "Hi", "tool_calls": []}}],
        }
        result = extract_tool_calls(response)
        assert result == []


class TestFormatConversationBlock:
    """Tests for format_conversation_block with tool call messages."""

    def test_basic_conversation(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = format_conversation_block(messages)
        assert "[System]: You are helpful." in result
        assert "[Turn 1 - user]: Hello" in result
        assert "[Turn 2 - assistant]: Hi there!" in result

    def test_assistant_with_tool_calls(self):
        messages = [
            {"role": "user", "content": "Delete the users table"},
            {
                "role": "assistant",
                "content": "Sure, I'll handle that.",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "function": {
                            "name": "delete_all_records",
                            "arguments": '{"table": "users"}',
                        },
                    },
                ],
            },
        ]
        result = format_conversation_block(messages)
        assert "[Turn 2 - assistant]: Sure, I'll handle that." in result
        assert '[Tool Call]: delete_all_records({"table": "users"}) [id: call_abc123]' in result

    def test_tool_result_message(self):
        messages = [
            {"role": "user", "content": "Search for docs"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "search", "arguments": '{"q": "docs"}'},
                    },
                ],
            },
            {
                "role": "tool",
                "content": "Found 3 documents.",
                "tool_call_id": "call_1",
            },
        ]
        result = format_conversation_block(messages)
        assert "[Tool Call]: search" in result
        assert "[Tool Result (call_1)]: Found 3 documents." in result

    def test_tool_result_does_not_increment_turn(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Searching..."},
            {"role": "tool", "content": "Result", "tool_call_id": "c1"},
            {"role": "assistant", "content": "Here you go."},
        ]
        result = format_conversation_block(messages)
        assert "[Turn 1 - user]" in result
        assert "[Turn 2 - assistant]: Searching..." in result
        assert "[Tool Result (c1)]" in result
        assert "[Turn 3 - assistant]: Here you go." in result

    def test_multiple_tool_calls_on_one_message(self):
        messages = [
            {
                "role": "assistant",
                "content": "Running both.",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "func_a", "arguments": "{}"}},
                    {"id": "c2", "function": {"name": "func_b", "arguments": "{}"}},
                ],
            },
        ]
        result = format_conversation_block(messages)
        assert "[Tool Call]: func_a({}) [id: c1]" in result
        assert "[Tool Call]: func_b({}) [id: c2]" in result


class TestFormatToolCallsBlock:
    """Tests for format_tool_calls_block."""

    def test_empty(self):
        assert format_tool_calls_block([]) == ""

    def test_single_tool_call(self):
        tool_calls = [
            {"id": "call_1", "function_name": "delete_records", "arguments": '{"table": "users"}'},
        ]
        result = format_tool_calls_block(tool_calls)
        assert "Tool calls in this response:" in result
        assert '- delete_records({"table": "users"}) [id: call_1]' in result

    def test_multiple_tool_calls(self):
        tool_calls = [
            {"id": "c1", "function_name": "read", "arguments": "{}"},
            {"id": "c2", "function_name": "write", "arguments": "{}"},
        ]
        result = format_tool_calls_block(tool_calls)
        assert "- read({}) [id: c1]" in result
        assert "- write({}) [id: c2]" in result

    def test_no_id(self):
        tool_calls = [
            {"function_name": "search", "arguments": '{"q": "test"}'},
        ]
        result = format_tool_calls_block(tool_calls)
        assert "- search" in result
        # No id suffix
        assert "[id:" not in result


class TestToolCallEndToEnd:
    """End-to-end: judge evaluates a response with tool calls."""

    @pytest.fixture
    def engine(self):
        return JudgePolicyEngine()

    @pytest.fixture
    def config(self):
        return {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
        }

    @pytest.fixture
    def request_with_tool_messages(self):
        return {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Delete all user records"},
            ],
        }

    @pytest.fixture
    def response_with_tool_calls(self):
        return {
            "choices": [{
                "message": {
                    "content": "I'll delete all records now.",
                    "tool_calls": [
                        {
                            "id": "call_dangerous",
                            "function": {
                                "name": "delete_all_records",
                                "arguments": '{"table": "users", "confirm": true}',
                            },
                        },
                    ],
                },
            }],
        }

    async def test_tool_calls_visible_to_judge(
        self, engine, config, request_with_tool_messages, response_with_tool_calls
    ):
        """Judge LLM receives tool call information in its prompt."""
        await engine.initialize(config)

        captured_prompts = []

        async def capture_judge_call(
            model_name: str, system_prompt: str, user_prompt: str, **kwargs: object
        ) -> dict:
            captured_prompts.append(user_prompt)
            return {
                "scores": [
                    {"criterion": "instruction_following", "score": 1, "reasoning": "ok",
                     "evidence": [], "confidence": 0.9},
                    {"criterion": "tool_use_safety", "score": 0, "reasoning": "Dangerous delete",
                     "evidence": ["delete_all_records"], "confidence": 0.95},
                    {"criterion": "no_hallucination", "score": 5, "reasoning": "ok",
                     "evidence": [], "confidence": 0.9},
                    {"criterion": "task_completion", "score": 4, "reasoning": "ok",
                     "evidence": [], "confidence": 0.8},
                ],
                "summary": "Dangerous tool call detected.",
            }

        engine._client.call_judge = capture_judge_call

        await engine.evaluate_response(
            "s1", response_with_tool_calls, request_with_tool_messages
        )

        # The judge prompt should contain the tool call info
        assert len(captured_prompts) >= 1
        prompt = captured_prompts[0]
        assert "delete_all_records" in prompt
        assert "call_dangerous" in prompt

    async def test_dangerous_tool_call_flagged(
        self, engine, config, request_with_tool_messages, response_with_tool_calls
    ):
        """Judge flags dangerous tool calls via low tool_use_safety score."""
        await engine.initialize(config)

        async def failing_judge(
            model_name: str, system_prompt: str, user_prompt: str, **kwargs: object
        ) -> dict:
            return {
                "scores": [
                    {"criterion": "instruction_following", "score": 1, "reasoning": "ok",
                     "evidence": [], "confidence": 0.9},
                    {"criterion": "tool_use_safety", "score": 0, "reasoning": "Unauthorized delete",
                     "evidence": ["delete_all_records"], "confidence": 0.95},
                    {"criterion": "no_hallucination", "score": 5, "reasoning": "ok",
                     "evidence": [], "confidence": 0.9},
                    {"criterion": "task_completion", "score": 3, "reasoning": "ok",
                     "evidence": [], "confidence": 0.8},
                ],
                "summary": "Dangerous: unauthorized data deletion.",
            }

        engine._client.call_judge = failing_judge

        result = await engine.evaluate_response(
            "s1", response_with_tool_calls, request_with_tool_messages
        )

        # Low composite score should trigger non-ALLOW decision
        assert result.decision in (Decision.INTERVENE, Decision.BLOCK)
        assert len(result.metadata.get("violations", [])) > 0

    async def test_response_without_tool_calls_still_works(
        self, engine, config, request_with_tool_messages
    ):
        """Responses without tool calls should still evaluate normally."""
        await engine.initialize(config)

        async def passing_judge(
            model_name: str, system_prompt: str, user_prompt: str, **kwargs: object
        ) -> dict:
            return {
                "scores": [
                    {"criterion": "instruction_following", "score": 1, "reasoning": "ok",
                     "evidence": [], "confidence": 0.9},
                    {"criterion": "tool_use_safety", "score": 1, "reasoning": "ok",
                     "evidence": [], "confidence": 0.9},
                    {"criterion": "no_hallucination", "score": 5, "reasoning": "ok",
                     "evidence": [], "confidence": 0.9},
                    {"criterion": "task_completion", "score": 5, "reasoning": "ok",
                     "evidence": [], "confidence": 0.9},
                ],
                "summary": "Good response.",
            }

        engine._client.call_judge = passing_judge
        simple_response = {"choices": [{"message": {"content": "Hello!"}}]}

        result = await engine.evaluate_response(
            "s1", simple_response, request_with_tool_messages
        )
        assert result.decision == Decision.ALLOW


class TestExtractToolDefinitions:
    """Tests for JudgePolicyEngine._extract_tool_definitions()."""

    @pytest.fixture
    def engine(self):
        return JudgePolicyEngine()

    def test_extract_openai_format(self, engine):
        request = {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "delete_user",
                        "description": "Permanently removes a user account and all associated data",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer", "description": "the user ID to delete"},
                            },
                            "required": ["id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_users",
                        "description": "Search for users by name or email",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "search term"},
                            },
                        },
                    },
                },
            ],
        }
        result = engine._extract_tool_definitions(request)
        assert "delete_user" in result
        assert result["delete_user"]["description"] == "Permanently removes a user account and all associated data"
        assert "id" in result["delete_user"]["parameters"]
        assert "search_users" in result
        assert "query" in result["search_users"]["parameters"]

    def test_extract_no_tools(self, engine):
        result = engine._extract_tool_definitions({"messages": []})
        assert result == {}

    def test_extract_tool_without_description(self, engine):
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "do_thing",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "string"},
                            },
                        },
                    },
                },
            ],
        }
        result = engine._extract_tool_definitions(request)
        assert "do_thing" in result
        assert "description" not in result["do_thing"]
        assert result["do_thing"]["parameters"]["x"] == "string"


class TestClassifyToolOperation:
    """Tests for the read/write/destructive classification heuristic."""

    def test_destructive_names(self):
        assert "destructive" in _classify_tool_operation("delete_user", "")
        assert "destructive" in _classify_tool_operation("remove_record", "")
        assert "destructive" in _classify_tool_operation("drop_table", "")

    def test_write_names(self):
        assert "write" in _classify_tool_operation("create_user", "")
        assert "write" in _classify_tool_operation("update_record", "")

    def test_read_only_names(self):
        assert _classify_tool_operation("get_user", "") == "read-only"
        assert _classify_tool_operation("list_items", "") == "read-only"
        assert _classify_tool_operation("search_docs", "") == "read-only"

    def test_description_overrides_neutral_name(self):
        assert "destructive" in _classify_tool_operation(
            "manage_user", "Permanently deletes the user"
        )


class TestFormatToolCallsBlockWithDefinitions:
    """Tests for format_tool_calls_block with tool definitions."""

    def test_with_definitions(self):
        tool_calls = [
            {"function_name": "delete_user", "arguments": '{"id": 123}'},
        ]
        definitions = {
            "delete_user": {
                "description": "Permanently removes a user account",
                "parameters": {"id": "integer — the user ID to delete"},
            },
        }
        result = format_tool_calls_block(tool_calls, definitions)
        assert "1. delete_user" in result
        assert "Description: Permanently removes a user account" in result
        assert "destructive" in result
        assert "Parameter: id (integer — the user ID to delete)" in result

    def test_without_definitions_uses_compact_format(self):
        tool_calls = [
            {"id": "c1", "function_name": "foo", "arguments": "{}"},
        ]
        result = format_tool_calls_block(tool_calls)
        assert "- foo({}) [id: c1]" in result
        assert "Description:" not in result

    def test_mixed_known_and_unknown(self):
        tool_calls = [
            {"function_name": "known_tool", "arguments": "{}"},
            {"function_name": "unknown_tool", "arguments": "{}"},
        ]
        definitions = {
            "known_tool": {"description": "A known tool"},
        }
        result = format_tool_calls_block(tool_calls, definitions)
        assert "Description: A known tool" in result
        assert "2. unknown_tool" in result

    def test_empty_tool_calls(self):
        assert format_tool_calls_block([], {"foo": {}}) == ""
