"""Tests for openbias.core.utils response extraction utilities."""

from __future__ import annotations

from types import SimpleNamespace

from openbias.core.utils import (
    extract_response_content,
    extract_tool_call_names,
    extract_tool_calls,
    extract_usage_info,
)


class TestExtractResponseContent:
    def test_none(self) -> None:
        assert extract_response_content(None) == ""

    def test_string(self) -> None:
        assert extract_response_content("hello") == "hello"

    def test_dict_choices_format(self) -> None:
        data = {"choices": [{"message": {"content": "response text"}}]}
        assert extract_response_content(data) == "response text"

    def test_dict_choices_none_content(self) -> None:
        data = {"choices": [{"message": {"content": None}}]}
        assert extract_response_content(data) == ""

    def test_dict_direct_content(self) -> None:
        data = {"content": "direct content"}
        assert extract_response_content(data) == "direct content"

    def test_dict_empty_choices(self) -> None:
        data = {"choices": []}
        assert extract_response_content(data) == ""

    def test_dict_no_relevant_keys(self) -> None:
        data = {"other": "value"}
        assert extract_response_content(data) == ""

    def test_object_choices_format(self) -> None:
        message = SimpleNamespace(content="obj content")
        choice = SimpleNamespace(message=message)
        response = SimpleNamespace(choices=[choice])
        assert extract_response_content(response) == "obj content"

    def test_object_text_format(self) -> None:
        choice = SimpleNamespace(message=None, text="text content")
        response = SimpleNamespace(choices=[choice])
        assert extract_response_content(response) == "text content"

    def test_object_direct_content(self) -> None:
        response = SimpleNamespace(content="direct", choices=None)
        assert extract_response_content(response) == "direct"

    def test_fallback_to_str(self) -> None:
        assert extract_response_content(42) == "42"

    def test_object_empty_choices(self) -> None:
        response = SimpleNamespace(choices=[])
        assert extract_response_content(response) == ""


class TestExtractToolCalls:
    def test_dict_choices_format(self) -> None:
        data = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                    }]
                }
            }]
        }
        result = extract_tool_calls(data)
        assert len(result) == 1
        assert result[0]["id"] == "call_1"
        assert result[0]["function_name"] == "read_file"
        assert result[0]["arguments"] == '{"path": "a.py"}'

    def test_dict_direct_tool_calls(self) -> None:
        data = {
            "tool_calls": [{
                "id": "call_2",
                "function": {"name": "write_file", "arguments": "{}"},
            }]
        }
        result = extract_tool_calls(data)
        assert len(result) == 1
        assert result[0]["function_name"] == "write_file"

    def test_object_format(self) -> None:
        func = SimpleNamespace(name="search", arguments='{"q": "test"}')
        tc = SimpleNamespace(id="call_3", function=func)
        message = SimpleNamespace(tool_calls=[tc])
        choice = SimpleNamespace(message=message)
        response = SimpleNamespace(choices=[choice])
        result = extract_tool_calls(response)
        assert len(result) == 1
        assert result[0]["function_name"] == "search"

    def test_no_tool_calls(self) -> None:
        data = {"choices": [{"message": {"content": "hi"}}]}
        assert extract_tool_calls(data) == []

    def test_none_input(self) -> None:
        assert extract_tool_calls(None) == []

    def test_string_input(self) -> None:
        assert extract_tool_calls("text") == []


class TestExtractToolCallNames:
    def test_dict_choices_format(self) -> None:
        data = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"function": {"name": "read_file"}},
                        {"function": {"name": "write_file"}},
                    ]
                }
            }]
        }
        assert extract_tool_call_names(data) == ["read_file", "write_file"]

    def test_dict_direct_tool_calls_with_name_field(self) -> None:
        data = {"tool_calls": [{"name": "simple_tool"}]}
        assert extract_tool_call_names(data) == ["simple_tool"]

    def test_object_format(self) -> None:
        func = SimpleNamespace(name="search")
        tc = SimpleNamespace(function=func)
        message = SimpleNamespace(tool_calls=[tc])
        choice = SimpleNamespace(message=message)
        response = SimpleNamespace(choices=[choice])
        assert extract_tool_call_names(response) == ["search"]

    def test_empty(self) -> None:
        assert extract_tool_call_names({}) == []
        assert extract_tool_call_names(None) == []


class TestExtractUsageInfo:
    def test_with_usage(self) -> None:
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = SimpleNamespace(usage=usage)
        result = extract_usage_info(response)
        assert result == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }

    def test_no_usage(self) -> None:
        response = SimpleNamespace(usage=None)
        assert extract_usage_info(response) is None

    def test_no_usage_attr(self) -> None:
        assert extract_usage_info("text") is None

    def test_partial_usage(self) -> None:
        usage = SimpleNamespace(prompt_tokens=5)
        response = SimpleNamespace(usage=usage)
        result = extract_usage_info(response)
        assert result is not None
        assert result["prompt_tokens"] == 5
        assert result["completion_tokens"] == 0
