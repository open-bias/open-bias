"""Shared utilities for response data extraction.

Consolidates response content, tool call, and usage extraction logic
that was previously duplicated across 7+ locations in engines and hooks.
"""

from __future__ import annotations

from typing import Any


def extract_response_content(response_data: Any) -> str:
    """Extract text content from response data.

    Handles None, str, dict (OpenAI choices format), and object (LiteLLM SDK) responses.
    """
    if response_data is None:
        return ""

    if isinstance(response_data, str):
        return response_data

    if isinstance(response_data, dict):
        # OpenAI format: choices[0].message.content
        choices = response_data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "") or ""
        # Direct content field (NeMo result, simple message)
        if "content" in response_data:
            return response_data.get("content", "") or ""
        return ""

    # Object with choices attribute (LiteLLM ModelResponse)
    if hasattr(response_data, "choices") and response_data.choices:
        choice = response_data.choices[0]
        if hasattr(choice, "message") and choice.message:
            return getattr(choice.message, "content", "") or ""
        if hasattr(choice, "text"):
            return choice.text or ""
        return ""

    # Object with direct content attribute
    if hasattr(response_data, "content"):
        return response_data.content or ""

    # Object with choices attribute but empty/None — not a valid response
    if hasattr(response_data, "choices"):
        return ""

    return str(response_data)


def extract_tool_calls(response_data: Any) -> list[dict[str, Any]]:
    """Extract tool calls from response data in rich format.

    Returns a list of dicts with 'id', 'function_name', and 'arguments'.
    Handles OpenAI dict format and object format responses.
    """
    tool_calls: list[dict[str, Any]] = []

    if isinstance(response_data, dict):
        choices = response_data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            raw_calls = message.get("tool_calls", [])
        else:
            raw_calls = response_data.get("tool_calls", [])

        for tc in raw_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "function_name": func.get("name", ""),
                    "arguments": func.get("arguments", ""),
                })

    elif hasattr(response_data, "choices") and response_data.choices:
        choice = response_data.choices[0]
        if hasattr(choice, "message") and choice.message:
            raw_calls = getattr(choice.message, "tool_calls", None) or []
            for tc in raw_calls:
                if hasattr(tc, "function") and tc.function:
                    tool_calls.append({
                        "id": getattr(tc, "id", ""),
                        "function_name": getattr(tc.function, "name", ""),
                        "arguments": getattr(tc.function, "arguments", ""),
                    })

    return tool_calls


def extract_tool_call_names(response_data: Any) -> list[str]:
    """Extract tool call function names from response data.

    Convenience wrapper returning just names. Handles dict format, object format,
    and direct tool_calls fields with name-only entries.
    """
    names: list[str] = []

    if isinstance(response_data, dict):
        choices = response_data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            raw_calls = message.get("tool_calls", [])
        elif "tool_calls" in response_data:
            raw_calls = response_data.get("tool_calls", [])
        else:
            raw_calls = []

        for tc in raw_calls:
            if isinstance(tc, dict):
                if func_name := tc.get("function", {}).get("name"):
                    names.append(func_name)
                elif name := tc.get("name"):
                    names.append(name)

    elif hasattr(response_data, "choices") and response_data.choices:
        choice = response_data.choices[0]
        if hasattr(choice, "message") and choice.message:
            raw_calls = getattr(choice.message, "tool_calls", None) or []
            for tc in raw_calls:
                if hasattr(tc, "function") and tc.function:
                    name = getattr(tc.function, "name", None)
                    if name:
                        names.append(name)

    return names


def extract_usage_info(response: Any) -> dict[str, int] | None:
    """Extract token usage info from a response object.

    Returns dict with prompt_tokens, completion_tokens, total_tokens, or None.
    """
    if hasattr(response, "usage") and response.usage:
        return {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        }
    return None
