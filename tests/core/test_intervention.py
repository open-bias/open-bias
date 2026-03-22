"""Tests for intervention system."""

from opensentinel.core.intervention.strategies import (
    ResponseModificationStrategy,
    StrategyType,
    SystemPromptAppendStrategy,
    UserMessageInjectStrategy,
    format_message,
)


class TestInterventionStrategies:
    """Tests for intervention strategies."""

    def test_strategy_types(self):
        """Test that all strategy types exist."""
        types = {st.value for st in StrategyType}
        assert types == {
            "system_prompt_append",
            "user_message_inject",
            "response_modification",
        }

    def test_format_message_with_missing_key(self):
        """Test message formatting handles missing context keys."""
        template = "Value: {missing_key}"

        # Should not raise, just leave placeholder
        result = format_message(template, {})
        assert template == result  # Returns original if key missing

    def test_user_message_inject_after_last_user(self):
        """Fix 2A: injected guidance must appear AFTER the last user message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]

        result = UserMessageInjectStrategy.merge(messages, "Check this.")
        # Guidance should be right after the last user message (index 3)
        assert result[4]["role"] == "user"
        assert "[System Note]" in result[4]["content"]
        assert "Check this." in result[4]["content"]
        # Last user message should be untouched at index 3
        assert result[3]["content"] == "Second question"

    def test_user_message_inject_not_before_last_user(self):
        """Fix 2A: guidance must NOT appear before the last user message."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "My question"},
        ]

        result = UserMessageInjectStrategy.merge(messages, "Verify identity.")
        # The [System Note] should come AFTER "My question", not before
        assert result[1]["content"] == "My question"
        assert result[2]["content"] == "[System Note]: Verify identity."

    def test_user_message_inject_no_user_messages(self):
        """When no user message exists, guidance is appended at end."""
        messages = [
            {"role": "system", "content": "System prompt"},
        ]

        result = UserMessageInjectStrategy.merge(messages, "Guidance.")
        assert len(result) == 2
        assert result[-1]["content"] == "[System Note]: Guidance."


class TestSystemPromptAppendStrategy:
    """Tests for SystemPromptAppendStrategy."""

    def test_append_to_string_content(self):
        """Guidance appended to plain string system message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = SystemPromptAppendStrategy.merge(messages, "Stay on task.")
        assert isinstance(result[0]["content"], str)
        assert "You are helpful." in result[0]["content"]
        assert "[WORKFLOW GUIDANCE]: Stay on task." in result[0]["content"]

    def test_append_to_multimodal_list_content(self):
        """Guidance appended as a text part when system content is a list."""
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are helpful."},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            },
            {"role": "user", "content": "Describe the image."},
        ]
        result = SystemPromptAppendStrategy.merge(messages, "Stay on task.")
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 3
        assert content[2]["type"] == "text"
        assert "[WORKFLOW GUIDANCE]: Stay on task." in content[2]["text"]

    def test_no_system_message_creates_one(self):
        """When no system message exists, one is inserted at index 0."""
        messages = [{"role": "user", "content": "Hello"}]
        result = SystemPromptAppendStrategy.merge(messages, "Be careful.")
        assert result[0]["role"] == "system"
        assert "[WORKFLOW GUIDANCE]: Be careful." in result[0]["content"]

    def test_append_to_empty_string_content(self):
        """Guidance appended when system content is empty string."""
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": "Hi"}]
        result = SystemPromptAppendStrategy.merge(messages, "Guide.")
        assert "[WORKFLOW GUIDANCE]: Guide." in result[0]["content"]

    def test_append_to_none_content(self):
        """Guidance appended when system content is None."""
        messages = [{"role": "system", "content": None}, {"role": "user", "content": "Hi"}]
        result = SystemPromptAppendStrategy.merge(messages, "Guide.")
        assert "[WORKFLOW GUIDANCE]: Guide." in result[0]["content"]

    def test_does_not_mutate_original(self):
        """Original messages list is not mutated."""
        messages = [{"role": "system", "content": "Original."}]
        result = SystemPromptAppendStrategy.merge(messages, "Added.")
        assert "Added" not in messages[0]["content"]
        assert "Added" in result[0]["content"]


class TestResponseModificationStrategy:
    """Tests for ResponseModificationStrategy."""

    def _make_response_obj(self, content="Hello", tool_calls=None):
        """Create a mock LiteLLM-style response object."""
        from unittest.mock import MagicMock

        response = MagicMock()
        message = MagicMock()
        message.content = content
        message.tool_calls = tool_calls
        choice = MagicMock()
        choice.message = message
        response.choices = [choice]
        return response

    def _make_response_dict(self, content="Hello", tool_calls=None):
        """Create a dict-style response."""
        msg = {"role": "assistant", "content": content}
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        return {"choices": [{"message": msg}]}

    def test_append_warning_to_object_response(self):
        """Message appends a policy warning to response content."""
        response = self._make_response_obj("I'll delete those records.")
        result = ResponseModificationStrategy.apply_to_response(
            response, message="Unauthorized destructive operation"
        )
        assert "[POLICY WARNING]" in result.choices[0].message.content
        assert "Unauthorized destructive operation" in result.choices[0].message.content
        assert "I'll delete those records." in result.choices[0].message.content

    def test_append_warning_to_dict_response(self):
        """Message appends warning to dict-style response."""
        response = self._make_response_dict("Original content")
        result = ResponseModificationStrategy.apply_to_response(
            response, message="Warning text"
        )
        content = result["choices"][0]["message"]["content"]
        assert "Original content" in content
        assert "[POLICY WARNING]: Warning text" in content

    def test_replace_content_with_modified_messages(self):
        """modified_messages replaces entire response content."""
        response = self._make_response_obj("Dangerous output")
        modified = [{"role": "assistant", "content": "I cannot do that."}]
        result = ResponseModificationStrategy.apply_to_response(
            response, modified_messages=modified
        )
        assert result.choices[0].message.content == "I cannot do that."

    def test_replace_strips_tool_calls(self):
        """modified_messages replacement also strips tool calls."""
        tool_calls = [{"id": "call_123", "function": {"name": "delete_all"}}]
        response = self._make_response_obj("Bad", tool_calls=tool_calls)
        modified = [{"role": "assistant", "content": "Safe response."}]
        result = ResponseModificationStrategy.apply_to_response(
            response, modified_messages=modified
        )
        assert result.choices[0].message.content == "Safe response."
        assert result.choices[0].message.tool_calls is None

    def test_strip_tool_calls_dict_response(self):
        """Tool calls stripped from dict-style response."""
        tool_calls = [{"id": "call_123", "function": {"name": "delete_all"}}]
        response = self._make_response_dict("Bad", tool_calls=tool_calls)
        modified = [{"role": "assistant", "content": "Safe."}]
        result = ResponseModificationStrategy.apply_to_response(
            response, modified_messages=modified
        )
        assert "tool_calls" not in result["choices"][0]["message"]

    def test_no_message_no_modified_messages_unchanged(self):
        """No message and no modified_messages — response unchanged."""
        response = self._make_response_obj("Original")
        result = ResponseModificationStrategy.apply_to_response(response)
        assert result.choices[0].message.content == "Original"

    def test_modified_messages_takes_precedence(self):
        """modified_messages takes precedence over message."""
        response = self._make_response_obj("Original")
        result = ResponseModificationStrategy.apply_to_response(
            response,
            message="This warning should not appear",
            modified_messages=[{"role": "assistant", "content": "Replaced."}],
        )
        assert result.choices[0].message.content == "Replaced."
        assert "warning" not in result.choices[0].message.content.lower()


