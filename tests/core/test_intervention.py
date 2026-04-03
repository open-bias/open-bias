"""Tests for intervention system."""

from openbias.core.intervention.strategies import (
    StrategyType,
    SystemPromptAppendStrategy,
    UserMessageInjectStrategy,
    format_message,
)


class TestInterventionStrategies:
    """Tests for intervention strategies."""

    def test_strategy_types(self):
        """Only request-time strategy types are public."""
        types = {st.value for st in StrategyType}
        assert types == {
            "system_prompt_append",
            "user_message_inject",
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
        assert "<system-reminder>" in result[0]["content"]
        assert "Please address this message and continue with your tasks." in result[0]["content"]
        assert "Stay on task." in result[0]["content"]
        assert "</system-reminder>" in result[0]["content"]

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
        assert "<system-reminder>" in content[2]["text"]
        assert "Stay on task." in content[2]["text"]

    def test_no_system_message_creates_one(self):
        """When no system message exists, one is inserted at index 0."""
        messages = [{"role": "user", "content": "Hello"}]
        result = SystemPromptAppendStrategy.merge(messages, "Be careful.")
        assert result[0]["role"] == "system"
        assert "<system-reminder>" in result[0]["content"]
        assert "Be careful." in result[0]["content"]

    def test_append_to_empty_string_content(self):
        """Guidance appended when system content is empty string."""
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": "Hi"}]
        result = SystemPromptAppendStrategy.merge(messages, "Guide.")
        assert "<system-reminder>" in result[0]["content"]
        assert "Guide." in result[0]["content"]

    def test_append_to_none_content(self):
        """Guidance appended when system content is None."""
        messages = [{"role": "system", "content": None}, {"role": "user", "content": "Hi"}]
        result = SystemPromptAppendStrategy.merge(messages, "Guide.")
        assert "<system-reminder>" in result[0]["content"]
        assert "Guide." in result[0]["content"]

    def test_does_not_mutate_original(self):
        """Original messages list is not mutated."""
        messages = [{"role": "system", "content": "Original."}]
        result = SystemPromptAppendStrategy.merge(messages, "Added.")
        assert "Added" not in messages[0]["content"]
        assert "Added" in result[0]["content"]

    def test_cleanup_rules_are_strategy_specific(self):
        """System prompt append does not register response cleanup markers."""
        rules = SystemPromptAppendStrategy.cleanup_rules()
        assert rules == []


class TestUserMessageInjectStrategy:
    """Tests for UserMessageInjectStrategy cleanup behavior."""

    def test_cleanup_rules_are_strategy_specific(self):
        """Injected user cleanup strips note markers without system markers."""
        rules = UserMessageInjectStrategy.cleanup_rules()
        assert "[REPAIR-INSTRUCTION]" in rules
        assert "[END-REPAIR-INSTRUCTION]" in rules
        assert "[System Note]" in rules
        assert "<system-reminder>" not in rules

