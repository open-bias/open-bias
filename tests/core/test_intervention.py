"""Tests for intervention system."""

import pytest

from opensentinel.core.intervention.strategies import (
    StrategyType,
    StrategyConfig,
    SystemPromptAppendStrategy,
    UserMessageInjectStrategy,
    ResponseModificationStrategy,
    WorkflowViolationError,
)
from opensentinel.policy.engines.fsm.intervention import InterventionHandler


class TestInterventionStrategies:
    """Tests for intervention strategies."""

    @pytest.fixture
    def sample_data(self):
        """Sample LLM request data."""
        return {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Help me with my account."},
            ],
        }

    @pytest.fixture
    def config(self):
        """Sample intervention config."""
        return StrategyConfig(
            strategy_type=StrategyType.SYSTEM_PROMPT_APPEND,
            message_template="Please verify identity first. Current state: {current_state}",
        )

    def test_system_prompt_append(self, sample_data, config):
        """Test system prompt append strategy."""
        strategy = SystemPromptAppendStrategy()
        context = {"current_state": "greeting"}

        result = strategy.apply(sample_data, config, context)

        # Should have modified system message
        system_msg = result["messages"][0]
        assert "[WORKFLOW GUIDANCE]" in system_msg["content"]
        assert "verify identity" in system_msg["content"]
        assert "greeting" in system_msg["content"]

    def test_system_prompt_append_no_existing_system(self, config):
        """Test system prompt append when no system message exists."""
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }
        strategy = SystemPromptAppendStrategy()
        context = {"current_state": "test"}

        result = strategy.apply(data, config, context)

        # Should have inserted system message
        assert result["messages"][0]["role"] == "system"
        assert "[WORKFLOW GUIDANCE]" in result["messages"][0]["content"]

    def test_user_message_inject(self, sample_data):
        """Test user message inject strategy."""
        config = StrategyConfig(
            strategy_type=StrategyType.USER_MESSAGE_INJECT,
            message_template="Please check this first.",
        )
        strategy = UserMessageInjectStrategy()
        context = {}

        result = strategy.apply(sample_data, config, context)

        # Should have injected a user message
        messages = result["messages"]
        # Find the injected message
        injected = [m for m in messages if "[System Note]" in m.get("content", "")]
        assert len(injected) == 1
        assert injected[0]["role"] == "user"

    def test_strategy_types(self):
        """Test that all strategy types exist."""
        types = {st.value for st in StrategyType}
        assert types == {
            "system_prompt_append",
            "user_message_inject",
            "response_modification",
        }

    def test_message_format_with_missing_key(self):
        """Test message formatting handles missing context keys."""
        strategy = SystemPromptAppendStrategy()
        template = "Value: {missing_key}"

        # Should not raise, just leave placeholder
        result = strategy.format_message(template, {})
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


class TestInterventionHandler:
    """Tests for InterventionHandler."""

    @pytest.fixture
    def handler(self):
        """Create InterventionHandler."""
        return InterventionHandler()

    @pytest.fixture
    def violation(self):
        """Create a sample constraint violation."""
        from opensentinel.policy.engines.fsm.workflow.constraints import (
            ConstraintViolation,
            ConstraintType,
        )

        return ConstraintViolation(
            constraint_name="verify_before_action",
            message="Please verify identity first.",
            constraint_type=ConstraintType.PRECEDENCE,
            details={"current_state": "start", "proposed_state": "action"},
        )

    def test_get_config_for_violation(self, handler, violation):
        """Test building StrategyConfig from a violation."""
        config = handler.get_config_for_violation(violation)

        assert config is not None
        assert config.message_template == "Please verify identity first."
        assert config.strategy_type == StrategyType.SYSTEM_PROMPT_APPEND

    def test_get_message(self, handler, violation):
        """Test getting message from a violation."""
        msg = handler.get_message(violation)

        assert msg == "Please verify identity first."

    def test_get_message_empty(self, handler):
        """Test getting message from a violation with no message."""
        from opensentinel.policy.engines.fsm.workflow.constraints import (
            ConstraintViolation,
            ConstraintType,
        )

        violation = ConstraintViolation(
            constraint_name="test",
            message="",
            constraint_type=ConstraintType.NEVER,
            details={},
        )

        assert handler.get_message(violation) is None
