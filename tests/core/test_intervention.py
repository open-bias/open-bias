"""Tests for intervention system."""

import pytest

from opensentinel.core.intervention.strategies import (
    StrategyType,
    InterventionConfig,
    SystemPromptAppendStrategy,
    UserMessageInjectStrategy,
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
        return InterventionConfig(
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
        config = InterventionConfig(
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
        """Test that only the two supported strategy types exist."""
        types = {st.value for st in StrategyType}
        assert types == {"system_prompt_append", "user_message_inject"}

    def test_message_format_with_missing_key(self):
        """Test message formatting handles missing context keys."""
        strategy = SystemPromptAppendStrategy()
        template = "Value: {missing_key}"

        # Should not raise, just leave placeholder
        result = strategy.format_message(template, {})
        assert template == result  # Returns original if key missing


class TestInterventionHandler:
    """Tests for InterventionHandler."""

    @pytest.fixture
    def workflow(self, simple_workflow):
        """Use simple workflow fixture."""
        return simple_workflow

    @pytest.fixture
    def handler(self, workflow):
        """Create InterventionHandler."""
        return InterventionHandler(workflow)

    def test_list_interventions(self, handler):
        """Test listing available interventions."""
        interventions = handler.list_interventions()

        assert "prompt_search" in interventions

    def test_get_intervention_info(self, handler):
        """Test getting intervention info."""
        info = handler.get_intervention_info("prompt_search")

        assert info is not None
        assert info["name"] == "prompt_search"
        assert "strategy" in info
        assert "template" in info

    def test_get_config_known(self, handler):
        """Test getting config for a known intervention."""
        config = handler.get_config("prompt_search")

        assert config is not None
        assert config.message_template is not None

    def test_get_config_unknown(self, handler):
        """Test getting config for unknown intervention returns None."""
        config = handler.get_config("nonexistent")

        assert config is None
