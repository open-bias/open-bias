"""
Tests for engine intervention integration.

Verifies InterventionHandler across engine types.
"""

import pytest
from unittest.mock import MagicMock

from opensentinel.policy.protocols import (
    PolicyEngine,
)
from opensentinel.core.intervention.strategies import InterventionConfig, StrategyType


# ---------------------------------------------------------------------------
# LLM engine InterventionHandler
# ---------------------------------------------------------------------------


class TestLLMInterventionHandler:
    """LLMPolicyEngine InterventionHandler integration."""

    def test_handler_satisfies_protocol(self):
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[],
            interventions={"remind": "Stay focused."},
        )
        handler = InterventionHandler(workflow)
        assert callable(getattr(handler, "get_config", None))
        assert callable(getattr(handler, "list_interventions", None))

    def test_get_config_returns_config(self):
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[],
            interventions={"remind": "Stay focused."},
        )
        handler = InterventionHandler(workflow)
        config = handler.get_config("remind")
        assert config is not None
        assert isinstance(config, InterventionConfig)
        assert config.message_template == "Stay focused."
        assert config.strategy_type == StrategyType.SYSTEM_PROMPT_APPEND

    def test_get_config_with_block_prefix(self):
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[],
            interventions={"hard_stop": "block: This action is blocked."},
        )
        handler = InterventionHandler(workflow)
        config = handler.get_config("hard_stop")
        assert config is not None
        assert config.strategy_type == StrategyType.SYSTEM_PROMPT_APPEND
        assert config.message_template == "This action is blocked."

    def test_get_config_with_inject_prefix(self):
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[],
            interventions={"inject_msg": "inject: Please clarify your request."},
        )
        handler = InterventionHandler(workflow)
        config = handler.get_config("inject_msg")
        assert config is not None
        assert config.strategy_type == StrategyType.USER_MESSAGE_INJECT
        assert config.message_template == "Please clarify your request."

    def test_get_config_with_remind_prefix(self):
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[],
            interventions={"reminder": "remind: Remember the policy."},
        )
        handler = InterventionHandler(workflow)
        config = handler.get_config("reminder")
        assert config is not None
        assert config.strategy_type == StrategyType.SYSTEM_PROMPT_APPEND

    def test_get_config_unknown_returns_none(self):
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[{"name": "start", "is_initial": True}],
            transitions=[],
            constraints=[],
            interventions={},
        )
        handler = InterventionHandler(workflow)
        assert handler.get_config("nonexistent") is None

    def test_list_interventions(self):
        from opensentinel.policy.engines.llm.intervention import InterventionHandler
        from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            version="1.0",
            states=[{"name": "start", "is_initial": True}],
            transitions=[],
            constraints=[],
            interventions={"a": "msg a", "b": "msg b"},
        )
        handler = InterventionHandler(workflow)
        names = handler.list_interventions()
        assert set(names) == {"a", "b"}
