"""
Tests for engine intervention integration.

Verifies InterventionHandler across engine types.
"""

import pytest
from unittest.mock import MagicMock

from opensentinel.policy.protocols import (
    PolicyEngine,
)


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
        assert callable(getattr(handler, "get_template", None))
        assert callable(getattr(handler, "list_interventions", None))

    def test_get_template_returns_string(self):
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
        template = handler.get_template("remind")
        assert template == "Stay focused."

    def test_get_template_with_block_prefix(self):
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
        # get_template returns the raw template string as defined in workflow
        template = handler.get_template("hard_stop")
        assert template == "block: This action is blocked."

    def test_get_template_with_inject_prefix(self):
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
        template = handler.get_template("inject_msg")
        assert template == "inject: Please clarify your request."

    def test_get_template_unknown_returns_none(self):
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
        assert handler.get_template("nonexistent") is None

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
