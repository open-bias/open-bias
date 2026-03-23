"""
Tests for engine intervention integration.

Verifies InterventionHandler across engine types.
"""

import pytest
from unittest.mock import MagicMock

from openbias.policy.protocols import (
    PolicyEngine,
)


# ---------------------------------------------------------------------------
# LLM engine InterventionHandler
# ---------------------------------------------------------------------------


class TestLLMInterventionHandler:
    """LLMPolicyEngine InterventionHandler integration."""

    def test_handler_satisfies_protocol(self):
        from openbias.policy.engines.llm.intervention import InterventionHandler
        from openbias.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[
                {
                    "name": "remind",
                    "type": "never",
                    "target": "off_topic",
                    "message": "Stay focused.",
                }
            ],
        )
        handler = InterventionHandler(workflow)
        assert callable(getattr(handler, "get_template", None))
        assert callable(getattr(handler, "list_interventions", None))

    def test_get_template_returns_string(self):
        from openbias.policy.engines.llm.intervention import InterventionHandler
        from openbias.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[
                {
                    "name": "remind",
                    "type": "never",
                    "target": "off_topic",
                    "message": "Stay focused.",
                }
            ],
        )
        handler = InterventionHandler(workflow)
        template = handler.get_template("remind")
        assert template == "Stay focused."

    def test_get_template_with_block_message(self):
        from openbias.policy.engines.llm.intervention import InterventionHandler
        from openbias.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[
                {
                    "name": "hard_stop",
                    "type": "never",
                    "target": "blocked_action",
                    "message": "This action is not permitted.",
                }
            ],
        )
        handler = InterventionHandler(workflow)
        # get_template returns the message string defined on the constraint
        template = handler.get_template("hard_stop")
        assert template == "This action is not permitted."

    def test_get_template_with_clarification_message(self):
        from openbias.policy.engines.llm.intervention import InterventionHandler
        from openbias.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            states=[
                {"name": "start", "is_initial": True},
                {"name": "end", "is_terminal": True},
            ],
            transitions=[{"from_state": "start", "to_state": "end"}],
            constraints=[
                {
                    "name": "inject_msg",
                    "type": "never",
                    "target": "ambiguous_request",
                    "message": "Please clarify your request.",
                }
            ],
        )
        handler = InterventionHandler(workflow)
        template = handler.get_template("inject_msg")
        assert template == "Please clarify your request."

    def test_get_template_unknown_returns_none(self):
        from openbias.policy.engines.llm.intervention import InterventionHandler
        from openbias.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            states=[{"name": "start", "is_initial": True}],
            transitions=[],
            constraints=[],
        )
        handler = InterventionHandler(workflow)
        assert handler.get_template("nonexistent") is None

    def test_list_interventions(self):
        from openbias.policy.engines.llm.intervention import InterventionHandler
        from openbias.policy.engines.fsm.workflow.schema import WorkflowDefinition

        workflow = WorkflowDefinition(
            name="test",
            states=[{"name": "start", "is_initial": True}],
            transitions=[],
            constraints=[
                {
                    "name": "constraint_a",
                    "type": "never",
                    "target": "forbidden_a",
                    "message": "msg a",
                },
                {
                    "name": "constraint_b",
                    "type": "never",
                    "target": "forbidden_b",
                    "message": "msg b",
                },
            ],
        )
        handler = InterventionHandler(workflow)
        names = handler.list_interventions()
        assert set(names) == {"constraint_a", "constraint_b"}
