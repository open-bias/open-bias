"""Tests for workflow parsing and validation."""

import pytest
from pathlib import Path

from openbias.policy.engines.fsm.workflow.parser import WorkflowParser
from openbias.policy.engines.fsm.workflow.schema import (
    ConstraintType,
    SimpleWorkflowConfig,
    WorkflowDefinition,
)


class TestWorkflowParser:
    """Tests for WorkflowParser."""

    def test_parse_file_yaml(self, sample_workflow_path: Path):
        """Test parsing a YAML workflow file (simple format auto-compiled)."""
        workflow = WorkflowParser.parse_file(sample_workflow_path)

        assert workflow.name == "customer-support-agent"
        assert len(workflow.states) > 0

    def test_parse_string_yaml_internal_format(self):
        """Test parsing internal format YAML from string."""
        yaml_content = """
name: test-workflow
states:
  - name: start
    is_initial: true
"""
        workflow = WorkflowParser.parse_string(yaml_content)

        assert workflow.name == "test-workflow"
        assert len(workflow.states) == 1
        assert workflow.states[0].is_initial is True

    def test_parse_string_yaml_simple_format(self):
        """Test parsing simple format YAML — auto-compiled to internal."""
        yaml_content = """
name: test-simple
steps:
  - greet the customer
  - resolve and close
rules:
  - must eventually resolve and close
"""
        workflow = WorkflowParser.parse_string(yaml_content)

        assert workflow.name == "test-simple"
        assert len(workflow.states) >= 2
        assert len(workflow.constraints) >= 1

    def test_parse_dict_internal_format(self, simple_workflow_dict):
        """Test parsing internal format from dictionary."""
        workflow = WorkflowParser.parse_dict(simple_workflow_dict)

        assert workflow.name == "test-workflow"
        assert len(workflow.states) == 3

    def test_parse_dict_simple_format(self, simple_config_dict):
        """Test parsing simple format from dictionary — auto-compiled."""
        workflow = WorkflowParser.parse_dict(simple_config_dict)

        assert workflow.name == "test-simple"
        assert len(workflow.states) >= 3
        assert any(s.is_initial for s in workflow.states)

    def test_parse_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            WorkflowParser.parse_file("/nonexistent/path.yaml")

    def test_validate_file(self, sample_workflow_path: Path):
        """Test workflow validation."""
        is_valid, message = WorkflowParser.validate_file(sample_workflow_path)

        assert is_valid is True
        assert "Valid workflow" in message

    def test_format_detection_steps_key(self):
        """'steps' key routes through compiler."""
        data = {
            "name": "test",
            "steps": ["greet", "resolve and close"],
            "rules": [],
        }
        workflow = WorkflowParser.parse_dict(data)
        assert isinstance(workflow, WorkflowDefinition)
        assert len(workflow.states) >= 2

    def test_format_detection_states_key(self):
        """'states' key loads directly as internal format."""
        data = {
            "name": "test",
            "states": [{"name": "start", "is_initial": True}],
        }
        workflow = WorkflowParser.parse_dict(data)
        assert isinstance(workflow, WorkflowDefinition)
        assert len(workflow.states) == 1


class TestWorkflowSchema:
    """Tests for workflow schema validation."""

    def test_requires_initial_state(self):
        """Test that workflow must have an initial state."""
        with pytest.raises(ValueError, match="initial state"):
            WorkflowDefinition.model_validate(
                {
                    "name": "test",
                    "states": [{"name": "not_initial"}],
                }
            )

    def test_transition_references_valid_states(self):
        """Test that transitions must reference valid states."""
        with pytest.raises(ValueError, match="unknown state"):
            WorkflowDefinition.model_validate(
                {
                    "name": "test",
                    "states": [{"name": "start", "is_initial": True}],
                    "transitions": [{"from_state": "start", "to_state": "nonexistent"}],
                }
            )

    def test_constraint_references_valid_states(self):
        """Test that non-NEVER constraints must reference valid states."""
        with pytest.raises(ValueError, match="unknown"):
            WorkflowDefinition.model_validate(
                {
                    "name": "test",
                    "states": [{"name": "start", "is_initial": True}],
                    "constraints": [
                        {
                            "name": "test",
                            "type": "eventually",
                            "target": "nonexistent",
                        }
                    ],
                }
            )

    def test_never_constraint_allows_conceptual_target(self):
        """NEVER constraints can reference states not in the workflow."""
        workflow = WorkflowDefinition.model_validate(
            {
                "name": "test",
                "states": [{"name": "start", "is_initial": True}],
                "constraints": [
                    {
                        "name": "no_sharing",
                        "type": "never",
                        "target": "share_internal_info",
                    }
                ],
            }
        )
        assert len(workflow.constraints) == 1

    def test_constraint_requires_parameters(self):
        """Test that constraints require appropriate parameters."""
        with pytest.raises(ValueError, match="requires"):
            WorkflowDefinition.model_validate(
                {
                    "name": "test",
                    "states": [{"name": "start", "is_initial": True}],
                    "constraints": [
                        {
                            "name": "test",
                            "type": "precedence",
                            # Missing trigger and target
                        }
                    ],
                }
            )

    def test_valid_constraint_types(self):
        """Test all constraint types are valid."""
        assert set(ct.value for ct in ConstraintType) == {
            "eventually", "never", "response", "precedence",
        }

    def test_get_state(self, simple_workflow):
        """Test getting state by name."""
        state = simple_workflow.get_state("start")
        assert state is not None
        assert state.name == "start"

        assert simple_workflow.get_state("nonexistent") is None

    def test_get_initial_states(self, simple_workflow):
        """Test getting initial states."""
        initial = simple_workflow.get_initial_states()
        assert len(initial) == 1
        assert initial[0].name == "start"

    def test_get_terminal_states(self, simple_workflow):
        """Test getting terminal states."""
        terminal = simple_workflow.get_terminal_states()
        assert len(terminal) == 1
        assert terminal[0].name == "end"

    def test_simple_workflow_config_validation(self):
        """Test SimpleWorkflowConfig model."""
        config = SimpleWorkflowConfig(
            name="test",
            steps=["step one", "step two"],
            rules=["never do bad things"],
        )
        assert config.name == "test"
        assert config.tools is None

    def test_simple_workflow_config_with_tools(self):
        config = SimpleWorkflowConfig(
            name="test",
            steps=["step one"],
            rules=[],
            tools={"step one": ["tool_a", "tool_b"]},
        )
        assert config.tools == {"step one": ["tool_a", "tool_b"]}
