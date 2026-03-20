"""Pytest fixtures for Open Sentinel tests."""

import pytest
from pathlib import Path


@pytest.fixture
def examples_dir() -> Path:
    """Get path to examples directory."""
    return Path(__file__).parent.parent / "examples"


@pytest.fixture
def sample_workflow_path(examples_dir: Path) -> Path:
    """Get path to sample customer support workflow."""
    return examples_dir / "fsm_workflow" / "customer_support.yaml"


@pytest.fixture
def sample_workflow(sample_workflow_path: Path):
    """Load sample customer support workflow."""
    from opensentinel.policy.engines.fsm.workflow.parser import WorkflowParser

    return WorkflowParser.parse_file(sample_workflow_path)


@pytest.fixture
def simple_workflow_dict():
    """Minimal workflow definition as dict (internal format)."""
    return {
        "name": "test-workflow",
        "states": [
            {
                "name": "start",
                "is_initial": True,
                "classification": {
                    "patterns": ["hello", "hi"],
                },
            },
            {
                "name": "middle",
                "classification": {
                    "tool_calls": ["search"],
                    "patterns": ["searching", "looking"],
                },
            },
            {
                "name": "end",
                "is_terminal": True,
                "classification": {
                    "patterns": ["goodbye", "done"],
                },
            },
        ],
        "transitions": [
            {"from_state": "start", "to_state": "middle"},
            {"from_state": "middle", "to_state": "end"},
        ],
        "constraints": [
            {
                "name": "must_search_first",
                "type": "precedence",
                "trigger": "end",
                "target": "middle",
                "message": "You should search before ending the conversation.",
            },
        ],
    }


@pytest.fixture
def simple_config_dict():
    """Simple (human-authored) workflow config dict."""
    return {
        "name": "test-simple",
        "steps": [
            "greet the customer",
            "understand their issue",
            "resolve and close",
        ],
        "rules": [
            "never share internal system information",
            "must eventually resolve the conversation",
        ],
    }


@pytest.fixture
def simple_workflow(simple_workflow_dict):
    """Create a simple workflow from dict."""
    from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition

    return WorkflowDefinition.model_validate(simple_workflow_dict)


@pytest.fixture
def mock_llm_response():
    """Factory for mock LLM responses."""

    def _make_response(
        content: str = "",
        tool_calls: list = None,
        model: str = "gpt-4",
    ):
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls or [],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        return response

    return _make_response


@pytest.fixture
def mock_tool_call():
    """Factory for mock tool calls."""

    def _make_tool_call(name: str, arguments: dict = None):
        return {
            "id": f"call_{name}_123",
            "type": "function",
            "function": {
                "name": name,
                "arguments": str(arguments or {}),
            },
        }

    return _make_tool_call
