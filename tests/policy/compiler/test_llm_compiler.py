"""
Tests for LLMCompiler.

Tests prompt building, response parsing, export, and validation
without making real LLM calls.
"""

import json

import pytest
import yaml
from pathlib import Path
from unittest.mock import AsyncMock, patch

from openbias.policy.engines.llm.compiler import LLMCompiler
from openbias.policy.compiler.protocol import CompilationResult, PolicyCompiler
from openbias.policy.engines.fsm.workflow.schema import WorkflowDefinition


@pytest.fixture
def compiler():
    return LLMCompiler()


@pytest.fixture
def valid_llm_response():
    """A valid parsed JSON response from the LLM."""
    return {
        "name": "customer-support",
        "description": "Customer support workflow",
        "states": [
            {
                "name": "greeting",
                "description": "Greet the customer",
                "is_initial": True,
                "classification": {
                    "exemplars": ["Hello! How can I help you?", "Welcome!"],
                },
            },
            {
                "name": "resolve_issue",
                "description": "Resolve the customer's issue",
                "classification": {
                    "exemplars": ["I've fixed that for you", "The issue is resolved"],
                },
            },
            {
                "name": "closing",
                "description": "Close the conversation",
                "is_terminal": True,
                "classification": {
                    "exemplars": ["Anything else?", "Thank you for contacting us"],
                },
            },
        ],
        "transitions": [
            {"from_state": "greeting", "to_state": "resolve_issue"},
            {"from_state": "resolve_issue", "to_state": "closing"},
        ],
        "constraints": [
            {
                "name": "must_resolve",
                "type": "eventually",
                "target": "resolve_issue",
                "message": "Must resolve the issue before closing.",
            },
        ],
    }


@pytest.fixture
def valid_workflow(valid_llm_response):
    """A valid WorkflowDefinition instance."""
    return WorkflowDefinition.model_validate(valid_llm_response)


class TestLLMCompilerProperties:
    """Test basic compiler properties."""

    def test_engine_type(self, compiler):
        assert compiler.engine_type == "llm"

    def test_is_policy_compiler(self, compiler):
        assert isinstance(compiler, PolicyCompiler)


class TestBuildCompilationPrompt:
    """Test _build_compilation_prompt."""

    def test_includes_rules_text(self, compiler):
        prompt = compiler._build_compilation_prompt("Greet the customer first")
        assert "Greet the customer first" in prompt

    def test_includes_schema_instructions(self, compiler):
        prompt = compiler._build_compilation_prompt("some rules")
        assert "WorkflowDefinition" in prompt
        assert "states" in prompt
        assert "constraints" in prompt

    def test_includes_domain_context(self, compiler):
        prompt = compiler._build_compilation_prompt(
            "some rules", context={"domain": "healthcare"}
        )
        assert "healthcare" in prompt

    def test_works_without_context(self, compiler):
        prompt = compiler._build_compilation_prompt("some rules")
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestParseCompilationResponse:
    """Test _parse_compilation_response."""

    def test_success(self, compiler, valid_llm_response):
        result = compiler._parse_compilation_response(
            valid_llm_response, "Greet the customer"
        )
        assert result.success is True
        assert result.config is not None
        assert isinstance(result.config, WorkflowDefinition)
        assert len(result.errors) == 0

    def test_missing_states(self, compiler):
        response = {"name": "test", "transitions": []}
        result = compiler._parse_compilation_response(response, "test")
        assert result.success is False
        assert any("states" in e for e in result.errors)

    def test_empty_states(self, compiler):
        response = {"name": "test", "states": []}
        result = compiler._parse_compilation_response(response, "test")
        assert result.success is False
        assert any("states" in e for e in result.errors)

    def test_no_initial_state_auto_marks_first(self, compiler):
        response = {
            "name": "test",
            "states": [
                {"name": "step_one", "classification": {}},
            ],
        }
        result = compiler._parse_compilation_response(response, "test")
        assert result.success is True
        assert result.config.states[0].is_initial is True
        assert any("initial" in w.lower() for w in result.warnings)

    def test_metadata_includes_source(self, compiler, valid_llm_response):
        result = compiler._parse_compilation_response(
            valid_llm_response, "Greet the customer"
        )
        assert "source" in result.metadata
        assert "Greet the customer" in result.metadata["source"]

    def test_metadata_includes_counts(self, compiler, valid_llm_response):
        result = compiler._parse_compilation_response(
            valid_llm_response, "test rules"
        )
        assert result.metadata["state_count"] == 3
        assert result.metadata["constraint_count"] == 1

    def test_strips_unknown_top_level_keys(self, compiler):
        response = {
            "name": "test",
            "states": [
                {"name": "start", "is_initial": True, "classification": {}},
            ],
            "extra_key": "should be stripped",
        }
        result = compiler._parse_compilation_response(response, "test")
        assert result.success is True
        assert any("extra_key" in w for w in result.warnings)

    def test_slugifies_state_names(self, compiler):
        response = {
            "name": "test",
            "states": [
                {
                    "name": "Greet The Customer!",
                    "is_initial": True,
                    "classification": {},
                },
            ],
        }
        result = compiler._parse_compilation_response(response, "test")
        assert result.success is True
        assert result.config.states[0].name == "greet_the_customer"

    def test_invalid_workflow_returns_failure(self, compiler):
        response = {
            "name": "test",
            "states": [
                {"name": "a", "is_initial": True, "classification": {}},
            ],
            "transitions": [
                {"from_state": "a", "to_state": "nonexistent"},
            ],
        }
        result = compiler._parse_compilation_response(response, "test")
        assert result.success is False
        assert any("validation" in e.lower() for e in result.errors)


class TestValidateResult:
    """Test validate_result."""

    def test_valid_result(self, compiler, valid_workflow):
        result = CompilationResult(success=True, config=valid_workflow)
        errors = compiler.validate_result(result)
        assert errors == []

    def test_failed_result(self, compiler):
        result = CompilationResult.failure(["some error"])
        errors = compiler.validate_result(result)
        assert len(errors) > 0

    def test_non_workflow_config(self, compiler):
        result = CompilationResult(success=True, config={"not": "a workflow"})
        errors = compiler.validate_result(result)
        assert any("WorkflowDefinition" in e for e in errors)


class TestExport:
    """Test export to YAML file."""

    def test_export_creates_file(self, compiler, valid_workflow, tmp_path):
        result = CompilationResult(success=True, config=valid_workflow)
        output = tmp_path / "workflow.yaml"

        compiler.export(result, output)

        assert output.exists()
        data = yaml.safe_load(output.read_text())
        assert data["name"] == "customer-support"
        assert len(data["states"]) == 3

    def test_export_includes_transitions(self, compiler, valid_workflow, tmp_path):
        result = CompilationResult(success=True, config=valid_workflow)
        output = tmp_path / "workflow.yaml"

        compiler.export(result, output)

        data = yaml.safe_load(output.read_text())
        assert len(data["transitions"]) == 2

    def test_export_includes_constraints(self, compiler, valid_workflow, tmp_path):
        result = CompilationResult(success=True, config=valid_workflow)
        output = tmp_path / "workflow.yaml"

        compiler.export(result, output)

        data = yaml.safe_load(output.read_text())
        assert len(data["constraints"]) == 1
        assert data["constraints"][0]["type"] == "eventually"

    def test_export_fails_on_unsuccessful_result(self, compiler, tmp_path):
        result = CompilationResult.failure(["something went wrong"])
        with pytest.raises(ValueError, match="Cannot export failed"):
            compiler.export(result, tmp_path / "out.yaml")

    def test_export_creates_parent_dirs(self, compiler, valid_workflow, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "workflow.yaml"
        result = CompilationResult(success=True, config=valid_workflow)

        compiler.export(result, deep_path)
        assert deep_path.exists()


class TestCompileEndToEnd:
    """Test the full compile flow with mocked LLM."""

    async def test_compile_success(self, compiler, valid_llm_response):
        with patch.object(
            compiler,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=json.dumps(valid_llm_response),
        ):
            result = await compiler.compile("Greet the customer. Resolve their issue.")

        assert result.success is True
        assert isinstance(result.config, WorkflowDefinition)
        assert len(result.config.states) == 3

    async def test_compile_handles_invalid_json(self, compiler):
        with patch.object(
            compiler,
            "_call_llm",
            new_callable=AsyncMock,
            return_value="not valid json {{{",
        ):
            result = await compiler.compile("test rules")

        assert result.success is False
        assert any("JSON" in e for e in result.errors)

    async def test_compile_handles_llm_error(self, compiler):
        with patch.object(
            compiler,
            "_call_llm",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM unavailable"),
        ):
            result = await compiler.compile("test rules")

        assert result.success is False
        assert any("RuntimeError" in e for e in result.errors)

    async def test_compile_empty_rules(self, compiler):
        """Empty rules should produce a valid minimal workflow."""
        minimal_response = {
            "name": "minimal",
            "states": [
                {"name": "processing", "is_initial": True, "is_terminal": True, "classification": {}},
            ],
        }
        with patch.object(
            compiler,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=json.dumps(minimal_response),
        ):
            result = await compiler.compile("Be professional.")

        assert result.success is True
        assert len(result.config.states) == 1
