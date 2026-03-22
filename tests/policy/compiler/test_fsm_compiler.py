"""Tests for FSM policy compiler (deterministic pipeline)."""

import pytest
from pathlib import Path

from opensentinel.policy.compiler.protocol import CompilationResult
from opensentinel.policy.compiler.registry import PolicyCompilerRegistry
from opensentinel.policy.engines.fsm.compiler import (
    FSMCompiler,
    compile_workflow,
    slugify,
)
from opensentinel.policy.engines.fsm.workflow.schema import (
    ConstraintType,
    SimpleWorkflowConfig,
    WorkflowDefinition,
)


class TestFSMCompilerRegistration:
    """Test FSM compiler is properly registered."""

    def test_fsm_compiler_registered(self):
        assert PolicyCompilerRegistry.is_registered("fsm")

    def test_create_fsm_compiler(self):
        from opensentinel.policy.engines.fsm.engine import FSMPolicyEngine

        engine = FSMPolicyEngine()
        compiler = engine.get_compiler()

        assert isinstance(compiler, FSMCompiler)
        assert compiler.engine_type == "fsm"


class TestSlugify:
    """Test slugify helper."""

    def test_basic(self):
        assert slugify("greet the customer") == "greet_the_customer"

    def test_strips_parentheticals(self):
        assert slugify("verify identity (if account action needed)") == "verify_identity"

    def test_collapses_runs(self):
        assert slugify("look up -- information") == "look_up_information"


class TestCompileWorkflow:
    """Test the deterministic compile_workflow pipeline."""

    @pytest.fixture
    def simple_config(self) -> SimpleWorkflowConfig:
        return SimpleWorkflowConfig(
            name="customer-support",
            steps=[
                "greet the customer",
                "understand their issue",
                "verify identity (if account action needed)",
                "take account action",
                "resolve and close",
            ],
            rules=[
                "verify identity before any account action",
                "never share internal system information",
                "must eventually resolve and close",
            ],
        )

    def test_produces_workflow_definition(self, simple_config: SimpleWorkflowConfig):
        result = compile_workflow(simple_config)
        assert isinstance(result, WorkflowDefinition)
        assert result.name == "customer-support"

    def test_step_count(self, simple_config: SimpleWorkflowConfig):
        result = compile_workflow(simple_config)
        # 5 steps + 1 hidden state for "share internal system information"
        assert len(result.states) >= 5

    def test_first_state_is_initial(self, simple_config: SimpleWorkflowConfig):
        result = compile_workflow(simple_config)
        assert result.states[0].is_initial is True

    def test_terminal_state_detected(self, simple_config: SimpleWorkflowConfig):
        result = compile_workflow(simple_config)
        terminal = [s for s in result.states if s.is_terminal]
        assert len(terminal) >= 1
        assert any("resolve" in s.name for s in terminal)

    def test_sequential_transitions(self, simple_config: SimpleWorkflowConfig):
        result = compile_workflow(simple_config)
        # At least n-1 sequential transitions for n steps
        assert len(result.transitions) >= 4

    def test_forward_skip_transition(self, simple_config: SimpleWorkflowConfig):
        """Any state can transition to any later state (forward-only)."""
        result = compile_workflow(simple_config)
        transition_pairs = {(t.from_state, t.to_state) for t in result.transitions}
        # understand_their_issue should be able to skip to take_account_action
        assert ("understand_their_issue", "take_account_action") in transition_pairs
        # greet should be able to skip to resolve_and_close
        assert ("greet_the_customer", "resolve_and_close") in transition_pairs

    def test_precedence_constraint(self, simple_config: SimpleWorkflowConfig):
        result = compile_workflow(simple_config)
        precedence = [c for c in result.constraints if c.type == ConstraintType.PRECEDENCE]
        assert len(precedence) == 1
        assert "verify" in precedence[0].target
        assert "account" in precedence[0].trigger

    def test_never_constraint_with_hidden_state(self, simple_config: SimpleWorkflowConfig):
        result = compile_workflow(simple_config)
        never = [c for c in result.constraints if c.type == ConstraintType.NEVER]
        assert len(never) == 1
        # Hidden state should be created for conceptual target
        hidden = [s for s in result.states if s.is_error]
        assert len(hidden) >= 1

    def test_eventually_constraint(self, simple_config: SimpleWorkflowConfig):
        result = compile_workflow(simple_config)
        eventually = [c for c in result.constraints if c.type == ConstraintType.EVENTUALLY]
        assert len(eventually) == 1
        assert "resolve" in eventually[0].target

    def test_constraints_have_messages(self, simple_config: SimpleWorkflowConfig):
        result = compile_workflow(simple_config)
        for c in result.constraints:
            assert c.message, f"Constraint {c.name} missing message"

    def test_classification_hints_generated(self, simple_config: SimpleWorkflowConfig):
        result = compile_workflow(simple_config)
        for state in result.states:
            if state.description:
                assert (
                    state.classification.patterns is not None
                    or state.classification.exemplars is not None
                ), f"State {state.name} missing classification hints"


class TestToolMapping:
    """Test tool-to-state mapping."""

    def test_tools_mapped_to_states(self):
        config = SimpleWorkflowConfig(
            name="test",
            steps=["verify identity", "process refund"],
            rules=[],
            tools={
                "verify": ["verify_customer_identity"],
                "refund": ["process_refund_tool"],
            },
        )
        result = compile_workflow(config)
        verify_state = next(s for s in result.states if "verify" in s.name)
        assert verify_state.classification.tool_calls is not None
        assert "verify_customer_identity" in verify_state.classification.tool_calls


class TestRuleParsing:
    """Test individual rule pattern matching."""

    def test_if_then_rule(self):
        config = SimpleWorkflowConfig(
            name="test",
            steps=["ask question", "provide answer"],
            rules=["if ask question then provide answer"],
        )
        result = compile_workflow(config)
        response = [c for c in result.constraints if c.type == ConstraintType.RESPONSE]
        assert len(response) == 1

    def test_unrecognized_rule_skipped(self):
        config = SimpleWorkflowConfig(
            name="test",
            steps=["step one", "step two"],
            rules=["this rule makes no sense at all"],
        )
        result = compile_workflow(config)
        assert len(result.constraints) == 0


class TestFSMCompilerInterface:
    """Test the PolicyCompiler interface wrapper."""

    @pytest.fixture
    def compiler(self):
        return FSMCompiler()

    async def test_compile_with_simple_config(self, compiler: FSMCompiler):
        config = SimpleWorkflowConfig(
            name="test",
            steps=["greet", "resolve and close"],
            rules=["must eventually resolve and close"],
        )
        result = await compiler.compile("ignored", context={"simple_config": config})
        assert result.success is True
        assert isinstance(result.config, WorkflowDefinition)

    async def test_compile_with_dict_config(self, compiler: FSMCompiler):
        result = await compiler.compile(
            "ignored",
            context={
                "simple_config": {
                    "name": "test",
                    "steps": ["greet", "resolve and close"],
                    "rules": [],
                }
            },
        )
        assert result.success is True

    async def test_compile_without_config_fails(self, compiler: FSMCompiler):
        result = await compiler.compile("natural language only")
        assert result.success is False
        assert any("simple_config" in e for e in result.errors)

    async def test_compile_with_invalid_config_fails(self, compiler: FSMCompiler):
        result = await compiler.compile(
            "ignored",
            context={"simple_config": {"name": "test"}},  # missing steps
        )
        assert result.success is False


class TestFSMCompilerExport:
    """Tests for YAML export."""

    @pytest.fixture
    def compiler(self):
        return FSMCompiler()

    @pytest.fixture
    def sample_workflow(self):
        return WorkflowDefinition(
            name="test-export",
            description="Test workflow for export",
            states=[
                {
                    "name": "greeting",
                    "is_initial": True,
                    "classification": {"patterns": ["hello", "hi"]},
                },
                {
                    "name": "resolution",
                    "is_terminal": True,
                },
            ],
            transitions=[
                {"from_state": "greeting", "to_state": "resolution"},
            ],
            constraints=[
                {
                    "name": "test_constraint",
                    "type": "eventually",
                    "target": "resolution",
                    "message": "Please work toward resolution.",
                },
            ],
        )

    def test_export_creates_yaml_file(self, compiler, sample_workflow, tmp_path):
        result = CompilationResult(success=True, config=sample_workflow)
        output_path = tmp_path / "workflow.yaml"

        compiler.export(result, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "test-export" in content
        assert "greeting" in content
        assert "resolution" in content

    def test_export_creates_parent_directories(self, compiler, sample_workflow, tmp_path):
        result = CompilationResult(success=True, config=sample_workflow)
        output_path = tmp_path / "nested" / "dir" / "workflow.yaml"

        compiler.export(result, output_path)

        assert output_path.exists()

    def test_export_failed_result_raises(self, compiler, tmp_path):
        result = CompilationResult.failure(["error"])
        output_path = tmp_path / "workflow.yaml"

        with pytest.raises(ValueError, match="Cannot export failed"):
            compiler.export(result, output_path)

    def test_exported_yaml_is_valid_workflow(self, compiler, sample_workflow, tmp_path):
        from opensentinel.policy.engines.fsm.workflow.parser import WorkflowParser

        result = CompilationResult(success=True, config=sample_workflow)
        output_path = tmp_path / "workflow.yaml"

        compiler.export(result, output_path)

        parsed = WorkflowParser.parse_file(output_path)
        assert parsed.name == "test-export"
        assert len(parsed.states) == 2
        assert len(parsed.constraints) == 1
