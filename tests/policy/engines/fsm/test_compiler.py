"""Tests for the deterministic FSM compiler pipeline."""

import pytest

from opensentinel.policy.engines.fsm.compiler import (
    compile_workflow,
    slugify,
    _parse_steps,
    _infer_transitions,
    _parse_rules,
    _map_tools,
    _generate_hints,
    _resolve_state,
    _generate_patterns,
)
from opensentinel.policy.engines.fsm.workflow.schema import (
    ConstraintType,
    SimpleWorkflowConfig,
    State,
)


class TestSlugify:
    """Tests for slugify()."""

    def test_simple(self):
        assert slugify("greet the customer") == "greet_the_customer"

    def test_strips_parentheticals(self):
        assert slugify("verify identity (if account action needed)") == "verify_identity"

    def test_lowercases(self):
        assert slugify("Resolve And Close") == "resolve_and_close"

    def test_strips_special_chars(self):
        assert slugify("look up info / take action") == "look_up_info_take_action"

    def test_collapses_underscores(self):
        assert slugify("a   b   c") == "a_b_c"


class TestParseSteps:
    """Tests for _parse_steps()."""

    def test_basic_steps(self):
        steps = ["greet the customer", "understand their issue", "resolve and close"]
        states = _parse_steps(steps)

        assert len(states) == 3
        assert states[0].name == "greet_the_customer"
        assert states[0].is_initial is True
        assert states[0].is_terminal is False

    def test_terminal_detection(self):
        steps = ["start", "resolve and close"]
        states = _parse_steps(steps)

        assert states[1].is_terminal is True

    def test_terminal_keywords(self):
        """Various terminal keywords are detected."""
        for keyword in ["resolve", "close", "end", "finish", "complete", "done"]:
            states = _parse_steps(["start", f"{keyword} the task"])
            assert states[1].is_terminal is True, f"'{keyword}' not detected as terminal"

    def test_first_step_is_initial(self):
        states = _parse_steps(["step one", "step two"])
        assert states[0].is_initial is True
        assert states[1].is_initial is False

    def test_description_preserved(self):
        steps = ["verify identity (if account action needed)"]
        states = _parse_steps(steps)
        assert states[0].description == "verify identity (if account action needed)"

    def test_parenthetical_stripped_from_name(self):
        steps = ["verify identity (if account action needed)"]
        states = _parse_steps(steps)
        assert states[0].name == "verify_identity"


class TestInferTransitions:
    """Tests for _infer_transitions()."""

    def test_sequential_transitions(self):
        steps = ["greet", "understand", "resolve and close"]
        states = _parse_steps(steps)
        transitions = _infer_transitions(states, steps)

        pairs = [(t.from_state, t.to_state) for t in transitions]
        assert ("greet", "understand") in pairs
        assert ("understand", "resolve_and_close") in pairs

    def test_conditional_skip_transition(self):
        steps = [
            "greet the customer",
            "verify identity (if account action needed)",
            "take account action",
        ]
        states = _parse_steps(steps)
        transitions = _infer_transitions(states, steps)

        pairs = [(t.from_state, t.to_state) for t in transitions]
        # Sequential
        assert ("greet_the_customer", "verify_identity") in pairs
        assert ("verify_identity", "take_account_action") in pairs
        # Skip: greet → take_account_action (bypass conditional verify)
        assert ("greet_the_customer", "take_account_action") in pairs

    def test_no_duplicate_transitions(self):
        steps = ["a", "b (if needed)", "c"]
        states = _parse_steps(steps)
        transitions = _infer_transitions(states, steps)

        pairs = [(t.from_state, t.to_state) for t in transitions]
        # Check no duplicates
        assert len(pairs) == len(set(pairs))


class TestParseRules:
    """Tests for _parse_rules()."""

    def _state_names(self, names: list[str]) -> list[State]:
        return [State(name=n, is_initial=(i == 0)) for i, n in enumerate(names)]

    def test_precedence_rule(self):
        states = self._state_names(["verify_identity", "account_action"])
        constraints, hidden = _parse_rules(
            ["verify identity before any account action"], states
        )

        assert len(constraints) == 1
        c = constraints[0]
        assert c.type == ConstraintType.PRECEDENCE
        assert c.target == "verify_identity"
        assert c.trigger == "account_action"
        assert len(hidden) == 0

    def test_precedence_not_matched_mid_string(self):
        """A RESPONSE rule containing 'before' should not match as PRECEDENCE."""
        states = self._state_names(["action", "check", "proceeding"])
        constraints, hidden = _parse_rules(
            ["if action then check before proceeding"], states
        )

        assert len(constraints) == 1
        c = constraints[0]
        assert c.type == ConstraintType.RESPONSE

    def test_never_rule_creates_hidden_state(self):
        states = self._state_names(["greet", "resolve"])
        constraints, hidden = _parse_rules(
            ["never share internal system information"], states
        )

        assert len(constraints) == 1
        c = constraints[0]
        assert c.type == ConstraintType.NEVER
        assert len(hidden) == 1
        assert hidden[0].name == c.target
        assert hidden[0].is_error is True

    def test_never_rule_existing_state_no_hidden(self):
        """NEVER targeting an existing state doesn't create a hidden state."""
        states = self._state_names(["greet", "forbidden"])
        constraints, hidden = _parse_rules(["never forbidden"], states)

        assert len(constraints) == 1
        assert constraints[0].target == "forbidden"
        assert len(hidden) == 0

    def test_eventually_rule(self):
        states = self._state_names(["greet", "resolve_and_close"])
        constraints, hidden = _parse_rules(
            ["must eventually resolve the conversation"], states
        )

        assert len(constraints) == 1
        c = constraints[0]
        assert c.type == ConstraintType.EVENTUALLY
        assert len(hidden) == 0

    def test_response_rule(self):
        states = self._state_names(["escalation", "notify_supervisor"])
        constraints, hidden = _parse_rules(
            ["if escalation then must notify supervisor"], states
        )

        assert len(constraints) == 1
        c = constraints[0]
        assert c.type == ConstraintType.RESPONSE
        assert len(hidden) == 0

    def test_constraint_gets_message(self):
        states = self._state_names(["greet"])
        constraints, _ = _parse_rules(["never do bad things"], states)

        assert constraints[0].message.startswith("Policy: ")

    def test_unrecognized_rule_skipped(self):
        states = self._state_names(["greet"])
        constraints, hidden = _parse_rules(["this is not a valid rule pattern"], states)

        assert len(constraints) == 0
        assert len(hidden) == 0


class TestMapTools:
    """Tests for _map_tools()."""

    def test_substring_match(self):
        steps = ["greet the customer", "verify identity", "take account action"]
        states = _parse_steps(steps)
        tools = {"verify identity": ["verify_id", "check_account"]}

        _map_tools(states, tools, steps)

        verify_state = states[1]
        assert verify_state.classification.tool_calls == ["verify_id", "check_account"]

    def test_case_insensitive(self):
        steps = ["Look Up Information"]
        states = _parse_steps(steps)
        tools = {"look up information": ["search_kb"]}

        _map_tools(states, tools, steps)

        assert states[0].classification.tool_calls == ["search_kb"]

    def test_no_tools_noop(self):
        steps = ["greet"]
        states = _parse_steps(steps)
        _map_tools(states, None, steps)

        assert states[0].classification.tool_calls is None

    def test_unmatched_tool_key(self):
        steps = ["greet"]
        states = _parse_steps(steps)
        tools = {"nonexistent step": ["tool_a"]}

        _map_tools(states, tools, steps)

        assert states[0].classification.tool_calls is None


class TestGenerateHints:
    """Tests for _generate_hints()."""

    def test_generates_patterns(self):
        states = _parse_steps(["greet the customer"])
        _generate_hints(states)

        assert states[0].classification.patterns is not None
        assert len(states[0].classification.patterns) > 0

    def test_generates_exemplars(self):
        states = _parse_steps(["greet the customer"])
        _generate_hints(states)

        assert states[0].classification.exemplars is not None
        assert len(states[0].classification.exemplars) > 0
        assert any("greet" in e.lower() for e in states[0].classification.exemplars)

    def test_does_not_overwrite_existing_patterns(self):
        states = _parse_steps(["greet the customer"])
        states[0].classification.patterns = ["custom_pattern"]
        _generate_hints(states)

        assert states[0].classification.patterns == ["custom_pattern"]

    def test_hidden_state_gets_hints(self):
        """Hidden states (from NEVER rules) also get classification hints."""
        states = [
            State(
                name="share_internal_info",
                description="share internal system information",
                is_initial=False,
                is_error=True,
            )
        ]
        _generate_hints(states)

        assert states[0].classification.patterns is not None
        assert states[0].classification.exemplars is not None


class TestCompileWorkflow:
    """End-to-end tests for compile_workflow()."""

    def test_basic_compilation(self):
        config = SimpleWorkflowConfig(
            name="test",
            steps=["greet the customer", "resolve and close"],
            rules=["must eventually resolve and close"],
        )
        workflow = compile_workflow(config)

        assert workflow.name == "test"
        assert len(workflow.states) >= 2
        assert any(s.is_initial for s in workflow.states)
        assert any(s.is_terminal for s in workflow.states)
        assert len(workflow.constraints) == 1
        assert workflow.constraints[0].type == ConstraintType.EVENTUALLY

    def test_compilation_with_tools(self):
        config = SimpleWorkflowConfig(
            name="test",
            steps=["verify identity", "take account action"],
            rules=["verify identity before any account action"],
            tools={"verify identity": ["verify_id"], "account action": ["process_refund"]},
        )
        workflow = compile_workflow(config)

        verify_state = workflow.get_state("verify_identity")
        assert verify_state is not None
        assert verify_state.classification.tool_calls == ["verify_id"]

        action_state = workflow.get_state("take_account_action")
        assert action_state is not None
        assert action_state.classification.tool_calls == ["process_refund"]

    def test_compilation_with_never_hidden_state(self):
        config = SimpleWorkflowConfig(
            name="test",
            steps=["greet", "resolve and close"],
            rules=["never share internal system information"],
        )
        workflow = compile_workflow(config)

        # Hidden state should be added
        hidden = workflow.get_state("share_internal_system_information")
        assert hidden is not None
        assert hidden.is_error is True

        # Constraint references it
        never_constraint = [
            c for c in workflow.constraints if c.type == ConstraintType.NEVER
        ]
        assert len(never_constraint) == 1
        assert never_constraint[0].target == hidden.name

    def test_compilation_generates_classification_hints(self):
        config = SimpleWorkflowConfig(
            name="test",
            steps=["greet the customer", "resolve and close"],
            rules=[],
        )
        workflow = compile_workflow(config)

        for state in workflow.states:
            assert (
                state.classification.patterns is not None
                or state.classification.tool_calls is not None
            ), f"State '{state.name}' has no classification hints"

    def test_customer_support_example(self):
        """Full compilation of the customer support example."""
        config = SimpleWorkflowConfig(
            name="customer-support-agent",
            steps=[
                "greet the customer",
                "understand their issue",
                "verify identity (if account action needed)",
                "look up information or take account action",
                "resolve and close",
            ],
            rules=[
                "verify identity before any account action",
                "never share internal system information",
                "must eventually resolve and close",
            ],
            tools={
                "verify identity": ["verify_identity", "check_account"],
                "account action": ["process_refund", "change_subscription"],
                "look up information": ["search_kb", "get_article"],
            },
        )
        workflow = compile_workflow(config)

        assert workflow.name == "customer-support-agent"
        assert len(workflow.states) >= 5  # 5 steps + hidden NEVER state
        assert len(workflow.constraints) == 3
        assert any(s.is_initial for s in workflow.states)
        assert any(s.is_terminal for s in workflow.states)

        # Check constraint types
        types = {c.type for c in workflow.constraints}
        assert ConstraintType.PRECEDENCE in types
        assert ConstraintType.NEVER in types
        assert ConstraintType.EVENTUALLY in types


class TestResolveState:
    """Tests for _resolve_state() word-overlap threshold."""

    def test_single_word_overlap_does_not_match(self):
        """Single shared word should not resolve to a state (threshold >= 2)."""
        states = ["resolve_and_close", "greet_the_customer"]
        result = _resolve_state("resolve_the_conversation", states)
        # Falls through to returning the slug as-is
        assert result == "resolve_the_conversation"

    def test_two_word_overlap_matches(self):
        """Two shared words should resolve to the best matching state."""
        states = ["resolve_customer_issue", "greet_new_visitor"]
        result = _resolve_state("resolve_customer_complaint", states)
        assert result == "resolve_customer_issue"

    def test_exact_substring_still_matches(self):
        """Exact substring match takes priority over word overlap."""
        states = ["verify_identity", "take_action"]
        result = _resolve_state("verify_identity", states)
        assert result == "verify_identity"


class TestGeneratePatterns:
    """Tests for _generate_patterns() word-length threshold."""

    def test_short_words_excluded_from_standalone_patterns(self):
        """Words under 8 chars should not appear as standalone patterns."""
        patterns = _generate_patterns("greet the customer")
        # "greet" (5 chars) and "customer" (8 chars) are key words
        # Only "customer" should appear as standalone (>= 8 chars)
        standalone = [p for p in patterns if "greet" in p and "customer" not in p]
        assert len(standalone) == 0

    def test_long_words_included_as_standalone(self):
        """Words with 8+ chars should appear as standalone patterns."""
        patterns = _generate_patterns("customer information")
        standalone_customer = [p for p in patterns if "customer" in p and "information" not in p]
        standalone_info = [p for p in patterns if "information" in p and "customer" not in p]
        assert len(standalone_customer) > 0
        assert len(standalone_info) > 0

    def test_phrase_pattern_still_generated(self):
        """Multi-word descriptions should still produce a phrase pattern."""
        patterns = _generate_patterns("greet the user")
        # Should have at least a phrase pattern even if no standalone words qualify
        assert len(patterns) >= 1
