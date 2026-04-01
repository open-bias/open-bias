"""
Tests for rubric registry and built-in rubrics.
"""

import pytest
from openbias.policy.engines.judge.models import (
    EvaluationType,
    EvaluationScope,
    ScoreScale,
)
from openbias.policy.engines.judge.rubrics import RubricRegistry


class TestRubricRegistry:
    def test_builtin_rubrics_registered(self):
        """All built-in rubrics should be available in new instances."""
        registry = RubricRegistry()
        names = registry.list_rubrics()
        assert "safety" in names
        assert "agent_behavior" in names
        assert "conversation_rules" in names

    def test_get_existing_rubric(self):
        registry = RubricRegistry()
        rubric = registry.get("agent_behavior")
        assert rubric is not None
        assert rubric.name == "agent_behavior"
        assert len(rubric.criteria) == 4

    def test_get_nonexistent_rubric(self):
        registry = RubricRegistry()
        assert registry.get("nonexistent") is None

    def test_agent_behavior_rubric(self):
        registry = RubricRegistry()
        rubric = registry.get("agent_behavior")
        assert rubric.evaluation_type == EvaluationType.POINTWISE
        assert rubric.scope == EvaluationScope.TURN
        criteria_names = [c.name for c in rubric.criteria]
        assert "instruction_following" in criteria_names
        assert "tool_use_safety" in criteria_names
        assert "no_hallucination" in criteria_names
        assert "task_completion" in criteria_names

    def test_agent_behavior_safety_criteria_are_binary(self):
        """Safety-critical criteria in agent_behavior should be binary pass/fail."""
        registry = RubricRegistry()
        rubric = registry.get("agent_behavior")
        criteria_map = {c.name: c for c in rubric.criteria}
        # Safety-critical criteria are binary
        assert criteria_map["instruction_following"].scale == ScoreScale.BINARY
        assert criteria_map["instruction_following"].fail_threshold == 0.5
        assert criteria_map["tool_use_safety"].scale == ScoreScale.BINARY
        assert criteria_map["tool_use_safety"].fail_threshold == 0.5
        # Non-critical criteria keep their scale
        assert criteria_map["no_hallucination"].scale == ScoreScale.LIKERT_5
        assert criteria_map["task_completion"].scale == ScoreScale.LIKERT_5

    def test_safety_rubric(self):
        registry = RubricRegistry()
        rubric = registry.get("safety")
        assert rubric.pass_threshold == 0.8
        for criterion in rubric.criteria:
            assert criterion.scale == ScoreScale.BINARY

    def test_conversation_rules_rubric(self):
        registry = RubricRegistry()
        rubric = registry.get("conversation_rules")
        assert rubric.scope == EvaluationScope.CONVERSATION

class TestCreateRulesRubric:
    def test_creates_one_criterion_per_rule(self):
        from openbias.policy.engines.judge.rubrics import create_rules_rubric
        rules = ["No financial advice", "Be professional"]
        rubric = create_rules_rubric(rules)

        assert rubric.name == "inline_rules"
        assert len(rubric.criteria) == 2
        assert rubric.criteria[0].scale == ScoreScale.BINARY
        assert rubric.criteria[1].scale == ScoreScale.BINARY
        # Each criterion has a descriptive name derived from the rule
        assert "rule_1" in rubric.criteria[0].name
        assert "rule_2" in rubric.criteria[1].name
        # Rule text appears in criterion descriptions
        assert "No financial advice" in rubric.criteria[0].description
        assert "Be professional" in rubric.criteria[1].description
        # Instructions reference each criterion
        assert "additional_instructions" in rubric.prompt_overrides
        assert "No financial advice" in rubric.prompt_overrides["additional_instructions"]
        assert "Be professional" in rubric.prompt_overrides["additional_instructions"]

    def test_single_rule_creates_single_criterion(self):
        from openbias.policy.engines.judge.rubrics import create_rules_rubric
        rubric = create_rules_rubric(["Never lie"])
        assert len(rubric.criteria) == 1
        assert rubric.criteria[0].scale == ScoreScale.BINARY

    def test_custom_name(self):
        from openbias.policy.engines.judge.rubrics import create_rules_rubric
        rubric = create_rules_rubric(["rule1"], name="my_policy")
        assert rubric.name == "my_policy"

    def test_three_rules_all_binary(self):
        from openbias.policy.engines.judge.rubrics import create_rules_rubric
        rules = ["Rule A", "Rule B", "Rule C"]
        rubric = create_rules_rubric(rules)
        assert len(rubric.criteria) == 3
        for criterion in rubric.criteria:
            assert criterion.scale == ScoreScale.BINARY

