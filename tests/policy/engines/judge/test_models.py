"""
Tests for judge engine data models.
"""

import pytest
from openbias.policy.engines.judge.models import (
    EvaluationType,
    EvaluationScope,
    ScoreScale,
    VerdictAction,
    RubricCriterion,
    Rubric,
    JudgeScore,
    JudgeVerdict,
    JudgeSessionContext,
)


class TestScoreScale:
    def test_max_score_values(self):
        assert ScoreScale.BINARY.max_score == 1
        assert ScoreScale.LIKERT_3.max_score == 3
        assert ScoreScale.LIKERT_5.max_score == 5

    def test_min_score_values(self):
        assert ScoreScale.BINARY.min_score == 0
        assert ScoreScale.LIKERT_5.min_score == 1


class TestJudgeScore:
    def test_normalized_binary(self):
        score = JudgeScore(criterion="test", score=1, max_score=1, reasoning="ok")
        assert score.normalized == 1.0

        score = JudgeScore(criterion="test", score=0, max_score=1, reasoning="fail")
        assert score.normalized == 0.0

    def test_normalized_likert(self):
        # 5/5 -> 1.0
        score = JudgeScore(criterion="test", score=5, max_score=5, reasoning="ok")
        assert score.normalized == 1.0

        # 1/5 -> 0.0
        score = JudgeScore(criterion="test", score=1, max_score=5, reasoning="ok")
        assert score.normalized == 0.0

        # 3/5 -> 0.5
        score = JudgeScore(criterion="test", score=3, max_score=5, reasoning="ok")
        assert score.normalized == 0.5

    def test_normalized_zero_max(self):
        score = JudgeScore(criterion="test", score=0, max_score=0, reasoning="ok")
        assert score.normalized == 0.0


class TestJudgeVerdict:
    def test_to_dict(self):
        verdict = JudgeVerdict(
            scores=[
                JudgeScore(criterion="c1", score=4, max_score=5, reasoning="good"),
            ],
            composite_score=0.75,
            action=VerdictAction.PASS,
            summary="Looks good",
            judge_model="gpt-4o-mini",
            latency_ms=500.0,
        )
        d = verdict.to_dict()
        assert d["composite_score"] == 0.75
        assert d["action"] == "pass"
        assert d["judge_model"] == "gpt-4o-mini"
        assert len(d["scores"]) == 1
        assert d["scores"][0]["criterion"] == "c1"
        assert "normalized" in d["scores"][0]


class TestRubric:
    def test_defaults(self):
        rubric = Rubric(
            name="test",
            description="test rubric",
            criteria=[RubricCriterion(name="c1", description="criterion 1")],
        )
        assert rubric.evaluation_type == EvaluationType.POINTWISE
        assert rubric.scope == EvaluationScope.TURN
        assert rubric.pass_threshold == 0.6
        assert rubric.prompt_overrides == {}


class TestJudgeSessionContext:
    def test_record_verdict(self):
        session = JudgeSessionContext(session_id="s1")
        assert session.turn_count == 0

        verdict = JudgeVerdict(
            scores=[],
            composite_score=0.8,
            action=VerdictAction.PASS,
            summary="ok",
            judge_model="test",
            token_usage=100,
        )
        session.record_verdict(verdict)

        # turn_count is managed by the engine, not record_verdict
        assert session.turn_count == 0
        assert session.score_trend == [0.8]
        assert session.total_tokens_used == 100
        assert len(session.evaluation_history) == 1

    def test_record_violation_counts(self):
        session = JudgeSessionContext(session_id="s1")

        intervene_verdict = JudgeVerdict(
            scores=[], composite_score=0.3, action=VerdictAction.INTERVENE,
            summary="intervention", judge_model="test",
        )
        session.record_verdict(intervene_verdict)
        session.record_verdict(intervene_verdict)

        assert session.violation_counts["intervene"] == 2

    def test_to_dict(self):
        session = JudgeSessionContext(session_id="s1")
        d = session.to_dict()
        assert d["session_id"] == "s1"
        assert d["turn_count"] == 0
        assert "created_at" in d

    def test_last_intervention_criteria_accumulates_across_verdicts(self):
        """Criteria from multiple non-PASS verdicts in one cycle are unioned, not overwritten."""
        session = JudgeSessionContext(session_id="s1")

        turn_verdict = JudgeVerdict(
            scores=[],
            composite_score=0.3,
            action=VerdictAction.INTERVENE,
            summary="turn fail",
            judge_model="test",
            metadata={"criterion_failures": ["tool_use_safety", "instruction_following"]},
        )
        conv_verdict = JudgeVerdict(
            scores=[],
            composite_score=0.4,
            action=VerdictAction.INTERVENE,
            summary="conv fail",
            judge_model="test",
            metadata={"criterion_failures": ["scope_creep"]},
        )

        session.record_verdict(turn_verdict)
        session.record_verdict(conv_verdict)

        assert set(session.last_intervention_criteria) == {
            "tool_use_safety",
            "instruction_following",
            "scope_creep",
        }

    def test_last_intervention_criteria_no_duplicates(self):
        """Shared criteria between verdicts appear only once in last_intervention_criteria."""
        session = JudgeSessionContext(session_id="s1")

        v1 = JudgeVerdict(
            scores=[],
            composite_score=0.3,
            action=VerdictAction.INTERVENE,
            summary="fail",
            judge_model="test",
            metadata={"criterion_failures": ["tool_use_safety"]},
        )
        v2 = JudgeVerdict(
            scores=[],
            composite_score=0.3,
            action=VerdictAction.INTERVENE,
            summary="fail",
            judge_model="test",
            metadata={"criterion_failures": ["tool_use_safety", "scope_creep"]},
        )

        session.record_verdict(v1)
        session.record_verdict(v2)

        assert session.last_intervention_criteria.count("tool_use_safety") == 1
        assert set(session.last_intervention_criteria) == {"tool_use_safety", "scope_creep"}

    def test_last_intervention_criteria_reset_on_new_cycle(self):
        """Resetting last_intervention_criteria before a cycle clears prior-cycle data."""
        session = JudgeSessionContext(session_id="s1")

        v1 = JudgeVerdict(
            scores=[],
            composite_score=0.3,
            action=VerdictAction.INTERVENE,
            summary="fail",
            judge_model="test",
            metadata={"criterion_failures": ["old_criterion"]},
        )
        session.record_verdict(v1)
        assert "old_criterion" in session.last_intervention_criteria

        # Simulate what engine.py does before recording the next cycle's verdicts
        session.last_intervention_criteria = []

        v2 = JudgeVerdict(
            scores=[],
            composite_score=0.3,
            action=VerdictAction.INTERVENE,
            summary="fail",
            judge_model="test",
            metadata={"criterion_failures": ["new_criterion"]},
        )
        session.record_verdict(v2)

        assert session.last_intervention_criteria == ["new_criterion"]
        assert "old_criterion" not in session.last_intervention_criteria
