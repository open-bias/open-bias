"""
Tests for the core judge evaluator.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from opensentinel.policy.engines.judge.evaluator import JudgeEvaluator
from opensentinel.policy.engines.judge.client import JudgeClient
from opensentinel.policy.engines.judge.models import (
    VerdictAction,
    EvaluationScope,
    ScoreScale,
    RubricCriterion,
    Rubric,
    EvaluationType,
)


@pytest.fixture
def mock_client():
    """Create a JudgeClient with a mocked call_judge method."""
    client = JudgeClient()
    client.add_model("primary", "gpt-4o-mini")
    client.call_judge = AsyncMock()
    client.get_model_id = MagicMock(return_value="gpt-4o-mini")
    client.get_tokens_for_model = MagicMock(return_value=100)
    return client


@pytest.fixture
def evaluator(mock_client):
    return JudgeEvaluator(client=mock_client)


@pytest.fixture
def simple_rubric():
    return Rubric(
        name="test_rubric",
        description="Test rubric",
        criteria=[
            RubricCriterion(
                name="quality",
                description="Response quality",
                scale=ScoreScale.LIKERT_5,
                weight=1.0,
            ),
            RubricCriterion(
                name="safety",
                description="Response safety",
                scale=ScoreScale.LIKERT_5,
                weight=1.0,
            ),
        ],
        pass_threshold=0.6,
    )


@pytest.fixture
def conversation():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]


class TestEvaluateTurn:
    async def test_passing_verdict(self, evaluator, mock_client, simple_rubric, conversation):
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "quality", "score": 5, "reasoning": "Good", "evidence": ["clear"], "confidence": 0.9},
                {"criterion": "safety", "score": 5, "reasoning": "Safe", "evidence": [], "confidence": 0.95},
            ],
            "summary": "Good response",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=simple_rubric,
            response_content="Python is a programming language.",
            conversation=conversation,
        )

        assert verdict.action == VerdictAction.PASS
        assert verdict.composite_score == 1.0
        assert verdict.scope == EvaluationScope.TURN
        assert verdict.judge_model == "gpt-4o-mini"
        assert len(verdict.scores) == 2

    async def test_failing_verdict(self, evaluator, mock_client, simple_rubric, conversation):
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "quality", "score": 1, "reasoning": "Bad", "evidence": [], "confidence": 0.9},
                {"criterion": "safety", "score": 1, "reasoning": "Unsafe", "evidence": [], "confidence": 0.9},
            ],
            "summary": "Poor response",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=simple_rubric,
            response_content="bad response",
            conversation=conversation,
            fail_action=VerdictAction.INTERVENE,
        )

        assert verdict.action == VerdictAction.INTERVENE
        assert verdict.composite_score == 0.0

    async def test_below_threshold_verdict(self, evaluator, mock_client, simple_rubric, conversation):
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "quality", "score": 3, "reasoning": "OK", "evidence": [], "confidence": 0.8},
                {"criterion": "safety", "score": 3, "reasoning": "OK", "evidence": [], "confidence": 0.8},
            ],
            "summary": "Average response",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=simple_rubric,
            response_content="ok response",
            conversation=conversation,
            fail_action=VerdictAction.INTERVENE,
        )

        # 3/5 -> normalized 0.5, below pass_threshold 0.6 -> fail_action = INTERVENE
        assert verdict.action == VerdictAction.INTERVENE

    async def test_missing_criterion_filled(self, evaluator, mock_client, simple_rubric, conversation):
        """Missing criteria should be filled with min score but excluded from composite."""
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "quality", "score": 5, "reasoning": "Good", "evidence": [], "confidence": 0.9},
                # "safety" is missing
            ],
            "summary": "Partial response",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=simple_rubric,
            response_content="test",
            conversation=conversation,
        )

        assert len(verdict.scores) == 2
        safety_score = next(s for s in verdict.scores if s.criterion == "safety")
        assert safety_score.score == 1  # min_score for LIKERT_5
        assert safety_score.confidence == 0.0
        # Synthetic fill should not drag down composite; only quality (1.0) contributes
        assert verdict.composite_score == 1.0
        assert verdict.action == VerdictAction.PASS


class TestEvaluateConversation:
    async def test_conversation_scope(self, evaluator, mock_client, conversation):
        conv_rubric = Rubric(
            name="conv_test",
            description="Test",
            criteria=[
                RubricCriterion(name="consistency", description="Consistent?", scale=ScoreScale.LIKERT_5),
            ],
            scope=EvaluationType.POINTWISE,
            pass_threshold=0.6,
        )

        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "consistency", "score": 4, "reasoning": "Good", "evidence": ["turn 1"], "confidence": 0.9},
            ],
            "summary": "Consistent conversation",
        }

        verdict = await evaluator.evaluate_conversation(
            model_name="primary",
            rubric=conv_rubric,
            full_conversation=conversation,
        )

        assert verdict.scope == EvaluationScope.CONVERSATION
        assert verdict.action == VerdictAction.PASS


class TestEvaluatePairwise:
    async def test_pairwise_evaluation(self, evaluator, mock_client, conversation):
        pairwise_rubric = Rubric(
            name="pair_test",
            description="Test",
            criteria=[
                RubricCriterion(name="overall_preference", description="Which is better?", scale=ScoreScale.LIKERT_5),
            ],
            evaluation_type=EvaluationType.PAIRWISE,
            pass_threshold=0.5,
        )

        mock_client.call_judge.return_value = {
            "scores": [
                {
                    "criterion": "overall_preference",
                    "score_a": 4, "score_b": 2,
                    "winner": "a",
                    "reasoning": "A is better",
                    "evidence": ["clearer"],
                    "confidence": 0.85,
                },
            ],
            "overall_winner": "a",
            "summary": "Response A is better",
        }

        verdict = await evaluator.evaluate_pairwise(
            model_name="primary",
            rubric=pairwise_rubric,
            response_a="Good answer",
            response_b="Bad answer",
            conversation=conversation,
        )

        assert verdict.metadata.get("pairwise") is True
        assert len(verdict.scores) == 1


class TestCompositeScoring:
    def test_weighted_composite(self, evaluator):
        from opensentinel.policy.engines.judge.models import JudgeScore

        scores = [
            JudgeScore(criterion="a", score=5, max_score=5, reasoning="ok"),  # normalized 1.0
            JudgeScore(criterion="b", score=1, max_score=5, reasoning="ok"),  # normalized 0.0
        ]
        criteria = [
            RubricCriterion(name="a", description="", weight=1.0),
            RubricCriterion(name="b", description="", weight=1.0),
        ]
        composite = evaluator._compute_composite(scores, criteria)
        assert composite == 0.5

    def test_weighted_composite_unequal_weights(self, evaluator):
        from opensentinel.policy.engines.judge.models import JudgeScore

        scores = [
            JudgeScore(criterion="a", score=5, max_score=5, reasoning="ok"),  # normalized 1.0
            JudgeScore(criterion="b", score=1, max_score=5, reasoning="ok"),  # normalized 0.0
        ]
        criteria = [
            RubricCriterion(name="a", description="", weight=3.0),
            RubricCriterion(name="b", description="", weight=1.0),
        ]
        composite = evaluator._compute_composite(scores, criteria)
        assert composite == 0.75

    def test_empty_scores(self, evaluator):
        assert evaluator._compute_composite([], []) == 0.0

    def test_zero_confidence_fills_excluded(self, evaluator):
        """Synthetic fills (confidence=0.0) should not drag down composite score."""
        from opensentinel.policy.engines.judge.models import JudgeScore

        scores = [
            JudgeScore(criterion="a", score=5, max_score=5, reasoning="ok", confidence=0.9),  # normalized 1.0
            JudgeScore(criterion="b", score=5, max_score=5, reasoning="ok", confidence=0.9),  # normalized 1.0
            JudgeScore(criterion="c", score=1, max_score=5, reasoning="fill", confidence=0.0),  # synthetic fill
            JudgeScore(criterion="d", score=1, max_score=5, reasoning="fill", confidence=0.0),  # synthetic fill
        ]
        criteria = [
            RubricCriterion(name="a", description="", weight=1.0),
            RubricCriterion(name="b", description="", weight=1.0),
            RubricCriterion(name="c", description="", weight=1.0),
            RubricCriterion(name="d", description="", weight=1.0),
        ]
        # Only a and b should contribute; composite should be 1.0, not 0.5
        composite = evaluator._compute_composite(scores, criteria)
        assert composite == 1.0

    def test_all_zero_confidence_returns_zero(self, evaluator):
        """If all scores have confidence=0.0, composite should be 0.0 (not division by zero)."""
        from opensentinel.policy.engines.judge.models import JudgeScore

        scores = [
            JudgeScore(criterion="a", score=1, max_score=5, reasoning="fill", confidence=0.0),
            JudgeScore(criterion="b", score=1, max_score=5, reasoning="fill", confidence=0.0),
        ]
        criteria = [
            RubricCriterion(name="a", description="", weight=1.0),
            RubricCriterion(name="b", description="", weight=1.0),
        ]
        assert evaluator._compute_composite(scores, criteria) == 0.0


class TestActionMapping:
    def test_pass(self, evaluator, simple_rubric):
        assert evaluator._map_action(0.8, simple_rubric, VerdictAction.INTERVENE) == VerdictAction.PASS

    def test_fail(self, evaluator, simple_rubric):
        assert evaluator._map_action(0.5, simple_rubric, VerdictAction.INTERVENE) == VerdictAction.INTERVENE


class TestBinarySafetyCriteria:
    """Tests for binary pass/fail decision logic on safety-critical criteria."""

    @pytest.fixture
    def agent_behavior_rubric(self):
        """Mirrors the built-in agent_behavior rubric with binary safety criteria."""
        return Rubric(
            name="agent_behavior",
            description="Agent behavior evaluation",
            criteria=[
                RubricCriterion(
                    name="instruction_following",
                    description="Follows instructions?",
                    scale=ScoreScale.BINARY,
                    weight=1.0,
                    fail_threshold=0.5,
                ),
                RubricCriterion(
                    name="tool_use_safety",
                    description="Safe tool use?",
                    scale=ScoreScale.BINARY,
                    weight=1.0,
                    fail_threshold=0.5,
                ),
                RubricCriterion(
                    name="no_hallucination",
                    description="No hallucination?",
                    scale=ScoreScale.LIKERT_5,
                    weight=1.0,
                ),
                RubricCriterion(
                    name="task_completion",
                    description="Task progress?",
                    scale=ScoreScale.LIKERT_5,
                    weight=0.8,
                ),
            ],
            pass_threshold=0.6,
        )

    async def test_tool_use_safety_fail_triggers_intervene(
        self, evaluator, mock_client, agent_behavior_rubric, conversation,
    ):
        """A single tool_use_safety failure should trigger INTERVENE regardless of other scores."""
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "instruction_following", "score": 1, "reasoning": "Good", "evidence": [], "confidence": 0.9},
                {"criterion": "tool_use_safety", "score": 0, "reasoning": "Called dangerous API", "evidence": ["delete_all()"], "confidence": 0.95},
                {"criterion": "no_hallucination", "score": 5, "reasoning": "Grounded", "evidence": [], "confidence": 0.9},
                {"criterion": "task_completion", "score": 5, "reasoning": "Done", "evidence": [], "confidence": 0.9},
            ],
            "summary": "Dangerous tool use detected",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=agent_behavior_rubric,
            response_content="I deleted all records.",
            conversation=conversation,
            fail_action=VerdictAction.INTERVENE,
        )

        assert verdict.action == VerdictAction.INTERVENE
        assert "tool_use_safety" in verdict.metadata.get("criterion_failures", [])

    async def test_all_criteria_pass(
        self, evaluator, mock_client, agent_behavior_rubric, conversation,
    ):
        """All criteria passing should result in PASS."""
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "instruction_following", "score": 1, "reasoning": "Good", "evidence": [], "confidence": 0.9},
                {"criterion": "tool_use_safety", "score": 1, "reasoning": "Safe", "evidence": [], "confidence": 0.9},
                {"criterion": "no_hallucination", "score": 5, "reasoning": "Grounded", "evidence": [], "confidence": 0.9},
                {"criterion": "task_completion", "score": 5, "reasoning": "Done", "evidence": [], "confidence": 0.9},
            ],
            "summary": "All good",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=agent_behavior_rubric,
            response_content="Good response.",
            conversation=conversation,
        )

        assert verdict.action == VerdictAction.PASS
        assert verdict.metadata.get("criterion_failures") is None

    async def test_low_noncritical_score_still_passes(
        self, evaluator, mock_client, agent_behavior_rubric, conversation,
    ):
        """Low scores on non-critical criteria should not trigger intervention
        if all safety-critical criteria pass."""
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "instruction_following", "score": 1, "reasoning": "Good", "evidence": [], "confidence": 0.9},
                {"criterion": "tool_use_safety", "score": 1, "reasoning": "Safe", "evidence": [], "confidence": 0.9},
                {"criterion": "no_hallucination", "score": 4, "reasoning": "OK", "evidence": [], "confidence": 0.8},
                {"criterion": "task_completion", "score": 4, "reasoning": "Progressing", "evidence": [], "confidence": 0.8},
            ],
            "summary": "Acceptable",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=agent_behavior_rubric,
            response_content="Decent response.",
            conversation=conversation,
        )

        assert verdict.action == VerdictAction.PASS

    async def test_corrective_actions_parsed_from_judge_response(
        self, evaluator, mock_client, agent_behavior_rubric, conversation,
    ):
        """corrective_actions from judge response should be parsed into JudgeScore."""
        mock_client.call_judge.return_value = {
            "scores": [
                {
                    "criterion": "instruction_following", "score": 1,
                    "reasoning": "Good", "evidence": [], "confidence": 0.9,
                },
                {
                    "criterion": "tool_use_safety", "score": 0,
                    "reasoning": "Called delete_database()",
                    "evidence": ["delete_database(table='users')"],
                    "confidence": 0.95,
                    "corrective_actions": "Ask for user confirmation before destructive ops.",
                },
                {
                    "criterion": "no_hallucination", "score": 5,
                    "reasoning": "Grounded", "evidence": [], "confidence": 0.9,
                },
                {
                    "criterion": "task_completion", "score": 5,
                    "reasoning": "Done", "evidence": [], "confidence": 0.9,
                },
            ],
            "summary": "Dangerous tool use",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=agent_behavior_rubric,
            response_content="I deleted the database.",
            conversation=conversation,
        )

        tool_score = next(s for s in verdict.scores if s.criterion == "tool_use_safety")
        assert tool_score.corrective_actions == "Ask for user confirmation before destructive ops."

        # Other scores should have None corrective_actions
        inst_score = next(s for s in verdict.scores if s.criterion == "instruction_following")
        assert inst_score.corrective_actions is None

    async def test_instruction_following_fail_triggers_intervene(
        self, evaluator, mock_client, agent_behavior_rubric, conversation,
    ):
        """instruction_following failure should also trigger INTERVENE."""
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "instruction_following", "score": 0, "reasoning": "Ignored task", "evidence": ["off-topic"], "confidence": 0.9},
                {"criterion": "tool_use_safety", "score": 1, "reasoning": "Safe", "evidence": [], "confidence": 0.9},
                {"criterion": "no_hallucination", "score": 5, "reasoning": "Grounded", "evidence": [], "confidence": 0.9},
                {"criterion": "task_completion", "score": 5, "reasoning": "Done", "evidence": [], "confidence": 0.9},
            ],
            "summary": "Instructions violated",
        }

        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=agent_behavior_rubric,
            response_content="Ignoring your request.",
            conversation=conversation,
            fail_action=VerdictAction.INTERVENE,
        )

        assert verdict.action == VerdictAction.INTERVENE
        assert "instruction_following" in verdict.metadata.get("criterion_failures", [])


class TestBinaryEvaluationPath:
    """Tests for the binary evaluation path (all-binary rubrics skip thresholds)."""

    @pytest.fixture
    def binary_rubric(self) -> Rubric:
        return Rubric(
            name="binary_test",
            description="All-binary rubric",
            criteria=[
                RubricCriterion(
                    name="rule_a",
                    description="Rule A",
                    scale=ScoreScale.BINARY,
                    weight=1.0,
                ),
                RubricCriterion(
                    name="rule_b",
                    description="Rule B",
                    scale=ScoreScale.BINARY,
                    weight=1.0,
                ),
            ],
        )

    async def test_all_pass_returns_pass(
        self, evaluator: JudgeEvaluator, mock_client: JudgeClient,
        binary_rubric: Rubric, conversation: list,
    ) -> None:
        """All binary criteria scoring 1 -> PASS with composite 1.0."""
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "rule_a", "score": 1, "reasoning": "OK", "evidence": [], "confidence": 0.9},
                {"criterion": "rule_b", "score": 1, "reasoning": "OK", "evidence": [], "confidence": 0.9},
            ],
            "summary": "All good",
        }
        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=binary_rubric,
            response_content="Good response.",
            conversation=conversation,
        )
        assert verdict.action == VerdictAction.PASS
        assert verdict.composite_score == 1.0
        assert not verdict.metadata.get("criterion_failures")

    async def test_one_fail_returns_fail_action(
        self, evaluator: JudgeEvaluator, mock_client: JudgeClient,
        binary_rubric: Rubric, conversation: list,
    ) -> None:
        """Any binary criterion scoring 0 -> fail_action applied, composite 0.0."""
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "rule_a", "score": 0, "reasoning": "Violated", "evidence": [], "confidence": 0.9},
                {"criterion": "rule_b", "score": 1, "reasoning": "OK", "evidence": [], "confidence": 0.9},
            ],
            "summary": "Violation",
        }
        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=binary_rubric,
            response_content="Bad response.",
            conversation=conversation,
            fail_action=VerdictAction.INTERVENE,
        )
        assert verdict.action == VerdictAction.INTERVENE
        assert verdict.composite_score == 0.0
        assert "rule_a" in verdict.metadata.get("criterion_failures", [])

    async def test_binary_path_default_fail_action_is_intervene(
        self, evaluator: JudgeEvaluator, mock_client: JudgeClient,
        binary_rubric: Rubric, conversation: list,
    ) -> None:
        """Default fail_action (INTERVENE) is used when not explicitly passed."""
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "rule_a", "score": 0, "reasoning": "Bad", "evidence": [], "confidence": 0.9},
                {"criterion": "rule_b", "score": 0, "reasoning": "Bad", "evidence": [], "confidence": 0.9},
            ],
            "summary": "All bad",
        }
        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=binary_rubric,
            response_content="Terrible.",
            conversation=conversation,
        )
        assert verdict.action == VerdictAction.INTERVENE
        assert verdict.composite_score == 0.0

    async def test_synthetic_fill_ignored_in_binary_path(
        self, evaluator: JudgeEvaluator, mock_client: JudgeClient,
        binary_rubric: Rubric, conversation: list,
    ) -> None:
        """Synthetic fills (confidence=0.0) should not count as failures in binary path."""
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "rule_a", "score": 1, "reasoning": "OK", "evidence": [], "confidence": 0.9},
                # rule_b omitted — will be filled with score=0, confidence=0.0
            ],
            "summary": "Partial",
        }
        verdict = await evaluator.evaluate_turn(
            model_name="primary",
            rubric=binary_rubric,
            response_content="OK response.",
            conversation=conversation,
        )
        # Synthetic fill has confidence=0.0, should not trigger failure
        assert verdict.action == VerdictAction.PASS
        assert verdict.composite_score == 1.0


class TestEvaluateWithReference:
    """Tests for _evaluate_with_reference — reference_alignment criterion handling."""

    @pytest.fixture
    def ref_rubric(self):
        return Rubric(
            name="ref_rubric",
            description="Reference rubric",
            criteria=[
                RubricCriterion(
                    name="accuracy",
                    description="Factual accuracy",
                    scale=ScoreScale.LIKERT_5,
                    weight=1.0,
                ),
            ],
            pass_threshold=0.6,
        )

    async def test_reference_alignment_score_is_parsed(
        self, evaluator, mock_client, ref_rubric, conversation,
    ):
        """reference_alignment score from judge should be included in verdict scores."""
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "accuracy", "score": 4, "reasoning": "Mostly correct", "evidence": [], "confidence": 0.9},
                {"criterion": "reference_alignment", "score": 5, "reasoning": "Matches reference", "evidence": [], "confidence": 0.85},
            ],
            "summary": "Good alignment",
        }

        verdict = await evaluator._evaluate_with_reference(
            model_name="primary",
            rubric=ref_rubric,
            response_content="Python is a high-level language.",
            conversation=conversation,
            reference="Python is a high-level, interpreted programming language.",
        )

        criterion_names = {s.criterion for s in verdict.scores}
        assert "reference_alignment" in criterion_names
        ref_score = next(s for s in verdict.scores if s.criterion == "reference_alignment")
        assert ref_score.score == 5

    async def test_reference_alignment_does_not_affect_verdict(
        self, evaluator, mock_client, ref_rubric, conversation,
    ):
        """A low reference_alignment score should not affect pass/fail since it's not in rubric criteria."""
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "accuracy", "score": 5, "reasoning": "Correct", "evidence": [], "confidence": 0.95},
                {"criterion": "reference_alignment", "score": 1, "reasoning": "Poor alignment", "evidence": [], "confidence": 0.8},
            ],
            "summary": "Accurate but misaligned",
        }

        verdict = await evaluator._evaluate_with_reference(
            model_name="primary",
            rubric=ref_rubric,
            response_content="Python is a language.",
            conversation=conversation,
            reference="Python is a high-level, interpreted programming language.",
        )

        # accuracy=5 -> normalized 1.0, above pass_threshold 0.6 -> PASS
        assert verdict.action == VerdictAction.PASS

    async def test_reference_alignment_absent_does_not_break_parsing(
        self, evaluator, mock_client, ref_rubric, conversation,
    ):
        """If judge omits reference_alignment, parsing should still succeed."""
        mock_client.call_judge.return_value = {
            "scores": [
                {"criterion": "accuracy", "score": 3, "reasoning": "OK", "evidence": [], "confidence": 0.8},
            ],
            "summary": "Average",
        }

        verdict = await evaluator._evaluate_with_reference(
            model_name="primary",
            rubric=ref_rubric,
            response_content="Python is a language.",
            conversation=conversation,
            reference="Python is a high-level, interpreted programming language.",
        )

        # Should still produce a verdict without error
        assert verdict is not None
