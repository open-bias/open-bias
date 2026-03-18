"""
Multi-judge ensemble aggregation for the Judge Policy Engine.

Runs multiple judge models in parallel and aggregates their verdicts
using configurable strategies. Computes inter-judge agreement rates
and can escalate when judges disagree significantly.
"""

import asyncio
import logging
from typing import Callable, Dict, Any, List, Optional, Tuple

from opensentinel.policy.engines.judge.models import (
    Rubric,
    JudgeScore,
    JudgeVerdict,
    EnsembleVerdict,
    VerdictAction,
    EvaluationScope,
)
from opensentinel.policy.engines.judge.evaluator import JudgeEvaluator

logger = logging.getLogger(__name__)


class AggregationStrategy:
    """Strategies for combining multiple judge verdicts."""

    MEAN_SCORE = "mean_score"
    MAJORITY_VOTE = "majority_vote"
    MEDIAN_SCORE = "median_score"
    CONSERVATIVE = "conservative"
    MINORITY_VETO = "minority_veto"

    ALL = {MEAN_SCORE, MAJORITY_VOTE, MEDIAN_SCORE, CONSERVATIVE, MINORITY_VETO}


class JudgeEnsemble:
    """Aggregates verdicts from multiple judge models.

    Runs all configured judges in parallel via JudgeEvaluator,
    then merges results using a configurable aggregation strategy.
    If inter-judge agreement falls below a threshold, the final
    action is escalated.

    Args:
        evaluator: The JudgeEvaluator that handles individual model calls.
        strategy: Aggregation strategy name (default: "mean_score").
        min_agreement: Minimum agreement rate before escalation (default: 0.6).
    """

    def __init__(
        self,
        evaluator: JudgeEvaluator,
        strategy: str = AggregationStrategy.MEAN_SCORE,
        min_agreement: float = 0.6,
    ) -> None:
        if strategy not in AggregationStrategy.ALL:
            raise ValueError(
                f"Unknown aggregation strategy: {strategy}. "
                f"Must be one of: {sorted(AggregationStrategy.ALL)}"
            )
        self._evaluator = evaluator
        self._strategy = strategy
        self._min_agreement = min_agreement

    @property
    def strategy(self) -> str:
        return self._strategy

    async def evaluate_turn(
        self,
        model_names: List[str],
        rubric: Rubric,
        response_content: str,
        conversation: List[Dict[str, Any]],
        reference: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> EnsembleVerdict:
        """Run turn evaluation across multiple judges and aggregate.

        Args:
            model_names: List of judge model names to use.
            rubric: Rubric to evaluate against.
            response_content: The assistant response to evaluate.
            conversation: Full conversation history.
            reference: Optional reference answer.
            metadata: Optional metadata.
            tool_calls: Optional tool calls from the response.

        Returns:
            EnsembleVerdict with aggregated scores and agreement info.
        """
        verdicts = await self._run_judges(
            model_names,
            lambda name: self._evaluator.evaluate_turn(
                model_name=name,
                rubric=rubric,
                response_content=response_content,
                conversation=conversation,
                reference=reference,
                metadata=metadata,
                session_id=session_id,
                tool_calls=tool_calls,
            ),
        )
        return self._aggregate(verdicts, rubric)

    async def evaluate_conversation(
        self,
        model_names: List[str],
        rubric: Rubric,
        full_conversation: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> EnsembleVerdict:
        """Run conversation evaluation across multiple judges and aggregate."""
        verdicts = await self._run_judges(
            model_names,
            lambda name: self._evaluator.evaluate_conversation(
                model_name=name,
                rubric=rubric,
                full_conversation=full_conversation,
                metadata=metadata,
                session_id=session_id,
            ),
        )
        return self._aggregate(verdicts, rubric)

    async def _run_judges(self, model_names, eval_fn) -> List[JudgeVerdict]:
        """Run evaluation function across all models in parallel.

        Failed models are logged and excluded (fail-open).
        """
        tasks = [eval_fn(name) for name in model_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        verdicts = []
        for name, result in zip(model_names, results):
            if isinstance(result, Exception):
                logger.error(f"Ensemble judge '{name}' failed: {result}")
            else:
                verdicts.append(result)

        return verdicts

    def _aggregate(
        self,
        verdicts: List[JudgeVerdict],
        rubric: Rubric,
    ) -> EnsembleVerdict:
        """Aggregate multiple verdicts into a single EnsembleVerdict."""
        if not verdicts:
            return EnsembleVerdict(
                individual_verdicts=[],
                final_scores=[],
                final_composite=0.0,
                final_action=VerdictAction.ESCALATE,
                agreement_rate=0.0,
                aggregation_strategy=self._strategy,
            )

        if len(verdicts) == 1:
            v = verdicts[0]
            return EnsembleVerdict(
                individual_verdicts=verdicts,
                final_scores=v.scores,
                final_composite=v.composite_score,
                final_action=v.action,
                agreement_rate=1.0,
                aggregation_strategy=self._strategy,
            )

        # Compute agreement
        agreement_rate = self._compute_agreement(verdicts)
        criterion_agreement = self._compute_criterion_agreement(verdicts)

        # Aggregate based on strategy
        if self._strategy == AggregationStrategy.MEAN_SCORE:
            final_scores, final_composite = self._aggregate_mean(verdicts)
        elif self._strategy == AggregationStrategy.MEDIAN_SCORE:
            final_scores, final_composite = self._aggregate_median(verdicts)
        elif self._strategy == AggregationStrategy.MAJORITY_VOTE:
            final_scores, final_composite = self._aggregate_mean(verdicts)
        elif self._strategy == AggregationStrategy.CONSERVATIVE:
            final_scores, final_composite = self._aggregate_conservative(verdicts)
        elif self._strategy == AggregationStrategy.MINORITY_VETO:
            final_scores, final_composite = self._aggregate_mean(verdicts)
        else:
            final_scores, final_composite = self._aggregate_mean(verdicts)

        # Determine final action
        if self._strategy == AggregationStrategy.MAJORITY_VOTE:
            final_action = self._majority_vote_action(verdicts)
        elif self._strategy == AggregationStrategy.CONSERVATIVE:
            final_action = self._most_restrictive_action(verdicts)
        elif self._strategy == AggregationStrategy.MINORITY_VETO:
            final_action = self._most_restrictive_action(verdicts)
        else:
            final_action = self._evaluator._map_action(final_composite, rubric)

        # Escalate if agreement is too low
        if agreement_rate < self._min_agreement:
            logger.warning(
                f"Low inter-judge agreement ({agreement_rate:.2f} < {self._min_agreement}), escalating"
            )
            final_action = VerdictAction.ESCALATE

        return EnsembleVerdict(
            individual_verdicts=verdicts,
            final_scores=final_scores,
            final_composite=final_composite,
            final_action=final_action,
            agreement_rate=agreement_rate,
            criterion_agreement=criterion_agreement,
            aggregation_strategy=self._strategy,
        )

    # =========================================================================
    # Aggregation strategies
    # =========================================================================

    def _aggregate_mean(
        self, verdicts: List[JudgeVerdict]
    ) -> Tuple[List[JudgeScore], float]:
        """Average scores across judges."""
        final_composite = sum(v.composite_score for v in verdicts) / len(verdicts)
        final_scores = self._merge_scores_mean(verdicts)
        return final_scores, final_composite

    def _aggregate_median(
        self, verdicts: List[JudgeVerdict]
    ) -> Tuple[List[JudgeScore], float]:
        """Median scores across judges."""
        composites = sorted(v.composite_score for v in verdicts)
        n = len(composites)
        if n % 2 == 1:
            final_composite = composites[n // 2]
        else:
            final_composite = (composites[n // 2 - 1] + composites[n // 2]) / 2
        final_scores = self._merge_scores_median(verdicts)
        return final_scores, final_composite

    def _aggregate_conservative(
        self, verdicts: List[JudgeVerdict]
    ) -> Tuple[List[JudgeScore], float]:
        """Take the lowest (most conservative) scores."""
        final_composite = min(v.composite_score for v in verdicts)
        final_scores = self._merge_scores_min(verdicts)
        return final_scores, final_composite

    # =========================================================================
    # Score merging helpers
    # =========================================================================

    def _merge_scores_mean(self, verdicts: List[JudgeVerdict]) -> List[JudgeScore]:
        """Average scores per criterion across verdicts."""
        return self._merge_scores(verdicts, _agg_mean)

    def _merge_scores_median(self, verdicts: List[JudgeVerdict]) -> List[JudgeScore]:
        """Median scores per criterion across verdicts."""
        return self._merge_scores(verdicts, _agg_median)

    def _merge_scores_min(self, verdicts: List[JudgeVerdict]) -> List[JudgeScore]:
        """Minimum scores per criterion across verdicts."""
        return self._merge_scores(verdicts, _agg_min)

    def _merge_scores(
        self,
        verdicts: List[JudgeVerdict],
        agg_fn: Callable[[List[float]], float],
    ) -> List[JudgeScore]:
        """Merge per-criterion scores using an aggregation function."""
        # Group scores by criterion
        by_criterion: Dict[str, List[JudgeScore]] = {}
        for verdict in verdicts:
            for score in verdict.scores:
                by_criterion.setdefault(score.criterion, []).append(score)

        merged = []
        for criterion, scores in by_criterion.items():
            raw_scores = [s.score for s in scores]
            max_score = scores[0].max_score
            if not all(s.max_score == max_score for s in scores):
                logger.warning(
                    f"Criterion '{criterion}' has mismatched max_score across judges, "
                    f"using first judge's value ({max_score})"
                )
            confidences = [s.confidence for s in scores]

            merged.append(JudgeScore(
                criterion=criterion,
                score=round(agg_fn(raw_scores), 2),
                max_score=max_score,
                reasoning=f"Aggregated from {len(scores)} judges ({self._strategy})",
                evidence=[],
                confidence=_agg_mean(confidences),
            ))

        return merged

    # =========================================================================
    # Action voting
    # =========================================================================

    def _majority_vote_action(self, verdicts: List[JudgeVerdict]) -> VerdictAction:
        """Pick the action that the majority of judges agree on."""
        if not verdicts:
            return VerdictAction.ESCALATE

        counts: Dict[VerdictAction, int] = {}
        for v in verdicts:
            counts[v.action] = counts.get(v.action, 0) + 1

        return max(counts, key=counts.get)

    def _most_restrictive_action(self, verdicts: List[JudgeVerdict]) -> VerdictAction:
        """Pick the most restrictive action across all judges."""
        priority = {
            VerdictAction.PASS: 0,
            VerdictAction.WARN: 1,
            VerdictAction.ESCALATE: 2,
            VerdictAction.INTERVENE: 3,
            VerdictAction.BLOCK: 4,
        }
        return max(verdicts, key=lambda v: priority.get(v.action, 0)).action

    # =========================================================================
    # Agreement computation
    # =========================================================================

    def _compute_agreement(self, verdicts: List[JudgeVerdict]) -> float:
        """Compute overall inter-judge agreement rate.

        Agreement = fraction of judge pairs that agree on the action.
        """
        if len(verdicts) < 2:
            return 1.0

        actions = [v.action for v in verdicts]
        n = len(actions)
        agreements = 0
        total_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                if actions[i] == actions[j]:
                    agreements += 1

        return agreements / total_pairs if total_pairs > 0 else 1.0

    def _compute_criterion_agreement(
        self, verdicts: List[JudgeVerdict]
    ) -> Dict[str, float]:
        """Compute per-criterion agreement (score variance).

        Lower variance = higher agreement. Returns 1 - normalized_variance
        so that 1.0 = perfect agreement.
        """
        # Group normalized scores by criterion
        by_criterion: Dict[str, List[float]] = {}
        for verdict in verdicts:
            for score in verdict.scores:
                by_criterion.setdefault(score.criterion, []).append(score.normalized)

        result = {}
        for criterion, values in by_criterion.items():
            if len(values) < 2:
                result[criterion] = 1.0
                continue
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            # Normalized variance (max possible variance for [0,1] range is 0.25)
            result[criterion] = max(0.0, 1.0 - variance / 0.25)

        return result


# =========================================================================
# Module-level aggregation helpers
# =========================================================================

def _agg_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _agg_median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _agg_min(values: List[float]) -> float:
    return min(values) if values else 0.0
