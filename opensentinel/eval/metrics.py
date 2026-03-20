"""Aggregate metrics helpers for eval results."""

from __future__ import annotations

from dataclasses import dataclass, field

from opensentinel.eval.runner import EvalResult
from opensentinel.policy.protocols import Decision


@dataclass
class EvalMetrics:
    """Aggregate metrics across eval results."""

    total_turns: int = 0
    decisions: dict[str, int] = field(default_factory=dict)
    violation_count: int = 0
    intervention_count: int = 0


def compute_metrics(results: list[EvalResult]) -> EvalMetrics:
    """Compute aggregate metrics from a list of eval results."""
    metrics = EvalMetrics()

    for result in results:
        for turn in result.turns:
            metrics.total_turns += 1

            for eval_result in (turn.request_eval, turn.response_eval):
                decision_name = eval_result.decision.value
                metrics.decisions[decision_name] = metrics.decisions.get(decision_name, 0) + 1

                metrics.violation_count += len(
                    eval_result.metadata.get("violations", [])
                )

                if eval_result.decision in (Decision.INTERVENE, Decision.BLOCK):
                    metrics.intervention_count += 1

    return metrics
