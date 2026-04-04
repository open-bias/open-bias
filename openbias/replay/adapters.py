"""Adapters that bridge replay traces into existing eval concepts."""

from __future__ import annotations

from openbias.eval import EvalCase, EvalLabels
from openbias.traces import TraceCase


def trace_case_to_eval_case(case: TraceCase) -> EvalCase | None:
    """Convert a trace case carrying eval-style labels into an ``EvalCase``."""

    if not case.labels:
        return None

    labels = EvalLabels(
        violation=bool(case.labels["violation"]),
        detection_scope=case.labels["detection_scope"],
        detect_at_turn=case.labels["detect_at_turn"],
        repair_expected=case.labels.get("repair_expected"),
        repair_verified_at_turn=case.labels.get("repair_verified_at_turn"),
    )

    return EvalCase(
        id=case.id,
        messages=[dict(message) for message in case.messages],
        labels=labels,
        tags=list(case.metadata.tags),
        source=case.source,
    )
