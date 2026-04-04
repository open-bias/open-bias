"""Schema validation tests for replayable trace datasets."""

from __future__ import annotations

import pytest

from openbias.traces import (
    TraceCase,
    TraceDataset,
    TraceEvaluatorSummary,
    TraceMetadata,
    TraceValidationError,
)


def test_trace_case_accepts_single_request_response_pair():
    case = TraceCase(
        id="trace-1",
        session_id="sess-1",
        messages=[
            {"role": "system", "content": "Be safe."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
        metadata=TraceMetadata(model="gpt-4o-mini", final_action="allow"),
        evaluator_summaries=(
            TraceEvaluatorSummary(
                name="content-policy",
                engine_type="judge",
                phase="post_call",
                decision="allow",
            ),
        ),
    )

    dataset = TraceDataset(name="smoke", cases=[case])
    assert dataset.cases[0].id == "trace-1"


def test_trace_case_requires_exactly_one_assistant_message():
    with pytest.raises(TraceValidationError, match="exactly one assistant response"):
        TraceCase(
            id="bad-trace",
            session_id="sess-1",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "assistant", "content": "Again"},
            ],
        )


def test_trace_case_requires_assistant_as_final_message():
    with pytest.raises(TraceValidationError, match="assistant response must be the final message"):
        TraceCase(
            id="bad-order",
            session_id="sess-1",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "tool", "content": "later"},
            ],
        )


def test_trace_evaluator_summary_rejects_unknown_decision():
    with pytest.raises(TraceValidationError, match="Unsupported evaluator decision"):
        TraceEvaluatorSummary(
            name="content-policy",
            engine_type="judge",
            phase="post_call",
            decision="maybe",
        )


def test_trace_dataset_rejects_duplicate_case_ids():
    case = TraceCase(
        id="dup",
        session_id="sess-1",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    )

    with pytest.raises(TraceValidationError, match="Duplicate trace case id"):
        TraceDataset(name="dups", cases=[case, case])
