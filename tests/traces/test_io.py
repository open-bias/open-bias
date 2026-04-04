"""JSONL load/save tests for replayable trace datasets."""

from __future__ import annotations

from pathlib import Path

import pytest

from openbias.traces import (
    TraceCase,
    TraceDataset,
    TraceMetadata,
    TraceValidationError,
    append_trace_case,
    load_trace_dataset,
    save_trace_dataset,
)


def test_save_and_load_trace_dataset_round_trip(tmp_path: Path):
    dataset = TraceDataset(
        name="round-trip",
        cases=[
            TraceCase(
                id="trace-1",
                session_id="sess-1",
                messages=[
                    {"role": "user", "content": "Need a refund"},
                    {"role": "assistant", "content": "Please verify your identity first."},
                ],
                metadata=TraceMetadata(
                    model="gpt-4o-mini",
                    timestamp="2026-04-05T00:00:00Z",
                    final_action="intervene",
                    evaluator_names=("workflow-guard",),
                ),
            )
        ],
    )

    path = tmp_path / "traces.jsonl"
    save_trace_dataset(dataset, path)

    loaded = load_trace_dataset(path)
    assert loaded.name == "traces"
    assert loaded.source_path == str(path)
    assert len(loaded.cases) == 1
    assert loaded.cases[0].metadata.final_action == "intervene"


def test_append_trace_case_creates_jsonl_dataset(tmp_path: Path):
    path = tmp_path / "daily" / "2026-04-05.jsonl"
    append_trace_case(
        path,
        TraceCase(
            id="trace-1",
            session_id="sess-1",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        ),
    )
    append_trace_case(
        path,
        TraceCase(
            id="trace-2",
            session_id="sess-2",
            messages=[
                {"role": "user", "content": "Refund me"},
                {"role": "assistant", "content": "I need to verify your identity first."},
            ],
        ),
    )

    loaded = load_trace_dataset(path)
    assert [case.id for case in loaded.cases] == ["trace-1", "trace-2"]


def test_load_trace_dataset_rejects_malformed_json(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(TraceValidationError, match="Failed to parse JSONL row 1"):
        load_trace_dataset(path)


def test_load_trace_dataset_rejects_missing_required_fields(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"trace-1","messages":[]}\n', encoding="utf-8")

    with pytest.raises(TraceValidationError, match="missing required field 'session_id'"):
        load_trace_dataset(path)
