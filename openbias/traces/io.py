"""JSONL helpers for replayable trace datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from openbias.traces.schema import (
    TraceCase,
    TraceDataset,
    TraceMetadata,
    TraceValidationError,
    trace_case_to_dict,
    trace_dataset_name_from_path,
)


def load_trace_dataset(path: str | Path, *, name: str | None = None) -> TraceDataset:
    """Load a JSONL trace dataset from disk."""

    dataset_path = Path(path)
    cases: list[TraceCase] = []

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceValidationError(
                    f"Failed to parse JSONL row {line_number} in {dataset_path}: {exc.msg}"
                ) from exc

            if not isinstance(row, dict):
                raise TraceValidationError(
                    f"Trace dataset rows must be objects; row {line_number} in {dataset_path} was {type(row).__name__}."
                )

            metadata = row.get("metadata", {})
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise TraceValidationError(
                    f"Trace row {line_number} in {dataset_path} has invalid metadata; expected an object."
                )

            try:
                cases.append(
                    TraceCase(
                        id=row["id"],
                        session_id=row["session_id"],
                        messages=row["messages"],
                        metadata=TraceMetadata(**metadata),
                        evaluator_summaries=row.get("evaluator_summaries", ()),
                        interventions=row.get("interventions", ()),
                        tool_calls=row.get("tool_calls", ()),
                        expected_outcome=row.get("expected_outcome"),
                        labels=row.get("labels"),
                        source=row.get("source"),
                    )
                )
            except KeyError as exc:
                raise TraceValidationError(
                    f"Trace row {line_number} in {dataset_path} is missing required field {exc.args[0]!r}."
                ) from exc

    if not cases:
        raise TraceValidationError(f"Trace dataset {dataset_path} did not contain any cases.")

    return TraceDataset(
        name=name or trace_dataset_name_from_path(dataset_path),
        cases=cases,
        source_path=str(dataset_path),
    )


def save_trace_dataset(dataset: TraceDataset, path: str | Path) -> Path:
    """Write a trace dataset to JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for case in dataset.cases:
            handle.write(json.dumps(trace_case_to_dict(case), ensure_ascii=False))
            handle.write("\n")
    return output_path


def append_trace_case(path: str | Path, case: TraceCase) -> Path:
    """Append one trace case to a JSONL dataset."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace_case_to_dict(case), ensure_ascii=False))
        handle.write("\n")
    return output_path


def iter_trace_cases(path: str | Path) -> Iterable[TraceCase]:
    """Yield validated trace cases from a JSONL dataset."""

    dataset = load_trace_dataset(path)
    yield from dataset.cases
