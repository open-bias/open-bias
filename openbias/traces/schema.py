"""Canonical schema and validation helpers for replayable trace datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

TraceDecision = Literal["allow", "intervene", "block", "shadow", "error", "unknown"]
_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
_ALLOWED_DECISIONS = {"allow", "intervene", "block", "shadow", "error", "unknown"}


class TraceValidationError(ValueError):
    """Raised when trace data cannot be normalized into the canonical schema."""


@dataclass(frozen=True)
class TraceViolationSummary:
    """Compact summary of one reported violation."""

    reason: str
    rule_id: str | None = None
    severity: str | None = None

    def __post_init__(self) -> None:
        if not self.reason or not isinstance(self.reason, str):
            raise TraceValidationError("TraceViolationSummary.reason must be a non-empty string.")
        if self.rule_id is not None and (not isinstance(self.rule_id, str) or not self.rule_id):
            raise TraceValidationError("TraceViolationSummary.rule_id must be a non-empty string when present.")
        if self.severity is not None and (not isinstance(self.severity, str) or not self.severity):
            raise TraceValidationError("TraceViolationSummary.severity must be a non-empty string when present.")


@dataclass(frozen=True)
class TraceEvaluatorSummary:
    """Per-evaluator summary captured from a traced request."""

    name: str
    engine_type: str
    phase: Literal["pre_call", "post_call"]
    decision: TraceDecision = "unknown"
    violations: tuple[TraceViolationSummary, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise TraceValidationError("TraceEvaluatorSummary.name must be a non-empty string.")
        if not self.engine_type or not isinstance(self.engine_type, str):
            raise TraceValidationError("TraceEvaluatorSummary.engine_type must be a non-empty string.")
        if self.phase not in {"pre_call", "post_call"}:
            raise TraceValidationError("TraceEvaluatorSummary.phase must be pre_call or post_call.")
        if self.decision not in _ALLOWED_DECISIONS:
            raise TraceValidationError(
                f"Unsupported evaluator decision {self.decision!r}; expected one of {_ALLOWED_DECISIONS}."
            )
        object.__setattr__(
            self,
            "violations",
            tuple(_coerce_violation_summary(v) for v in self.violations),
        )


@dataclass(frozen=True)
class TraceIntervention:
    """Intervention that was applied or queued for the traced request."""

    name: str
    strategy: str | None = None
    applied_at: Literal["pre_call", "post_call", "next_turn"] | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise TraceValidationError("TraceIntervention.name must be a non-empty string.")
        if self.strategy is not None and (not isinstance(self.strategy, str) or not self.strategy):
            raise TraceValidationError("TraceIntervention.strategy must be a non-empty string when present.")
        if self.applied_at is not None and self.applied_at not in {"pre_call", "post_call", "next_turn"}:
            raise TraceValidationError(
                "TraceIntervention.applied_at must be pre_call, post_call, or next_turn when present."
            )
        if self.message is not None and not isinstance(self.message, str):
            raise TraceValidationError("TraceIntervention.message must be a string when present.")


@dataclass(frozen=True)
class TraceMetadata:
    """Metadata attached to one replayable trace case."""

    model: str | None = None
    timestamp: str | None = None
    environment: str | None = None
    policy_hash: str | None = None
    request_id: str | None = None
    final_action: TraceDecision = "unknown"
    evaluator_names: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("model", "timestamp", "environment", "policy_hash", "request_id"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TraceValidationError(f"TraceMetadata.{field_name} must be a non-empty string when present.")
        if self.final_action not in _ALLOWED_DECISIONS:
            raise TraceValidationError(
                f"Unsupported final_action {self.final_action!r}; expected one of {_ALLOWED_DECISIONS}."
            )
        object.__setattr__(self, "evaluator_names", _coerce_strings(self.evaluator_names, "TraceMetadata.evaluator_names"))
        object.__setattr__(self, "tags", _coerce_strings(self.tags, "TraceMetadata.tags"))
        if not isinstance(self.extras, dict):
            raise TraceValidationError("TraceMetadata.extras must be a dict.")


@dataclass(frozen=True)
class TraceCase:
    """One normalized request/response pair plus evaluation metadata."""

    id: str
    session_id: str
    messages: list[dict[str, Any]]
    metadata: TraceMetadata = field(default_factory=TraceMetadata)
    evaluator_summaries: tuple[TraceEvaluatorSummary, ...] = ()
    interventions: tuple[TraceIntervention, ...] = ()
    tool_calls: tuple[dict[str, Any], ...] = ()
    expected_outcome: dict[str, Any] | None = None
    labels: dict[str, Any] | None = None
    source: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise TraceValidationError("TraceCase.id must be a non-empty string.")
        if not self.session_id or not isinstance(self.session_id, str):
            raise TraceValidationError("TraceCase.session_id must be a non-empty string.")
        normalized_messages = _validate_messages(self.messages, case_id=self.id)
        _validate_request_response_pair(normalized_messages, case_id=self.id)
        if not isinstance(self.metadata, TraceMetadata):
            raise TraceValidationError("TraceCase.metadata must be a TraceMetadata instance.")
        object.__setattr__(
            self,
            "evaluator_summaries",
            tuple(_coerce_evaluator_summary(summary) for summary in self.evaluator_summaries),
        )
        object.__setattr__(
            self,
            "interventions",
            tuple(_coerce_intervention(intervention) for intervention in self.interventions),
        )
        object.__setattr__(self, "messages", normalized_messages)
        object.__setattr__(self, "tool_calls", _validate_dict_sequence(self.tool_calls, "TraceCase.tool_calls"))
        object.__setattr__(self, "expected_outcome", _optional_dict(self.expected_outcome, "TraceCase.expected_outcome"))
        object.__setattr__(self, "labels", _optional_dict(self.labels, "TraceCase.labels"))
        object.__setattr__(self, "source", _optional_dict(self.source, "TraceCase.source"))


@dataclass(frozen=True)
class TraceDataset:
    """Collection of replayable trace cases loaded from one dataset file."""

    name: str
    cases: list[TraceCase]
    source_path: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise TraceValidationError("TraceDataset.name must be a non-empty string.")
        if not isinstance(self.cases, list) or not self.cases:
            raise TraceValidationError("TraceDataset.cases must contain at least one case.")
        if self.source_path is not None and (not isinstance(self.source_path, str) or not self.source_path):
            raise TraceValidationError("TraceDataset.source_path must be a non-empty string when present.")
        if self.description is not None and (not isinstance(self.description, str) or not self.description):
            raise TraceValidationError("TraceDataset.description must be a non-empty string when present.")
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise TraceValidationError(f"Duplicate trace case id {case.id!r}.")
            seen.add(case.id)


def trace_case_to_dict(case: TraceCase) -> dict[str, Any]:
    """Serialize a ``TraceCase`` into a JSON-friendly dict."""

    return asdict(case)


def trace_dataset_name_from_path(path: str | Path) -> str:
    """Derive a default dataset name from a JSONL path."""

    return Path(path).stem or "trace-dataset"


def _coerce_strings(values: Any, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise TraceValidationError(f"{field_name} must be a list or tuple of strings.")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise TraceValidationError(f"{field_name} entries must be non-empty strings.")
        normalized.append(value)
    return tuple(normalized)


def _optional_dict(value: Any, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TraceValidationError(f"{field_name} must be a dict when present.")
    return value


def _validate_dict_sequence(values: Any, field_name: str) -> tuple[dict[str, Any], ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise TraceValidationError(f"{field_name} must be a list or tuple of dicts.")
    normalized: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            raise TraceValidationError(f"{field_name} entries must be dicts.")
        normalized.append(value)
    return tuple(normalized)


def _validate_messages(messages: Any, case_id: str) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise TraceValidationError(f"Trace case {case_id!r} must contain at least one message.")
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TraceValidationError(f"Trace case {case_id!r} message {index} must be a dict.")
        role = message.get("role")
        if role not in _ALLOWED_ROLES:
            raise TraceValidationError(
                f"Trace case {case_id!r} message {index} has unsupported role {role!r}."
            )
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise TraceValidationError(
                f"Trace case {case_id!r} message {index} content must be a string when present."
            )
        normalized.append(dict(message))
    return normalized


def _validate_request_response_pair(messages: list[dict[str, Any]], case_id: str) -> None:
    assistant_indexes = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    if len(assistant_indexes) != 1:
        raise TraceValidationError(
            f"Trace case {case_id!r} must contain exactly one assistant response message."
        )
    assistant_index = assistant_indexes[0]
    if assistant_index != len(messages) - 1:
        raise TraceValidationError(
            f"Trace case {case_id!r} assistant response must be the final message."
        )
    if not any(message["role"] != "assistant" for message in messages[:assistant_index]):
        raise TraceValidationError(
            f"Trace case {case_id!r} must contain at least one request-side message before the assistant response."
        )


def _coerce_violation_summary(value: Any) -> TraceViolationSummary:
    if isinstance(value, TraceViolationSummary):
        return value
    if isinstance(value, dict):
        return TraceViolationSummary(**value)
    raise TraceValidationError("TraceEvaluatorSummary.violations entries must be TraceViolationSummary or dict.")


def _coerce_evaluator_summary(value: Any) -> TraceEvaluatorSummary:
    if isinstance(value, TraceEvaluatorSummary):
        return value
    if isinstance(value, dict):
        return TraceEvaluatorSummary(**value)
    raise TraceValidationError("TraceCase.evaluator_summaries entries must be TraceEvaluatorSummary or dict.")


def _coerce_intervention(value: Any) -> TraceIntervention:
    if isinstance(value, TraceIntervention):
        return value
    if isinstance(value, dict):
        return TraceIntervention(**value)
    raise TraceValidationError("TraceCase.interventions entries must be TraceIntervention or dict.")
