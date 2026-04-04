"""Canonical schema and validation helpers for the rebuilt eval harness."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DetectionScope = Literal["request", "response", "either"]
EvalOutcomeName = Literal[
    "correct_non_violation",
    "detected_violation",
    "missed_violation",
    "false_positive",
    "detected_and_fixed",
    "detected_not_fixed",
]

_ALLOWED_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}
_ALLOWED_SCOPES = {"request", "response", "either"}


class EvalValidationError(ValueError):
    """Raised when suite data cannot be normalized into the canonical schema."""


@dataclass(frozen=True)
class EvalLabels:
    """Explicit ground-truth labels for a canonical eval case."""

    violation: bool
    detection_scope: DetectionScope
    detect_at_turn: int | None
    repair_expected: bool | None
    repair_verified_at_turn: int | None

    def __post_init__(self) -> None:
        if self.detection_scope not in _ALLOWED_SCOPES:
            raise EvalValidationError(
                f"Unsupported detection_scope {self.detection_scope!r}; "
                "expected one of request, response, or either."
            )
        if self.violation and self.detect_at_turn is None:
            raise EvalValidationError("Violation cases must set labels.detect_at_turn.")
        if not self.violation and self.detect_at_turn is not None:
            raise EvalValidationError("Non-violation cases must leave labels.detect_at_turn empty.")
        if self.detect_at_turn is not None and self.detect_at_turn < 0:
            raise EvalValidationError("labels.detect_at_turn must be >= 0.")
        if self.repair_expected is None and self.repair_verified_at_turn is not None:
            raise EvalValidationError(
                "labels.repair_verified_at_turn requires labels.repair_expected."
            )
        if self.repair_expected is not None and not self.violation:
            raise EvalValidationError("Repair labels are only valid for violation cases.")
        if self.repair_expected is not None and self.repair_verified_at_turn is None:
            raise EvalValidationError(
                "Repair cases must set labels.repair_verified_at_turn."
            )
        if self.repair_verified_at_turn is not None and self.repair_verified_at_turn < 0:
            raise EvalValidationError("labels.repair_verified_at_turn must be >= 0.")
        if (
            self.detect_at_turn is not None
            and self.repair_verified_at_turn is not None
            and self.repair_verified_at_turn < self.detect_at_turn
        ):
            raise EvalValidationError(
                "labels.repair_verified_at_turn must be on or after labels.detect_at_turn."
            )


@dataclass(frozen=True)
class EvalCase:
    """Conversation-shaped eval case with explicit labels."""

    id: str
    messages: list[dict[str, Any]]
    tags: list[str] = field(default_factory=list)
    source: dict[str, Any] | None = None
    labels: EvalLabels = field(
        default_factory=lambda: EvalLabels(
            violation=False,
            detection_scope="either",
            detect_at_turn=None,
            repair_expected=None,
            repair_verified_at_turn=None,
        )
    )

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise EvalValidationError("Each eval case must have a non-empty string id.")
        if not isinstance(self.messages, list) or not self.messages:
            raise EvalValidationError(f"Case {self.id!r} must contain at least one message.")
        normalized_messages = _validate_messages(self.messages, case_id=self.id)
        normalized_tags = _validate_tags(self.tags, case_id=self.id)
        normalized_source = _validate_source(self.source, case_id=self.id)
        turn_blueprint = build_turn_blueprint(normalized_messages)
        if not turn_blueprint:
            raise EvalValidationError(
                f"Case {self.id!r} does not contain any evaluable turns."
            )
        _validate_label_turns(self.id, self.labels, turn_blueprint)
        object.__setattr__(self, "messages", normalized_messages)
        object.__setattr__(self, "tags", normalized_tags)
        object.__setattr__(self, "source", normalized_source)


@dataclass(frozen=True)
class EvalSuite:
    """Suite metadata plus canonical eval cases."""

    name: str
    cases: list[EvalCase]
    description: str | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise EvalValidationError("EvalSuite.name must be a non-empty string.")
        if not isinstance(self.cases, list) or not self.cases:
            raise EvalValidationError("EvalSuite.cases must contain at least one case.")
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise EvalValidationError(f"Duplicate eval case id {case.id!r}.")
            seen.add(case.id)


@dataclass(frozen=True)
class EvalCaseOutcome:
    """Observed outcome for one canonical case."""

    case_id: str
    outcome: EvalOutcomeName
    passed: bool
    detected: bool
    false_positive: bool
    fixed: bool | None
    detection_turns: tuple[int, ...] = ()
    detection_boundaries: tuple[str, ...] = ()
    expected_outcome: EvalOutcomeName | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalSummary:
    """Binary aggregate metrics for v1 eval runs."""

    true_positive: int
    false_negative: int
    false_positive: int
    true_negative: int
    detection_recall: float
    false_positive_rate: float
    fix_success_count: int
    fix_failure_count: int
    fix_rate: float
    exact_case_pass_rate: float


@dataclass(frozen=True)
class EvalRunResult:
    """Per-case outcomes, execution failures, and aggregate binary metrics."""

    suite_name: str
    outcomes: list[EvalCaseOutcome]
    failures: list[dict[str, str]]
    summary: EvalSummary


@dataclass(frozen=True)
class TurnBlueprint:
    """Conversation turn derived from a flat message list."""

    index: int
    request_messages: list[dict[str, Any]]
    assistant_message: dict[str, Any] | None

    @property
    def has_response(self) -> bool:
        return self.assistant_message is not None


def build_turn_blueprint(messages: list[dict[str, Any]]) -> list[TurnBlueprint]:
    """Normalize a flat conversation into evaluable turns."""

    turns: list[TurnBlueprint] = []
    buffer: list[dict[str, Any]] = []
    seen_request_since_last_assistant = False

    for raw_message in messages:
        message = deepcopy(raw_message)
        role = message["role"]
        if role == "assistant":
            turns.append(
                TurnBlueprint(
                    index=len(turns),
                    request_messages=deepcopy(buffer),
                    assistant_message=message,
                )
            )
            buffer.append(message)
            seen_request_since_last_assistant = False
            continue

        buffer.append(message)
        if role == "user":
            seen_request_since_last_assistant = True

    if seen_request_since_last_assistant:
        turns.append(
            TurnBlueprint(
                index=len(turns),
                request_messages=deepcopy(buffer),
                assistant_message=None,
            )
        )

    return turns


def load_serialized_suite(path: Path) -> dict[str, Any]:
    """Load a native suite document from YAML or JSON."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        import json

        payload = json.loads(path.read_text())
    elif suffix in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(path.read_text())
    else:
        raise EvalValidationError(
            f"Unsupported suite format for {path}; expected .json, .yaml, or .yml."
        )

    if not isinstance(payload, dict):
        raise EvalValidationError(f"Suite file {path} must contain a top-level mapping.")
    return payload


def _validate_messages(messages: list[dict[str, Any]], *, case_id: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise EvalValidationError(
                f"Case {case_id!r} message {index} must be an object."
            )
        role = message.get("role")
        if role not in _ALLOWED_MESSAGE_ROLES:
            raise EvalValidationError(
                f"Case {case_id!r} message {index} has unsupported role {role!r}."
            )
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise EvalValidationError(
                f"Case {case_id!r} message {index} content must be a string or null."
            )
        if role == "assistant" and content is None and "tool_calls" not in message:
            raise EvalValidationError(
                f"Case {case_id!r} assistant message {index} needs content or tool_calls."
            )
        normalized.append(deepcopy(message))
    return normalized


def _validate_tags(tags: list[str], *, case_id: str) -> list[str]:
    if not isinstance(tags, list):
        raise EvalValidationError(f"Case {case_id!r} tags must be a list of strings.")
    normalized: list[str] = []
    for tag in tags:
        if not isinstance(tag, str) or not tag:
            raise EvalValidationError(f"Case {case_id!r} tags must be non-empty strings.")
        normalized.append(tag)
    return normalized


def _validate_source(source: dict[str, Any] | None, *, case_id: str) -> dict[str, Any] | None:
    if source is None:
        return None
    if not isinstance(source, dict):
        raise EvalValidationError(f"Case {case_id!r} source must be an object when present.")
    return deepcopy(source)


def _validate_label_turns(
    case_id: str,
    labels: EvalLabels,
    turns: list[TurnBlueprint],
) -> None:
    if labels.detect_at_turn is not None and labels.detect_at_turn >= len(turns):
        raise EvalValidationError(
            f"Case {case_id!r} labels.detect_at_turn points past the end of the conversation."
        )
    if (
        labels.repair_verified_at_turn is not None
        and labels.repair_verified_at_turn >= len(turns)
    ):
        raise EvalValidationError(
            f"Case {case_id!r} labels.repair_verified_at_turn points past the end of the conversation."
        )
    if labels.detect_at_turn is not None:
        turn = turns[labels.detect_at_turn]
        if labels.detection_scope == "response" and not turn.has_response:
            raise EvalValidationError(
                f"Case {case_id!r} labels.detect_at_turn targets a request-only turn."
            )
    if labels.repair_verified_at_turn is not None:
        repair_turn = turns[labels.repair_verified_at_turn]
        if not repair_turn.has_response:
            raise EvalValidationError(
                f"Case {case_id!r} repair verification requires an assistant response turn."
            )
