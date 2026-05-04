"""Suite loaders and adapters for the rebuilt eval harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from openbias.eval.schema import (
    DetectionScope,
    EvalCase,
    EvalLabels,
    EvalPolicyTarget,
    EvalSuite,
    EvalValidationError,
    load_serialized_suite,
)

_LABEL_FIELDS = {
    "labels.violation",
    "labels.detection_scope",
    "labels.detect_at_turn",
    "labels.repair_expected",
    "labels.repair_verified_at_turn",
}
_SUPPORTED_MAPPING_FIELDS = {"id", "messages", "tags", "source", *_LABEL_FIELDS}
_REQUIRED_MAPPING_FIELDS = {"id", "messages", *_LABEL_FIELDS}


class SuiteAdapter(Protocol):
    """Pure parser/normalizer contract for eval suite adapters."""

    def load(self, path: str | Path, mapping_config: dict[str, Any] | None = None) -> EvalSuite:
        """Load and validate a suite document into the canonical schema."""


@dataclass(frozen=True)
class NativeSuiteAdapter:
    """Load repo-owned YAML or JSON suites that already follow the canonical schema."""

    def load(self, path: str | Path, mapping_config: dict[str, Any] | None = None) -> EvalSuite:
        del mapping_config
        suite_path = Path(path)
        payload = load_serialized_suite(suite_path)
        cases_payload = payload.get("cases")
        if not isinstance(cases_payload, list) or not cases_payload:
            raise EvalValidationError(f"Suite file {suite_path} must define a non-empty cases list.")

        cases = [_build_case(case_payload) for case_payload in cases_payload]
        return EvalSuite(
            name=_expect_string(payload.get("name"), context=f"{suite_path}:name"),
            description=_optional_string(payload.get("description"), context=f"{suite_path}:description"),
            policy=_build_policy_target(payload.get("policy"), context=f"{suite_path}:policy"),
            source_path=str(suite_path),
            cases=cases,
        )


@dataclass(frozen=True)
class JsonlSuiteAdapter:
    """Load external JSONL datasets with explicit field mapping into the canonical schema."""

    suite_name: str | None = None

    def load(self, path: str | Path, mapping_config: dict[str, Any] | None = None) -> EvalSuite:
        suite_path = Path(path)
        if mapping_config is None:
            raise EvalValidationError("JSONL imports require an explicit mapping_config.")

        config = _normalize_mapping_config(mapping_config)
        cases: list[EvalCase] = []

        for line_number, raw_line in enumerate(suite_path.read_text().splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalValidationError(
                    f"{suite_path}:{line_number} contains invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise EvalValidationError(f"{suite_path}:{line_number} must be a JSON object.")

            events = _extract_events(row, config.event_path, suite_path=suite_path, line_number=line_number)
            for event_index, event in enumerate(events):
                cases.append(
                    _build_imported_case(
                        row=row,
                        event=event,
                        event_index=event_index,
                        event_count=len(events),
                        mapping=config.fields,
                        suite_path=suite_path,
                        line_number=line_number,
                    )
                )

        if not cases:
            raise EvalValidationError(f"{suite_path} did not yield any canonical eval cases.")

        suite_name = self.suite_name or suite_path.stem
        return EvalSuite(name=suite_name, source_path=str(suite_path), cases=cases)


@dataclass(frozen=True)
class _NormalizedMapping:
    fields: dict[str, str]
    event_path: str | None


def load_native_suite(path: str | Path) -> EvalSuite:
    """Convenience loader for canonical YAML/JSON suites."""

    return NativeSuiteAdapter().load(path)


def load_jsonl_suite(
    path: str | Path,
    *,
    mapping_config: dict[str, Any],
    suite_name: str | None = None,
) -> EvalSuite:
    """Convenience loader for mapped external JSONL datasets."""

    return JsonlSuiteAdapter(suite_name=suite_name).load(path, mapping_config=mapping_config)


def _build_case(payload: Any) -> EvalCase:
    if not isinstance(payload, dict):
        raise EvalValidationError("Each native case entry must be an object.")
    labels_payload = payload.get("labels")
    if not isinstance(labels_payload, dict):
        raise EvalValidationError("Each native case must include a labels object.")

    return EvalCase(
        id=_expect_string(payload.get("id"), context="case.id"),
        messages=_expect_list(payload.get("messages"), context="case.messages"),
        tags=_optional_string_list(payload.get("tags"), context="case.tags"),
        source=_optional_mapping(payload.get("source"), context="case.source"),
        labels=_build_labels(labels_payload, context="case.labels"),
    )


def _build_labels(payload: dict[str, Any], *, context: str) -> EvalLabels:
    expected_keys = {
        "violation",
        "detection_scope",
        "detect_at_turn",
        "repair_expected",
        "repair_verified_at_turn",
    }
    unexpected = sorted(set(payload) - expected_keys)
    if unexpected:
        raise EvalValidationError(f"{context} has unsupported fields: {', '.join(unexpected)}")

    detection_scope_raw = _expect_string(
        payload.get("detection_scope"),
        context=f"{context}.detection_scope",
    )
    if detection_scope_raw not in {"request", "response", "either"}:
        raise EvalValidationError(
            f"{context}.detection_scope must be one of request, response, or either."
        )

    return EvalLabels(
        violation=_expect_bool(payload.get("violation"), context=f"{context}.violation"),
        detection_scope=cast(DetectionScope, detection_scope_raw),
        detect_at_turn=_optional_int(payload.get("detect_at_turn"), context=f"{context}.detect_at_turn"),
        repair_expected=_optional_bool(
            payload.get("repair_expected"),
            context=f"{context}.repair_expected",
        ),
        repair_verified_at_turn=_optional_int(
            payload.get("repair_verified_at_turn"),
            context=f"{context}.repair_verified_at_turn",
        ),
    )


def _build_policy_target(payload: Any, *, context: str) -> EvalPolicyTarget | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise EvalValidationError(f"{context} must be an object when present.")

    expected_keys = {"name", "rules_path", "notes"}
    unexpected = sorted(set(payload) - expected_keys)
    if unexpected:
        raise EvalValidationError(f"{context} has unsupported fields: {', '.join(unexpected)}")

    return EvalPolicyTarget(
        name=_expect_string(payload.get("name"), context=f"{context}.name"),
        rules_path=_expect_string(payload.get("rules_path"), context=f"{context}.rules_path"),
        notes=_optional_string(payload.get("notes"), context=f"{context}.notes"),
    )


def _normalize_mapping_config(mapping_config: dict[str, Any]) -> _NormalizedMapping:
    mapping = mapping_config.get("fields")
    event_path = mapping_config.get("event_path")

    if not isinstance(mapping, dict):
        raise EvalValidationError("mapping_config.fields must be a mapping of canonical fields to source paths.")
    if event_path is not None and not isinstance(event_path, str):
        raise EvalValidationError("mapping_config.event_path must be a string when present.")

    unsupported_fields = sorted(set(mapping) - _SUPPORTED_MAPPING_FIELDS)
    if unsupported_fields:
        raise EvalValidationError(
            f"Unsupported field mappings: {', '.join(unsupported_fields)}"
        )

    missing_fields = sorted(field for field in _REQUIRED_MAPPING_FIELDS if field not in mapping)
    if missing_fields:
        raise EvalValidationError(
            "JSONL imports require explicit mappings for: " + ", ".join(missing_fields)
        )

    normalized_fields: dict[str, str] = {}
    for canonical_field, raw_path in mapping.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise EvalValidationError(
                f"Mapping for {canonical_field!r} must be a non-empty dotted path."
            )
        normalized_fields[canonical_field] = raw_path

    return _NormalizedMapping(fields=normalized_fields, event_path=event_path)


def _extract_events(
    row: dict[str, Any],
    event_path: str | None,
    *,
    suite_path: Path,
    line_number: int,
) -> list[dict[str, Any] | None]:
    if event_path is None:
        return [None]

    events = _resolve_path(row, event_path, suite_path=suite_path, line_number=line_number)
    if not isinstance(events, list) or not events:
        raise EvalValidationError(
            f"{suite_path}:{line_number} event_path {event_path!r} must resolve to a non-empty list."
        )
    normalized_events: list[dict[str, Any] | None] = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise EvalValidationError(
                f"{suite_path}:{line_number} event {event_index} must be an object."
            )
        normalized_events.append(event)
    return normalized_events


def _build_imported_case(
    *,
    row: dict[str, Any],
    event: dict[str, Any] | None,
    event_index: int,
    event_count: int,
    mapping: dict[str, str],
    suite_path: Path,
    line_number: int,
) -> EvalCase:
    row_id = _resolve_required(
        row,
        mapping["id"],
        suite_path=suite_path,
        line_number=line_number,
    )
    if not isinstance(row_id, str) or not row_id:
        raise EvalValidationError(f"{suite_path}:{line_number} mapped id must be a non-empty string.")

    case_id = row_id
    if event is not None:
        case_id = f"{row_id}#event-{event_index}"

    messages = _resolve_required(
        row,
        mapping["messages"],
        suite_path=suite_path,
        line_number=line_number,
    )
    tags = _resolve_optional(
        row,
        mapping.get("tags"),
        suite_path=suite_path,
        line_number=line_number,
    )
    source = _resolve_optional(
        row,
        mapping.get("source"),
        suite_path=suite_path,
        line_number=line_number,
    )

    if source is None:
        source = {
            "import_path": str(suite_path),
            "line_number": line_number,
        }
        if event_count > 1:
            source["event_index"] = event_index

    label_source = event if event is not None else row
    labels_payload = {
        "violation": _resolve_required(
            label_source,
            mapping["labels.violation"],
            suite_path=suite_path,
            line_number=line_number,
        ),
        "detection_scope": _resolve_required(
            label_source,
            mapping["labels.detection_scope"],
            suite_path=suite_path,
            line_number=line_number,
        ),
        "detect_at_turn": _resolve_required(
            label_source,
            mapping["labels.detect_at_turn"],
            suite_path=suite_path,
            line_number=line_number,
        ),
        "repair_expected": _resolve_required(
            label_source,
            mapping["labels.repair_expected"],
            suite_path=suite_path,
            line_number=line_number,
        ),
        "repair_verified_at_turn": _resolve_required(
            label_source,
            mapping["labels.repair_verified_at_turn"],
            suite_path=suite_path,
            line_number=line_number,
        ),
    }

    return EvalCase(
        id=case_id,
        messages=_expect_list(messages, context=f"{suite_path}:{line_number}:messages"),
        tags=_optional_string_list(tags, context=f"{suite_path}:{line_number}:tags"),
        source=_optional_mapping(source, context=f"{suite_path}:{line_number}:source"),
        labels=_build_labels(labels_payload, context=f"{suite_path}:{line_number}:labels"),
    )


def _resolve_required(
    payload: dict[str, Any],
    path: str,
    *,
    suite_path: Path,
    line_number: int,
) -> Any:
    return _resolve_path(payload, path, suite_path=suite_path, line_number=line_number)


def _resolve_optional(
    payload: dict[str, Any],
    path: str | None,
    *,
    suite_path: Path,
    line_number: int,
) -> Any:
    if path is None:
        return None
    return _resolve_path(payload, path, suite_path=suite_path, line_number=line_number)


def _resolve_path(
    payload: dict[str, Any],
    path: str,
    *,
    suite_path: Path,
    line_number: int,
) -> Any:
    current: Any = payload
    for segment in path.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
            continue
        if isinstance(current, list) and segment.isdigit():
            index = int(segment)
            if index < len(current):
                current = current[index]
                continue
        raise EvalValidationError(
            f"{suite_path}:{line_number} mapping {path!r} could not resolve segment {segment!r}."
        )
    return current


def _expect_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvalValidationError(f"{context} must be a non-empty string.")
    return value


def _optional_string(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, context=context)


def _expect_bool(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise EvalValidationError(f"{context} must be a boolean.")
    return value


def _optional_bool(value: Any, *, context: str) -> bool | None:
    if value is None:
        return None
    return _expect_bool(value, context=context)


def _optional_int(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise EvalValidationError(f"{context} must be an integer or null.")
    return value


def _expect_list(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvalValidationError(f"{context} must be a list.")
    return value


def _optional_mapping(value: Any, *, context: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise EvalValidationError(f"{context} must be an object when present.")
    return value


def _optional_string_list(value: Any, *, context: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EvalValidationError(f"{context} must be a list of strings.")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise EvalValidationError(f"{context} must only contain non-empty strings.")
        normalized.append(item)
    return normalized
