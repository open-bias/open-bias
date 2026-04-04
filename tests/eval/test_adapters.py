"""Adapter tests for native suites and mapped JSONL imports."""

from __future__ import annotations

from pathlib import Path

import pytest

from openbias.eval import EvalValidationError, load_jsonl_suite, load_native_suite

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO_SUITES = Path(__file__).resolve().parents[2] / "evals" / "suites"


def test_load_native_yaml_suite():
    suite = load_native_suite(FIXTURES / "native_suite.yaml")

    assert suite.name == "native-smoke"
    assert [case.id for case in suite.cases] == ["safe-single-turn", "detected-request-risk"]


def test_load_all_repo_owned_suite_files():
    suite_paths = sorted(REPO_SUITES.glob("*.yaml"))

    assert [path.name for path in suite_paths] == [
        "false_positive_guards.yaml",
        "repair.yaml",
        "request.yaml",
        "response.yaml",
        "safe.yaml",
    ]

    loaded = [load_native_suite(path) for path in suite_paths]

    assert [suite.name for suite in loaded] == [
        "false-positive-guards",
        "repair-basic",
        "request-basic",
        "response-basic",
        "safe-basic",
    ]
    assert sum(len(suite.cases) for suite in loaded) == 14


def test_load_native_json_suite(tmp_path):
    path = tmp_path / "native.json"
    path.write_text(
        "{"
        '"name": "json-suite",'
        '"cases": ['
        "{"
        '"id": "json-safe",'
        '"messages": [{"role": "user", "content": "hello"}],'
        '"tags": ["json"],'
        '"labels": {'
        '"violation": false,'
        '"detection_scope": "either",'
        '"detect_at_turn": null,'
        '"repair_expected": null,'
        '"repair_verified_at_turn": null'
        "}"
        "}"
        "]"
        "}"
    )

    suite = load_native_suite(path)
    assert suite.name == "json-suite"
    assert suite.cases[0].id == "json-safe"


def test_load_jsonl_suite_with_field_mapping(tmp_path):
    path = tmp_path / "import.jsonl"
    path.write_text(
        '{"case_id":"external-1","payload":{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"unsafe answer"}]},'
        '"meta":{"tags":["external"]},"labels":{"violation":true,"scope":"response","turn":0,"repair_expected":null,"repair_turn":null}}\n'
    )

    suite = load_jsonl_suite(
        path,
        mapping_config={
            "fields": {
                "id": "case_id",
                "messages": "payload.messages",
                "tags": "meta.tags",
                "labels.violation": "labels.violation",
                "labels.detection_scope": "labels.scope",
                "labels.detect_at_turn": "labels.turn",
                "labels.repair_expected": "labels.repair_expected",
                "labels.repair_verified_at_turn": "labels.repair_turn",
            }
        },
        suite_name="external-import",
    )

    assert suite.name == "external-import"
    assert suite.cases[0].id == "external-1"
    assert suite.cases[0].labels.detection_scope == "response"


def test_jsonl_import_rejects_malformed_row(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"case_id": "bad"\n')

    with pytest.raises(EvalValidationError, match="invalid JSON"):
        load_jsonl_suite(
            path,
            mapping_config={
                "fields": {
                    "id": "case_id",
                    "messages": "messages",
                    "labels.violation": "labels.violation",
                    "labels.detection_scope": "labels.scope",
                    "labels.detect_at_turn": "labels.turn",
                    "labels.repair_expected": "labels.repair_expected",
                    "labels.repair_verified_at_turn": "labels.repair_turn",
                }
            },
        )


def test_jsonl_import_rejects_missing_explicit_labels(tmp_path):
    path = tmp_path / "missing-labels.jsonl"
    path.write_text(
        '{"case_id":"external-1","messages":[{"role":"user","content":"hello"}],'
        '"labels":{"violation":true,"scope":"request","turn":0,"repair_turn":null}}\n'
    )

    with pytest.raises(EvalValidationError, match="explicit mappings"):
        load_jsonl_suite(
            path,
            mapping_config={
                "fields": {
                    "id": "case_id",
                    "messages": "messages",
                    "labels.violation": "labels.violation",
                    "labels.detection_scope": "labels.scope",
                    "labels.detect_at_turn": "labels.turn",
                    "labels.repair_verified_at_turn": "labels.repair_turn",
                }
            },
        )


def test_jsonl_import_rejects_unsupported_mapping_targets(tmp_path):
    path = tmp_path / "unsupported.jsonl"
    path.write_text(
        '{"case_id":"external-1","messages":[{"role":"user","content":"hello"}],'
        '"labels":{"violation":false,"scope":"either","turn":null,"repair_expected":null,"repair_turn":null}}\n'
    )

    with pytest.raises(EvalValidationError, match="Unsupported field mappings"):
        load_jsonl_suite(
            path,
            mapping_config={
                "fields": {
                    "id": "case_id",
                    "messages": "messages",
                    "labels.violation": "labels.violation",
                    "labels.detection_scope": "labels.scope",
                    "labels.detect_at_turn": "labels.turn",
                    "labels.repair_expected": "labels.repair_expected",
                    "labels.repair_verified_at_turn": "labels.repair_turn",
                    "labels.extra": "labels.extra",
                }
            },
        )


def test_jsonl_import_splits_multi_event_conversation(tmp_path):
    path = tmp_path / "multi-event.jsonl"
    path.write_text(
        '{"conversation_id":"thread-1","messages":['
        '{"role":"user","content":"first unsafe prompt"},'
        '{"role":"assistant","content":"unsafe answer"},'
        '{"role":"user","content":"second unsafe prompt"},'
        '{"role":"assistant","content":"still unsafe"}],'
        '"events":['
        '{"violation":true,"scope":"response","turn":0,"repair_expected":null,"repair_turn":null},'
        '{"violation":true,"scope":"response","turn":1,"repair_expected":null,"repair_turn":null}'
        "]}\n"
    )

    suite = load_jsonl_suite(
        path,
        mapping_config={
            "event_path": "events",
            "fields": {
                "id": "conversation_id",
                "messages": "messages",
                "labels.violation": "violation",
                "labels.detection_scope": "scope",
                "labels.detect_at_turn": "turn",
                "labels.repair_expected": "repair_expected",
                "labels.repair_verified_at_turn": "repair_turn",
            },
        },
    )

    assert [case.id for case in suite.cases] == ["thread-1#event-0", "thread-1#event-1"]
    assert [case.labels.detect_at_turn for case in suite.cases] == [0, 1]
