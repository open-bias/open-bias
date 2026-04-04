# Eval Harness

The rebuilt eval harness is a small Python API for running one initialized policy engine against one canonical suite.

## Canonical Suite Schema

Hand-authored suites live in YAML or JSON and normalize into `EvalSuite` plus `EvalCase` entries.

Each case must define:

- `id`
- `messages`
- `tags`
- optional `source`
- `labels`

`labels` is explicit and fixed in v1:

- `violation: bool`
- `detection_scope: request | response | either`
- `detect_at_turn: int | null`
- `repair_expected: bool | null`
- `repair_verified_at_turn: int | null`

V1 uses one primary violation event per case. If a source conversation contains multiple separate violation events, split it into multiple canonical cases or import it through the JSONL adapter with `event_path` so the adapter emits one case per event.

## Native Suite Example

```yaml
name: smoke-suite
cases:
  - id: safe-request
    tags: [safe]
    labels:
      violation: false
      detection_scope: either
      detect_at_turn: null
      repair_expected: null
      repair_verified_at_turn: null
    messages:
      - role: user
        content: "Hello"
```

## JSONL Import Mapping

External datasets stay in their raw JSONL form. Provide a mapping config instead of rewriting the source file.

```python
from openbias.eval import load_jsonl_suite

suite = load_jsonl_suite(
    "external.jsonl",
    suite_name="external-import",
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
)
```

If a conversation row contains multiple labeled events, add:

```python
"event_path": "events"
```

The adapter will emit `row-id#event-0`, `row-id#event-1`, and so on.

The import layer does not guess missing labels. If the dataset cannot explicitly supply violation truth, detection target, or repair expectation, validation fails.

## Running From Python

```python
from openbias.eval import EvalRunner, load_native_suite

suite = load_native_suite("tests/eval/fixtures/recovery_suite.yaml")
runner = EvalRunner()
result = await runner.run(engine, suite)
```

`EvalRunResult.summary` exposes only the v1 binary metrics:

- `true_positive`
- `false_negative`
- `false_positive`
- `true_negative`
- `detection_recall`
- `false_positive_rate`
- `fix_success_count`
- `fix_failure_count`
- `fix_rate`
- `exact_case_pass_rate`

## Converting Existing Repo Scenarios

Older repo conversations such as [`evals/judge/recovery_after_intervention.json`](/Users/sasha/Desktop/open-bias/evals/judge/recovery_after_intervention.json) should be copied into canonical cases with explicit labels. The repo fixture [`tests/eval/fixtures/recovery_suite.yaml`](/Users/sasha/Desktop/open-bias/tests/eval/fixtures/recovery_suite.yaml) shows that migration pattern for a recovery case.

## Outside Dataset Ingestion

Use JSONL when importing outside datasets:

- keep the raw source file unchanged
- map source fields into canonical fields with `mapping_config`
- split multi-event conversations with `event_path`
- reject unlabeled or ambiguous rows instead of guessing
