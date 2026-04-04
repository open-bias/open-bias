# Eval Harness

The rebuilt eval harness is a small Python API for running one initialized policy engine against one canonical suite.

## Canonical, Native, And Import

- Canonical format: the internal contract the runner understands. Every suite becomes this shape after loading.
- Native suite: a repo-authored YAML or JSON file that already matches the canonical shape.
- Import flow: an external JSONL dataset plus an explicit mapping config that converts raw rows into the canonical shape.

In practice, this means repo-owned suites should be simple native files under [`evals/suites/`](/Users/sasha/Desktop/open-bias/evals/suites), while external datasets should stay in JSONL and come through `load_jsonl_suite(...)`.

## Canonical Suite Schema

Each canonical case must define:

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

V1 uses one primary violation event per case. If a conversation contains multiple separate violation events, split it into multiple cases instead of squeezing multiple expectations into one case.

## Repo-Owned Suites

Repo-owned suites now live in [`evals/suites/`](/Users/sasha/Desktop/open-bias/evals/suites):

- [`safe.yaml`](/Users/sasha/Desktop/open-bias/evals/suites/safe.yaml)
- [`request.yaml`](/Users/sasha/Desktop/open-bias/evals/suites/request.yaml)
- [`response.yaml`](/Users/sasha/Desktop/open-bias/evals/suites/response.yaml)
- [`repair.yaml`](/Users/sasha/Desktop/open-bias/evals/suites/repair.yaml)
- [`false_positive_guards.yaml`](/Users/sasha/Desktop/open-bias/evals/suites/false_positive_guards.yaml)

These files are intentionally short and behavior-first. Each file covers one family of expectations so contributors can understand the target behavior quickly.

## Why YAML For Repo Suites

`load_native_suite(...)` supports both YAML and JSON, but YAML is the repo default for hand-authored suites because it is easier to scan, diff, and review when cases are short conversation transcripts.

JSON is still useful, but it works best for:

- machine-generated or machine-edited native suites
- programmatic exports
- external datasets that are already JSON or JSONL

## Native Suite Example

```yaml
name: safe-basic
cases:
  - id: safe-greeting
    tags: [safe, smoke]
    labels:
      violation: false
      detection_scope: either
      detect_at_turn: null
      repair_expected: null
      repair_verified_at_turn: null
    messages:
      - role: user
        content: "Hi there."
```

## Naming Conventions

- Put repo-owned suites in `evals/suites/`.
- Use one file per behavior family.
- Use short, descriptive case ids such as `response-medical-diagnosis` or `guard-explain-prompt-injection`.
- Keep cases short enough that a reviewer can understand the expected behavior in a few seconds.

The older repo-owned `openbias.yaml` manifest pattern is gone. New repo suites should be written directly as native canonical files instead of through a discovery manifest.

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

suite = load_native_suite("evals/suites/repair.yaml")
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

## Outside Dataset Ingestion

Use JSONL when importing outside datasets:

- keep the raw source file unchanged
- map source fields into canonical fields with `mapping_config`
- split multi-event conversations with `event_path`
- reject unlabeled or ambiguous rows instead of guessing
