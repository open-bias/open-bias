# Repo-Owned Eval Suites

This directory contains the repo-owned eval suites used to test a shared project policy across different engines or model configurations.

## Relationship To Rules

- `rules.md` defines the policy.
- Each suite file defines labeled examples and expected outcomes for that policy.
- The same suites should be runnable against different engines as long as those engines are configured to enforce the same policy.

## Terms

- Canonical format: the internal schema every eval suite must satisfy after loading.
- Native suite: a repo-authored YAML or JSON file that already matches the canonical schema.
- Import flow: an external JSONL dataset plus an explicit field mapping that converts each row into the canonical schema.
- Policy target: suite-level metadata that points at the rule source the suite is designed to evaluate.

## Conventions

- Use short YAML files for repo-owned suites because they are easier to read and review by hand.
- Keep one suite file per behavior family.
- Keep each case short and explicit.
- Model one primary violation event per case.
- Point every repo-owned suite at the shared project `rules.md` unless there is a deliberate reason to target a different policy source.

## First-Wave Suites

- `safe.yaml`
- `request.yaml`
- `response.yaml`
- `repair.yaml`
- `false_positive_guards.yaml`

## Loading The Suite Library

```python
from openbias.eval import load_native_suites

suites = load_native_suites("evals/suites")
```

That returns the repo-owned suites with their policy metadata intact, which makes it easy to run the same suite library across multiple engines.
