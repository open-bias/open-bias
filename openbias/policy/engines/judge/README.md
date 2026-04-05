# LLM-as-a-Judge Policy Engine

> A judge evaluator that checks the latest request or response against runtime-compiled rules from project `RULES.md`.

## Overview

The judge engine is the simplest policy evaluator in Open Bias: it asks a judge model whether the current turn follows the compiled project rules. The public contract is:

- User policy lives in project `RULES.md`.
- Open Bias compiles that file into runtime `_compiled_rules`.
- The judge engine receives those compiled rules plus model settings.
- Each judge evaluates one rule at a time with a binary pass/fail result.
- The engine aggregates judges per rule and maps the final `JudgeVerdict` to an `EvaluationResult`.

The engine does not load rubric registries, named rulesets, YAML rubric files, or pairwise comparisons.

## Module Map

```text
judge/
├── engine.py       # JudgePolicyEngine entry point and PolicyEngine adapter
├── evaluator.py    # Per-rule judge calls and result construction
├── client.py       # JudgeClient model management
├── models.py       # JudgeRuleResult, AggregatedRuleResult, JudgeVerdict
├── prompts.py      # Prompt templates for compiled-rule evaluation
├── compiler.py     # Runtime compiler wiring for project RULES.md
└── __init__.py     # Exports and registration
```

## Runtime Contract

`JudgePolicyEngine.initialize()` expects:

- `models`: judge model configs
- `_compiled_rules`: non-empty list of plain-text rules
- `_rules_source`: optional source label such as `RULES.md`

Any authored policy input is resolved before engine initialization. The engine rejects legacy `rules_file` input because user-facing policy is no longer configured directly on the evaluator.

## Evaluation Flow

### Pre-call

When an evaluator is configured for `phase: pre_call`, the judge evaluates the latest user message against the compiled rules before the agent runs.

### Post-call

When an evaluator is configured for post-call use, the judge evaluates the latest model response against the same compiled rules. Conversation history and tool definitions are provided as context, but each rule is still judged independently.

### Verdict Mapping

Each turn produces:

- Per-judge `JudgeRuleResult` entries
- Per-rule `AggregatedRuleResult` entries across all configured judges
- An action of `pass`, `intervene`, or `block`
- One `ViolationRecord` per failed aggregated rule
- Summary and tracing metadata, including `rules_source`

Open Bias then maps non-`pass` verdicts into `EvaluationStatus.VIOLATION`.

## Minimal Example

```python
from openbias.policy.engines.judge import JudgePolicyEngine

engine = JudgePolicyEngine()
await engine.initialize(
    {
        "models": [
            {
                "name": "primary",
                "model": "anthropic/claude-sonnet-4-5",
                "temperature": 0.0,
            }
        ],
        "_compiled_rules": [
            "Do not reveal secrets.",
            "Stay on task.",
        ],
        "_rules_source": "RULES.md",
    }
)
```

## Extension Points

- Update `prompts.py` if the judge instructions need to change.
- Update `compiler.py` if runtime compilation into `_compiled_rules` changes.
- Keep authored policy in `RULES.md`; do not add evaluator-specific rubric files.
