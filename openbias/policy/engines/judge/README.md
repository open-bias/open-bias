# LLM-as-a-Judge Policy Engine

> A pluggable evaluation engine for judging agent behavior against configurable rubrics using LLMs.

## Overview

The Judge engine provides a "just tell me if this response is okay" primitive for the Open Bias reliability layer. Unlike the FSM engine (which enforces state transitions) or the standard LLM engine (which classifies state drift), the Judge engine evaluates the **quality** and **safety** of agent responses using a separate "judge" LLM.

It supports:
- **Turn-level evaluation**: Judging the latest response (fast, cheap).
- **Conversation-level evaluation**: Judging the entire interaction trajectory for drift, goal progression, and cumulative policy violations.
- **Pointwise evaluation**: Scores a single response against rubric criteria.

## Architecture

### Module Map

```
judge/
├── engine.py            # JudgePolicyEngine — top-level PolicyEngine impl
├── evaluator.py         # Core single-judge evaluation logic
├── rubrics.py           # RubricRegistry + built-in rubrics
├── client.py            # JudgeClient — manages judge LLM interactions
├── models.py            # Data types: Rubric, JudgeScore, JudgeVerdict
├── prompts.py           # Prompt templates for evaluation
├── bias.py              # Position randomization (pairwise bias mitigation)
└── __init__.py          # Exports and registration
```

### Key Components

| Component | Responsibility |
|-----------|---------------|
| `JudgePolicyEngine` | Adapts the judge logic to the standard `PolicyEngine` interface. |
| `JudgeClient` | Wraps `LLMClient` to handle specific judge model interactions (e.g., JSON mode). |
| `Evaluator` | Implements the core evaluation loops: `evaluate_turn`, `evaluate_conversation`, `evaluate_pairwise`. |
| `RubricRegistry` | Manages available rubrics (built-ins like `safety`, `agent_behavior` + custom YAMLs). |
| `bias` (module) | Position randomization functions for pairwise comparisons to prevent positional bias. |

## Core Concepts

### Evaluation Scopes

The engine distinguishes between **what** is being judged:

- **Turn Scope (`scope="turn"`)**:
  - **Target**: The latest agent response.
  - **Context**: The conversation history (provided as background).
  - **Use Case**: checking instruction following, immediate safety, tool use correctness.
  - **Frequency**: Runs on every turn (typically).

- **Conversation Scope (`scope="conversation"`)**:
  - **Target**: The entire conversation history.
  - **Use Case**: Detecting gradual drift, inconsistency, goal abandonment, or cumulative safety risks.
  - **Frequency**: Runs periodically (e.g., every N turns or at session end) to save costs.

### Rubrics and Criteria

A **Rubric** defines *how* to judge. It contains a list of **Criteria**.

- **Rubrics**: Named collections of criteria (e.g., `safety`, `helpfulness`, `agent_behavior`).
- **Criteria**: Individual dimensions to score (e.g., "Instruction Following", "No PII Leaked").
- **Scales**: Scoring systems (e.g., `binary` (pass/fail), `likert_5` (1-5)).

### Verdicts

The result of an evaluation is a `JudgeVerdict`:
- **Scores**: Detailed scores per criterion with reasoning and evidence.
- **Action**: The recommended policy action (`pass`, `intervene`, `block`).
- **Summary**: A high-level explanation of the verdict.

## Configuration

In `openbias.yaml`, user-facing evaluator config should only declare evaluator
identity (for example `name`, `type`, and `phase`). Rule content is always
loaded from project `rules.md` and compiled internally before engine init.

### Initialization

```python
from openbias.policy.engines.judge import JudgePolicyEngine

engine = JudgePolicyEngine()
await engine.initialize({
    "models": [
        {
            "name": "primary",
            "model": "anthropic/claude-sonnet-4-5",
            "temperature": 0.0
        }
    ],
    "default_rubric": "agent_behavior",
    # Note: sync/async mode and phase routing are set at the top level
    # via the evaluator config in openbias.yaml
})
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `list` | -- | List of model configs. Must be provided or set via `judge.model` in YAML. |
| `default_rubric` | `str` | `agent_behavior` | Rubric to use if none specified in request. |
| `custom_rubrics_path` | `str` | `None` | Path to directory containing custom rubric YAMLs. |

## Evaluation Flow

### 1. Pre-Call (Optional)
To run the judge engine at pre-call phase, configure a separate evaluator entry
with `phase: pre_call` in your `openbias.yaml`. The engine will evaluate the
user's prompt against its configured rubric and can `BLOCK` the request before
it reaches the agent.

### 2. Post-Call (Main)
Inside `evaluate_response()`:
1.  **Extract Context**: Gets the agent's response and full conversation history.
2.  **Turn Evaluation**:
    -   Runs enabled **turn-scope** rubrics (e.g., `agent_behavior`) using `evaluator.evaluate_turn()`.
    -   The judge sees the history but scores only the latest response.
3.  **Conversation Evaluation**:
    -   Checks if triggered (interval reached, session end, or turn warning).
    -   Runs **conversation-scope** rubrics (e.g., `conversation_rules`) using `evaluator.evaluate_conversation()`.
    -   The judge evaluates the *entire trajectory* for patterns.
4.  **Aggregation**:
    -   Combines verdicts from all run rubrics.
    -   Takes the **most restrictive** action (e.g., if Turn says `PASS` but Conversation says `INTERVENE`, result is `INTERVENE`).
5.  **Mapping**:
    -   Converts `VerdictAction` to Open Bias `EvaluationResult`.

### Decision Logic

| Judge Action | `Decision` | Intervention |
|--------------|------------|--------------|
| `pass` | `ALLOW` | None |
| `intervene` | `INTERVENE` | Interceptor injects judge feedback into system prompt. |
| `block` | `BLOCK` | Interceptor returns `allowed=False`, raises `WorkflowViolationError`. |

## Built-in Rubrics

| Rubric | Scope | Type | Description |
|--------|-------|------|-------------|
| `safety` | turn | pointwise/binary | Checks for harm, PII, unauthorized actions. |
| `agent_behavior` | turn | pointwise/5-pt | **Default**. Checks instructions, tool use, hallucinations. |
| `conversation_rules` | conversation | pointwise/5-pt | Checks goal progression, consistency, drift. |

## Modes: Sync vs Async

The engine logic is identical in both modes; the difference is when it is invoked by the **Interceptor**.

-   **Async (`mode: async`)**:
    -   The user receives the agent's response immediately (zero latency impact).
    -   The judge runs in the background.
    -   Violations are stored and applied as "pending interventions" on the *next* turn.

-   **Sync (`mode: sync`)**:
    -   The agent's response is held back.
    -   The judge evaluates it.
    -   If `BLOCK` or `MODIFY`, the user sees the intervention immediately.
    -   Adds latency (LLM round-trip time).

## Extension Points

### Custom Rubrics
Create a YAML file in your `custom_rubrics_path`:
```yaml
name: "brand_voice"
description: "Checks for adherence to brand guidelines"
scope: "turn"
evaluation_type: "pointwise"
criteria:
  - name: "tone"
    description: "Is the tone professional yet friendly?"
    scale: "likert_5"
    weight: 1.0
```

### Custom Prompts
Modify `prompts.py` to change the system instructions given to the judge LLM. Templates use standard Python `str.format()` syntax.
