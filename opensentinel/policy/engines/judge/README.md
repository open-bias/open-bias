# LLM-as-a-Judge Policy Engine

> A pluggable evaluation engine for judging agent behavior against configurable rubrics using LLMs.

## Overview

The Judge engine provides a "just tell me if this response is okay" primitive for the Open Sentinel reliability layer. Unlike the FSM engine (which enforces state transitions) or the standard LLM engine (which classifies state drift), the Judge engine evaluates the **quality** and **safety** of agent responses using a separate "judge" LLM.

It supports:
- **Turn-level evaluation**: Judging the latest response (fast, cheap).
- **Conversation-level evaluation**: Judging the entire interaction trajectory for drift, goal progression, and cumulative policy violations.
- **Reference-free & Reference-based**: Can judge with or without a "golden" answer.
- **Pointwise & Pairwise**: Can score a single response or compare two responses (A/B testing).

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
├── ensemble.py          # Multi-model aggregation (P1)
└── __init__.py          # Exports and registration
```

### Key Components

| Component | Responsibility |
|-----------|---------------|
| `JudgePolicyEngine` | Adapts the judge logic to the standard `PolicyEngine` interface. |
| `JudgeClient` | Wraps `LLMClient` to handle specific judge model interactions (e.g., JSON mode). |
| `Evaluator` | Implements the core evaluation loops: `evaluate_turn`, `evaluate_conversation`, `evaluate_pairwise`. |
| `RubricRegistry` | Manages available rubrics (built-ins like `safety`, `agent_behavior` + custom YAMLs). |
| `BiasMitigator` | Handles position randomization for pairwise comparisons to prevent positional bias. |

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
- **Scales**: Scoring systems (e.g., `binary` (pass/fail), `likert_5` (1-5), `likert_10`).

### Verdicts

The result of an evaluation is a `JudgeVerdict`:
- **Scores**: Detailed scores per criterion with reasoning and evidence.
- **Action**: The recommended policy action (`pass`, `warn`, `intervene`, `block`, `escalate`).
- **Summary**: A high-level explanation of the verdict.

## Configuration

### Initialization

```python
from opensentinel.policy.engines.judge import JudgePolicyEngine

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
    "conversation_eval_interval": 5,  # Run conversation-level rubrics every 5 turns
    "checker_mode": "async"           # "async" (post-call) or "sync" (pre-response blocking)
})
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `list` | auto-detected | List of model configs. If omitted, a `primary` model is created using the system default (auto-detected from API keys). |
| `default_rubric` | `str` | `agent_behavior` | Rubric to use if none specified in request. |
| `custom_rubrics_path` | `str` | `None` | Path to directory containing custom rubric YAMLs. |
| `checker_mode` | `str` | `async` | `async` (evaluate after response) or `sync` (block response until evaluated). |
| `pre_call_enabled` | `bool` | `false` | Whether to run an evaluation on the *user's* request (input railing). |
| `pre_call_rubric` | `str` | `safety` | Rubric to use for pre-call evaluation. |
| `ensemble_enabled` | `bool` | `false` | Whether to use multiple models and aggregate results. |
| `conversation_eval_interval` | `int` | `5` | How often (in turns) to run conversation-scope rubrics. |
| `pass_threshold` | `float` | `0.6` | Normalized score required to pass (if not using specific actions). |

## Evaluation Flow

### 1. Pre-Call (Optional)
If `pre_call_enabled` is true:
1.  **Input**: User's prompt.
2.  **Rubric**: `pre_call_rubric` (default: `safety`).
3.  **Action**: If violated, can `BLOCK` request before it reaches the agent.

### 2. Post-Call (Main)
Inside `evaluate_response()`:
1.  **Extract Context**: Gets the agent's response and full conversation history.
2.  **Turn Evaluation**:
    -   Runs enabled **turn-scope** rubrics (e.g., `agent_behavior`) using `evaluator.evaluate_turn()`.
    -   The judge sees the history but scores only the latest response.
3.  **Conversation Evaluation**:
    -   Checks if triggered (interval reached, session end, or turn warning).
    -   Runs **conversation-scope** rubrics (e.g., `conversation_policy`) using `evaluator.evaluate_conversation()`.
    -   The judge evaluates the *entire trajectory* for patterns.
4.  **Aggregation**:
    -   Combines verdicts from all run rubrics.
    -   Takes the **most restrictive** action (e.g., if Turn says `PASS` but Conversation says `WARN`, result is `WARN`).
5.  **Mapping**:
    -   Converts `VerdictAction` to Open Sentinel `EngineResult`.

### Decision Logic

| Judge Action | `Decision` | Intervention |
|--------------|------------|--------------|
| `pass` | `ALLOW` | None |
| `warn` | `ALLOW` | None (logged in metadata) |
| `intervene` | `INTERVENE` | Interceptor injects judge feedback into system prompt. |
| `block` | `BLOCK` | Interceptor returns `allowed=False`, raises `WorkflowViolationError`. |
| `escalate` | `INTERVENE` | `metadata["escalate"]=True` for human-in-the-loop handling. |

## Built-in Rubrics

| Rubric | Scope | Type | Description |
|--------|-------|------|-------------|
| `general_quality` | turn | pointwise/5-pt | Basic helpfulness, accuracy, coherence. |
| `instruction_following` | turn | pointwise/5-pt | Checks if agent followed user instructions. |
| `safety` | turn | pointwise/binary | Checks for harm, PII, unauthorized actions. |
| `agent_behavior` | turn | pointwise/5-pt | **Default**. Checks instructions, tool use, hallucinations. |
| `conversation_policy` | conversation | pointwise/5-pt | Checks goal progression, consistency, drift. |
| `comparison` | turn | pairwise/5-pt | A/B testing preference (A vs B). |

## Modes: Sync vs Async

The engine logic is identical in both modes; the difference is when it is invoked by the **Interceptor**.

-   **Async (`checker_mode: async`)**:
    -   The user receives the agent's response immediately (zero latency impact).
    -   The judge runs in the background.
    -   Violations are stored and applied as "pending interventions" on the *next* turn.

-   **Sync (`checker_mode: sync`)**:
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
