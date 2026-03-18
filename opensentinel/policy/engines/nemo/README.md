# NeMo Guardrails Policy Engine

> NVIDIA NeMo Guardrails integration for input/output filtering, jailbreak detection, and content moderation.

## Overview

The NeMo engine wraps NVIDIA's [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) as a Open Sentinel `PolicyEngine`. It provides comprehensive content safety through NeMo's rail system — running each request and response through configurable input/output rails for jailbreak detection, PII filtering, toxicity checks, and hallucination detection.

**Key characteristics:**
- **Content safety** — Jailbreak detection, toxicity filtering, PII masking
- **Input + Output rails** — Bi-directional filtering (pre-call and post-call)
- **Colang integration** — Programmable dialog flows via NeMo's Colang language
- **Open Sentinel bridge** — Custom NeMo actions for logging violations and requesting interventions
- **Fail-open/closed** — Configurable behavior on evaluation errors

```
Request ──► evaluate_request() ──► NeMo Input Rails ──► ALLOW / DENY / MODIFY
                                       │
Response ──► evaluate_response() ──► NeMo Output Rails ──► ALLOW / DENY
                                       │
                                  Blocked response?
                                  ├── Yes → Decision.BLOCK + message
                                  └── No  → Decision.ALLOW
```

## Architecture

### Module Map

```
nemo/
├── __init__.py    # Package exports
└── engine.py      # NemoGuardrailsPolicyEngine — single-file implementation
```

The NeMo engine is intentionally compact — most of the heavy lifting is done by the `nemoguardrails` library itself. The engine acts as an adapter between NeMo's API and Open Sentinel' `PolicyEngine` protocol.

### How It Works

1. **Initialize**: Load a `RailsConfig` (from directory or dict) and create `LLMRails`
2. **Evaluate request**: Pass messages through NeMo's `generate_async` → check if response is blocked
3. **Evaluate response**: Append assistant message to conversation → run through output rails → check
4. **Detect blocks**: Compare NeMo's output against known blocked-response markers

---

## Configuration

### Prerequisites

```bash
pip install 'opensentinel[nemo]'
# or
pip install nemoguardrails
```

### Initialization

```python
from opensentinel.policy.engines.nemo import NemoGuardrailsPolicyEngine

engine = NemoGuardrailsPolicyEngine()

# Option 1: From config directory (contains config.yml + Colang files)
await engine.initialize({
    "config_path": "./nemo_config/"
})

# Option 2: From dict
await engine.initialize({
    "config": {
        "models": [{"type": "main", "engine": "anthropic", "model": "claude-sonnet-4-5"}],
        "rails": {"input": {"flows": ["self check input"]}}
    }
})
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | `str` | — | Path to NeMo config directory (must contain `config.yml`) |
| `config` | `dict` | — | Alternative: `RailsConfig` parameters as dict |
| `custom_actions` | `dict` | `{}` | Custom action functions to register with NeMo |
| `rails` | `list` | `["input", "output"]` | Which rails to enable |
| `fail_closed` | `bool` | `False` | If `True`, block on evaluation errors; if `False`, fail open |

### NeMo Config Directory Structure

```
nemo_config/
├── config.yml          # Main NeMo configuration
├── rails/
│   ├── input.co        # Input rail Colang flows
│   └── output.co       # Output rail Colang flows
├── prompts/
│   └── prompts.yml     # Custom prompts
└── actions/
    └── custom.py       # Custom Python actions
```

---

## Blocked Response Detection

NeMo doesn't always return a structured "blocked" signal — it often generates a natural language refusal. The engine detects blocks by matching the response against known refusal markers:

```python
BLOCKED_MARKERS = [
    "i cannot",
    "i'm not able to",
    "i am not able to",
    "refuse to",
    "[blocked]",
    "i can't help with",
    "i'm unable to",
    "sorry, but i can't",
]
```

If any marker is found (case-insensitive) in NeMo's output, the response is treated as blocked.

---

## Evaluation Flow

### `evaluate_request(session_id, request_data, context)`

Processes the request through NeMo's **input rails**:

1. Skip if `"input"` not in enabled rails → return `ALLOW`
2. Extract messages from `request_data["messages"]`
3. Call `rails.generate_async(messages=messages, ...)`
4. Check NeMo's response:
   - **Blocked** (refusal markers detected) → `DENY` with `nemo_input_blocked` violation
   - **Modified** (messages changed by NeMo) → `MODIFY` with sanitized messages
   - **Passed** → `ALLOW`
5. On error:
   - `fail_closed=True` → `DENY`
   - `fail_closed=False` → `WARN` (log and pass through)

### `evaluate_response(session_id, response_data, request_data, context)`

Processes the response through NeMo's **output rails**:

1. Skip if `"output"` not in enabled rails → return `ALLOW`
2. Extract response content (supports OpenAI format, dicts, LiteLLM objects)
3. Build full conversation: `original_messages + [{"role": "assistant", "content": response}]`
4. Call `rails.generate_async(messages=full_conversation, ...)`
5. Check NeMo's output:
   - **Blocked** → `DENY` with `nemo_output_blocked` violation and `intervention_needed`
   - **Passed** → `ALLOW`
6. On error: same fail-open/closed behavior as request evaluation

---

## Open Sentinel Bridge Actions

The engine registers two custom NeMo actions that allow Colang flows to interact with Open Sentinel:

### `sentinel_log_violation`

```colang
define flow log_violation
    $result = execute sentinel_log_violation(
        violation_name="pii_detected",
        severity="error",
        message="PII found in response"
    )
```

Logs a violation through Open Sentinel' logging system. Returns `{"logged": True, ...}`.

### `sentinel_request_intervention`

```colang
define flow request_intervention
    $result = execute sentinel_request_intervention(
        intervention_name="pii_remediation",
        context={"detected_pii": ["email", "phone"]}
    )
```

Requests a Open Sentinel intervention from within a Colang flow. Returns `{"intervention_requested": "...", "context": {...}}`.

---

## Response Content Extraction

The engine handles multiple response formats:

| Format | How Content is Extracted |
|--------|------------------------|
| `str` | Used directly |
| `dict` with `content` | `response["content"]` |
| `dict` with `choices` | `response["choices"][0]["message"]["content"]` (OpenAI format) |
| Object with `.content` | `response.content` |
| Object with `.choices` | `response.choices[0].message.content` (LiteLLM format) |

---

## Error Handling

The engine supports two failure modes:

### Fail Open (default: `fail_closed=False`)
- On evaluation error → return `WARN` with violation logged
- Request/response passes through unblocked
- Appropriate for non-critical guardrails

### Fail Closed (`fail_closed=True`)
- On evaluation error → return `DENY`
- Request/response is blocked
- Appropriate for safety-critical deployments

---

## Session Management

The NeMo engine maintains per-session contexts for debugging, but **NeMo itself manages conversation state internally** through `LLMRails`.

```python
# Get session context
state = await engine.get_session_state("session-123")

# Reset session
await engine.reset_session("session-123")

# Shutdown (clears all state)
await engine.shutdown()
```

---

## Public API

### `NemoGuardrailsPolicyEngine` (extends `PolicyEngine`)

| Method | Description |
|--------|-------------|
| `initialize(config)` | Load NeMo config, create `LLMRails`, register actions |
| `evaluate_request(session_id, request_data, context)` | Run input rails |
| `evaluate_response(session_id, response_data, request_data, context)` | Run output rails |
| `get_session_state(session_id)` | Get session debug context |
| `reset_session(session_id)` | Clear session context |
| `shutdown()` | Release all NeMo resources |

### Key Properties

| Property | Value |
|----------|-------|
| `name` | `"nemo:guardrails"` |
| `engine_type` | `"nemo"` |

---

## Comparison with Other Engines

| Feature | NeMo Engine | FSM Engine | LLM Engine |
|---------|------------|-----------|------------|
| Focus | Content safety | Workflow enforcement | Nuanced evaluation |
| Classification | N/A (content filtering) | Tool/regex/embeddings | LLM-based |
| Stateful | Minimal | Full FSM | Full with drift |
| External deps | `nemoguardrails` | None (embeddings optional) | `litellm` |
| Latency | Medium (NeMo LLM calls) | Low (~0ms local) | Medium (LLM API) |
| Best for | Safety guardrails | Deterministic workflows | Conversational policies |
