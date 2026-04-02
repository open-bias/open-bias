# LLM Policy Engine

> LLM-powered workflow enforcement with confidence scoring, drift detection, and soft constraint evaluation.

## Overview

The LLM engine is Open Bias' **intelligence-based** policy engine. Instead of relying on deterministic pattern matching, it uses a lightweight LLM (e.g., `anthropic/claude-sonnet-4-5`) as a reasoning backbone to classify states, detect behavioral drift, and evaluate soft constraints that are difficult to express as regex or tool-call matches.

**Key characteristics:**
- **LLM-powered** — Uses a sidecar LLM for nuanced classification and evaluation
- **Confidence-tiered** — Three-tier confidence system (CONFIDENT / UNCERTAIN / LOST)
- **Drift detection** — Combines temporal and semantic drift signals
- **Same internal workflow schema** — Reuses `WorkflowDefinition` from FSM engine after compilation from `rules.md`
- **Evidence memory** — Accumulates constraint evaluation evidence across turns

```
Response ──► evaluate_response()
                 │
                 ├──► LLMStateClassifier ──► Classify with confidence tiers
                 │                               │
                 ├──► DriftDetector ──► Temporal + Semantic drift scoring
                 │                          │
                 ├──► LLMConstraintEvaluator ──► Soft constraint evaluation
                 │                                    │
                 └──► InterventionHandler ──► Strategy selection + cooldown
                                                  │
                                           EvaluationResult
```

## Architecture

### Module Map

```
llm/
├── engine.py              # LLMPolicyEngine — main entry point
├── llm_client.py          # LLMClient — async LLM wrapper (litellm)
├── state_classifier.py    # LLMStateClassifier — LLM-based state classification
├── drift_detector.py      # DriftDetector — temporal + semantic drift
├── constraint_evaluator.py# LLMConstraintEvaluator — soft constraint checking
├── intervention.py        # InterventionHandler — strategy decision + application
├── models.py              # All dataclasses and enums
├── prompts.py             # LLM prompt templates
└── templates.py           # Deterministic intervention message templates
```

### Component Relationships

| Component | Responsibility | Dependencies |
|-----------|---------------|-------------|
| `LLMPolicyEngine` | Top-level `StatefulPolicyEngine` adapter | All below |
| `LLMClient` | Async JSON-structured LLM calls via `litellm` | `litellm` |
| `LLMStateClassifier` | State classification with confidence + skip detection | `LLMClient`, `WorkflowDefinition` |
| `DriftDetector` | Temporal (Levenshtein) + semantic (embedding) drift | `WorkflowDefinition`, `sentence-transformers` |
| `LLMConstraintEvaluator` | Batched soft constraint evaluation with evidence | `LLMClient`, `WorkflowDefinition` |
| `InterventionHandler` | Maps violations + drift to intervention strategies | `core.intervention.strategies` |

---

## Configuration

In `openbias.yaml`, user-facing evaluator config should only include evaluator
identity (`name`, `type`, `phase`) plus optional model/runtime tuning fields
such as `llm_model`, `temperature`, and thresholds. Workflow inputs are
compiled internally from project `rules.md`; do not set `workflow` or
`config_path` in user config.

### Initialization

```python
from openbias.policy.engines.llm import LLMPolicyEngine

engine = LLMPolicyEngine()
await engine.initialize({
    "llm_model": "anthropic/claude-sonnet-4-5",           # Sidecar LLM (default)
    "temperature": 0.0,                    # Deterministic outputs (default)
    "max_tokens": 1024,                    # Max response tokens (default)
    "timeout": 10.0,                       # LLM call timeout in seconds (default)
    "confident_threshold": 0.8,            # Threshold for CONFIDENT tier (default)
    "uncertain_threshold": 0.5,            # Threshold for UNCERTAIN tier (default)
    "temporal_weight": 0.55,               # Weight for temporal vs semantic drift (default)
    "cooldown_turns": 2,                   # Turns between interventions (default)
    "max_constraints_per_batch": 5,        # Max constraints per LLM call (default)
    # Compiled workflow data is injected by the runtime compiler.
})
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_model` | `str` | `"anthropic/claude-sonnet-4-5"` | Model for classification and constraint evaluation |
| `temperature` | `float` | `0.0` | LLM temperature (0.0 = deterministic) |
| `max_tokens` | `int` | `1024` | Max tokens per LLM response |
| `timeout` | `float` | `10.0` | LLM call timeout (seconds) |
| `confident_threshold` | `float` | `0.8` | Confidence score ≥ this → CONFIDENT |
| `uncertain_threshold` | `float` | `0.5` | Confidence score ≥ this → UNCERTAIN; below → LOST |
| `temporal_weight` | `float` | `0.55` | Weight for temporal drift (remainder goes to semantic) |
| `cooldown_turns` | `int` | `2` | Minimum turns between interventions |
| `max_constraints_per_batch` | `int` | `5` | Max constraints evaluated per LLM call |

---

## Core Concepts

### Confidence Tiers

Every state classification receives a confidence score and is bucketed into one of three tiers:

| Tier | Confidence Range | Meaning |
|------|-----------------|---------|
| `CONFIDENT` | ≥ 0.8 | High confidence; safe to proceed |
| `UNCERTAIN` | 0.5 – 0.8 | Medium confidence; may need attention |
| `LOST` | < 0.5 | Low confidence; agent may be off-track |

**Structural drift** is detected when **3 consecutive** classifications fall below the CONFIDENT threshold.

### Drift Detection

The `DriftDetector` computes a composite drift score from two independent signals:

#### Temporal Drift
- Measures divergence between the **expected** state sequence (from workflow transitions) and the **actual** sequence
- Uses **weighted Levenshtein distance** with exponential decay (recent states weighted more heavily)
- Score: `0.0` (on-track) to `1.0` (fully diverged)

#### Semantic Drift
- Compares last 5 assistant messages against an **on-policy centroid** (average embedding of all state descriptions + exemplars)
- Uses `sentence-transformers` for embeddings, cosine similarity for measurement
- Score: `0.0` (on-policy) to `1.0` (off-policy)

#### Composite Score
```
composite = temporal × temporal_weight + semantic × (1 - temporal_weight)
```

| Drift Level | Composite Score | Action |
|-------------|----------------|--------|
| `NOMINAL` | < 0.3 | No action |
| `WARNING` | 0.3 – 0.6 | Gentle nudge via `SYSTEM_PROMPT_APPEND` |
| `INTERVENTION` | 0.6 – 0.85 | Stronger correction via `USER_MESSAGE_INJECT` |
| `CRITICAL` | > 0.85 | Escalation; flagged for human review via `BLOCK` decision |

### Anomaly Flags

The drift detector also sets boolean anomaly flags:

| Flag | Meaning |
|------|---------|
| `unexpected_tool_call` | Tool called that isn't expected for the current state |
| `missing_expected_tool_call` | Expected tool for this state was not called |

### Soft Constraint Evaluation

Unlike the FSM engine's deterministic constraint evaluator, the LLM engine evaluates constraints **semantically**:

1. **Selection**: Only constraints relevant to the current state are evaluated (e.g., `NEVER` constraints are always active; `PRECEDENCE` constraints activate when the trigger state is current)
2. **Batching**: Active constraints are batched (max 5 per call) to minimize LLM round-trips
3. **Evidence memory**: High-confidence evaluations are stored in `SessionContext.constraint_memory` and fed back into subsequent evaluations for accumulated context
4. **Output**: Each constraint gets a `ConstraintEvaluation` with `violated`, `confidence`, `evidence`, and `severity`

---

## Evaluation Flow

### `evaluate_request(session_id, request_data, context)`

Called **before** the LLM call:

1. Check for a pending intervention from the previous evaluation
2. If found → parse stored config, apply it via `InterventionHandler.apply_intervention()`, return `MODIFY`
3. Otherwise → return `ALLOW`

### `evaluate_response(session_id, response_data, request_data, context)`

Called **after** the LLM call — this is the main evaluation pipeline:

1. **Extract content** — Parse assistant message text and tool call names from the response
2. **Add turn** — Append to session's sliding window (max 10 turns)
3. **Classify state** — LLM-based classification with confidence tiers and skip violation detection
4. **Compute drift** — Temporal + semantic drift scoring with anomaly detection
5. **Evaluate constraints** — Batched LLM constraint evaluation with evidence memory
6. **Decide intervention** — Map violations + drift to an intervention strategy (with cooldown)
7. **Record transition** — Update session state history
8. **Return result** — `EvaluationResult` with status, violations, and metadata

### Decision Logic

```
if any critical constraint violations:
    decision = BLOCK
elif intervention_message is not None:
    decision = INTERVENE
else:
    decision = ALLOW
```

---

## Session Management

Each session is tracked via a `SessionContext` dataclass:

```python
@dataclass
class SessionContext:
    session_id: str
    workflow_name: str
    current_state: str
    state_history: List[StateTransition]     # Full transition records
    drift_score: float                        # Latest composite drift
    violation_buffer: List[ConstraintEvaluation]
    pending_intervention: Optional[str]       # Scheduled for next request
    turn_count: int
    recent_confidences: List[float]           # Ring buffer (max 3)
    turn_window: List[Dict[str, Any]]         # Sliding window (max 10)
    constraint_memory: Dict[str, List[str]]   # Evidence per constraint
    last_intervention_turn: int               # Cooldown tracking
```

```python
# Inspect session state
state = await engine.get_session_state("session-123")
# Returns session.to_dict() with full history, drift scores, etc.

# Get state history
history = await engine.get_state_history("session-123")
# ["greeting", "identify_issue", "verify_identity"]
```

---

## Intervention Templates

The LLM engine uses **deterministic** templates (not LLM-generated) for intervention messages, defined in `templates.py`:

| Template | Use Case |
|----------|----------|
| `drift_warning` | Gentle nudge for minor drift |
| `drift_intervention` | Context reminder with state details |
| `drift_critical` | Hard block for severe drift |
| `constraint_violation` | Specific constraint violation notice |
| `structural_drift` | Multiple uncertain classifications |
| `skip_violation` | Skipped required intermediate state |
| `escalation` | Multiple issues, escalate to human |

Templates use `str.format()` with context variables from the session:

```python
"Remember: You are currently in the '{current_state}' state of the "
"'{workflow_name}' workflow. Please stay focused..."
```

---

## LLM Client

The `LLMClient` wraps `litellm.acompletion` with:

- **JSON-structured responses** — Automatic parsing with markdown fence stripping
- **Retry logic** — Configurable `max_retries` (default: 2)
- **Token tracking** — Cumulative token usage via `total_tokens_used`
- **Timeout** — Per-call timeout support

```python
from openbias.policy.engines.llm import LLMClient

client = LLMClient(model="anthropic/claude-sonnet-4-5", temperature=0.0)
result = await client.complete_json(
    system_prompt="Classify this response...",
    user_prompt="The agent said: 'Hello!'"
)
# Returns parsed JSON dict
```

---

## Public API

### `LLMPolicyEngine` (extends `StatefulPolicyEngine`)

| Method | Description |
|--------|-------------|
| `initialize(config)` | Load compiled workflow runtime config, create LLM client and all sub-components |
| `evaluate_request(session_id, request_data, context)` | Apply pending interventions |
| `evaluate_response(session_id, response_data, request_data, context)` | Full pipeline: classify → drift → constraints → intervene |
| `classify_response(session_id, response_data, current_state)` | Classify without side effects |
| `get_current_state(session_id)` | Current state name |
| `get_state_history(session_id)` | List of visited states |
| `get_valid_next_states(session_id)` | Legal transitions from current state |
| `get_session_state(session_id)` | Full session debug info (dict) |
| `reset_session(session_id)` | Delete session context |
| `shutdown()` | Clear all sessions |

### Key Properties

| Property | Value |
|----------|-------|
| `name` | `"llm:{workflow_name}"` |
| `engine_type` | `"llm"` |

---

## FSM vs LLM Engine Comparison

| Feature | FSM Engine | LLM Engine |
|---------|-----------|------------|
| Classification | Tool calls → Regex → Embeddings | LLM-based with confidence tiers |
| Constraints | Deterministic LTL-lite | LLM-evaluated with evidence memory |
| Drift detection | None (binary: legal/illegal transition) | Temporal + semantic composite scoring |
| Latency overhead | ~0ms (local) | ~100-500ms (LLM API calls) |
| Cost | Free | Per-token LLM cost |
| Handles ambiguity | No (falls back to embeddings) | Yes (confidence + reasoning) |
| Workflow schema | Internal `WorkflowDefinition` | Same internal `WorkflowDefinition` |
| Best for | Well-defined tool-based workflows | Conversational/nuanced workflows |

---

## Extension Points

### Custom Intervention Templates
Add templates to `templates.py`:

```python
DEFAULT_TEMPLATES["my_custom_warning"] = (
    "Custom warning: {details}\n"
    "Current state: {current_state}"
)
```

### Custom Drift Signals
Extend `DriftDetector` to add new drift dimensions:
1. Add a new compute method (e.g., `compute_behavioral_drift`)
2. Incorporate into `compute_drift()` composite calculation

### Adjusting LLM Prompts
Modify `prompts.py` to change how the LLM classifies states or evaluates constraints. The templates use `str.format()` with workflow and session context variables.
