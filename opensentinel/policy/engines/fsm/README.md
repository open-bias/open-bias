# FSM Policy Engine

> Deterministic workflow enforcement using finite state machines with LTL-lite temporal constraints.

## Overview

The FSM engine is Open Sentinel' **deterministic** policy engine. It models allowed agent behavior as a finite state machine defined in YAML, then classifies each LLM response to a workflow state, evaluates temporal constraints, and triggers interventions when deviations are detected.

**Key characteristics:**
- **Zero LLM overhead** — Classification uses tool-call matching, regex, and local embeddings (no external API calls)
- **Deterministic** — Same input always produces the same constraint evaluation
- **Formally grounded** — Constraints based on LTL-lite temporal logic
- **Stateful** — Tracks full state history per session, implements `StatefulPolicyEngine`

```
Request ──► evaluate_request() ──► Check pending interventions
                                       │
Response ──► evaluate_response() ──► Classify ──► Check Constraints ──► Transition
                                       │               │                    │
                                    StateClassifier  ConstraintEvaluator  StateMachine
                                       │               │                    │
                                    ◄──────────── EngineResult ─────────────┘
```

## Architecture

### Module Map

```
fsm/
├── engine.py           # FSMPolicyEngine — main entry point
├── classifier.py       # StateClassifier — response → state classification

├── intervention.py     # InterventionHandler + InterventionBuilder
├── compiler.py         # FSMCompiler — natural language → workflow YAML
└── workflow/
    ├── schema.py       # WorkflowDefinition, State, Transition, Constraint (Pydantic)
    ├── parser.py       # WorkflowParser — YAML/JSON → WorkflowDefinition
    ├── state_machine.py# WorkflowStateMachine — session + transition management
    └── constraints.py  # ConstraintEvaluator — LTL-lite constraint checking
```

### Component Relationships

| Component | Responsibility | Dependencies |
|-----------|---------------|-------------|
| `FSMPolicyEngine` | Top-level `PolicyEngine` adapter | All below |
| `StateClassifier` | Cascade classification (tools → regex → embeddings) | `schema.State` |
| `WorkflowStateMachine` | Session lifecycle, state transitions | `schema.WorkflowDefinition` |
| `ConstraintEvaluator` | LTL-lite constraint evaluation | `schema.Constraint`, `SessionState` |
| `InterventionHandler` | Prompt injection for corrections | `core.intervention.strategies` |

| `FSMCompiler` | NL → YAML via LLM | `schema.*`, `LLMClient` |

---

## Configuration

### Initialization

```python
from opensentinel.policy.engines.fsm import FSMPolicyEngine

engine = FSMPolicyEngine()
await engine.initialize({
    "config_path": "./workflows/customer_support.yaml"  # Path to YAML/JSON
})
```

Or with an inline workflow dict:

```python
await engine.initialize({
    "workflow": {
        "name": "support-flow",
        "version": "1.0",
        "states": [...],
        "transitions": [...],
        "constraints": [...],
        "interventions": {...},
    }
})
```

### Workflow YAML Schema

```yaml
name: customer-support
version: "1.0"
description: Customer support agent workflow

states:
  - name: greeting
    description: Initial greeting to the customer
    is_initial: true
    classification:
      patterns: ["hello", "hi there", "how can I help"]
      exemplars: ["Hello! How can I assist you today?"]

  - name: identify_issue
    description: Understanding the customer's problem
    classification:
      patterns: ["what issue", "tell me more", "can you describe"]
      exemplars: ["Could you tell me more about the issue?"]

  - name: verify_identity
    description: Verifying customer identity
    classification:
      tool_calls: ["lookup_customer", "verify_identity"]

  - name: process_refund
    description: Processing a refund
    classification:
      tool_calls: ["process_refund", "issue_refund"]

  - name: resolution
    description: Wrapping up the interaction
    is_terminal: true
    classification:
      patterns: ["resolved", "anything else", "is there anything"]

transitions:
  - from_state: greeting
    to_state: identify_issue
  - from_state: identify_issue
    to_state: verify_identity
  - from_state: verify_identity
    to_state: process_refund
  - from_state: process_refund
    to_state: resolution

constraints:
  - name: verify_before_refund
    type: precedence
    trigger: process_refund
    target: verify_identity
    severity: critical
    intervention: prompt_verify_first
    description: Must verify identity before processing refunds

  - name: no_internal_info
    type: never
    target: share_internal_info
    severity: error
    intervention: warn_internal_info
    description: Never share internal system information

  - name: must_resolve
    type: eventually
    target: resolution
    severity: warning
    description: Session must eventually reach resolution

interventions:
  prompt_verify_first: |
    You must verify the customer's identity before processing any refund.
    Please ask for their account number or order ID first.
  warn_internal_info: |
    block:Do not share internal system information with the customer.
```

### Constraint Types (LTL-Lite)

| Type | Semantics | Example |
|------|-----------|---------|
| `precedence` | Target must occur BEFORE trigger | Verify identity before refund |
| `never` | Target state must never occur | Never share internal info |
| `eventually` | Target must eventually be reached | Must reach resolution |
| `always` | Property must hold at all times | Always be professional |
| `response` | If trigger occurs, target must eventually follow | If complaint → must acknowledge |
| `until` | First condition holds until second occurs | Maintain formal tone until escalation |
| `next` | Target must be the immediate next state | After greeting, must identify issue |

### Intervention Prefixes

Intervention templates support strategy prefixes:

| Prefix | Strategy | Behavior |
|--------|----------|----------|
| *(none)* | `SYSTEM_PROMPT_APPEND` | Appends guidance to system message |
| `inject:` | `USER_MESSAGE_INJECT` | Inserts as user message |
| `remind:` | `CONTEXT_REMINDER` | Inserts as assistant context |
| `block:` | `HARD_BLOCK` | Raises `WorkflowViolationError` |

---

## Classification Cascade

The `StateClassifier` uses a three-tier classification cascade, stopping at the first confident match:

### 1. Tool Call Matching (highest priority)
- **Method**: Exact match on function/tool names
- **Speed**: Instant
- **Confidence**: `1.0`
- **When**: Response includes tool calls matching a state's `classification.tool_calls`

### 2. Regex Pattern Matching
- **Method**: `re.search()` against `classification.patterns`
- **Speed**: ~1ms
- **Confidence**: `0.85`
- **When**: Response content matches a state's regex patterns

### 3. Semantic Embedding Similarity
- **Method**: Cosine similarity via `sentence-transformers`
- **Speed**: ~50ms (lazy-loaded model)
- **Confidence**: Proportional to similarity score
- **When**: Falls through to compare against `classification.exemplars`

```python
# Direct classification (outside engine context)
from opensentinel.policy.engines.fsm import StateClassifier

classifier = StateClassifier(workflow.states)
result = classifier.classify(llm_response, current_state="greeting")
# ClassificationResult(state_name="identify_issue", confidence=0.92, method="pattern")
```

---

## Evaluation Flow

### `evaluate_request(session_id, request_data, context)`

Called **before** the LLM call. Checks for pending interventions from previous violations:

1. Look up session state
2. If a pending intervention exists → return `MODIFY` with `intervention_needed`
3. Otherwise → return `ALLOW`

### `evaluate_response(session_id, response_data, request_data, context)`

Called **after** the LLM call. This is where the core logic runs:

1. **Classify** the response to determine which workflow state it represents
2. **Evaluate constraints** against the proposed state transition
3. **Attempt transition** in the state machine
4. **Determine decision**:
   - No violations → `ALLOW`
   - Violations with intervention → `WARN` (schedules intervention for next call)
   - Critical violations → `DENY`

---

## Session Management

Each session maintains:
- **Current state**: Where the agent is in the workflow
- **State history**: Full sequence of transitions with timestamps
- **Pending intervention**: Intervention to apply on next request
- **Constraint violations**: Accumulated violation records

```python
# Inspect session state
state = await engine.get_session_state("session-123")
# {
#   "current_state": "verify_identity",
#   "history": ["greeting", "identify_issue", "verify_identity"],
#   "pending_intervention": null,
#   "constraint_violations": [],
#   "workflow": "customer-support",
#   "created_at": "2026-02-15T...",
#   "last_updated": "2026-02-15T...",
# }

# Query valid transitions
next_states = await engine.get_valid_next_states("session-123")
# ["process_refund"]
```

---

## Policy Compiler

The `FSMCompiler` converts natural language policy descriptions into `WorkflowDefinition` YAML:

```python
from opensentinel.policy.engines.fsm import FSMCompiler

compiler = FSMCompiler()
result = await compiler.compile(
    "The agent must greet the customer, identify their issue, "
    "verify their identity before processing any refund, "
    "and never share internal system information."
)

if result.success:
    compiler.export(result, Path("workflow.yaml"))
```

---

## Public API

### `FSMPolicyEngine` (extends `StatefulPolicyEngine`)

| Method | Description |
|--------|-------------|
| `initialize(config)` | Load workflow from path or dict |
| `evaluate_request(session_id, request_data, context)` | Check pending interventions |
| `evaluate_response(session_id, response_data, request_data, context)` | Classify + constrain + transition |
| `classify_response(session_id, response_data, current_state)` | Classify without side effects |
| `get_current_state(session_id)` | Current state name |
| `get_state_history(session_id)` | List of visited states |
| `get_valid_next_states(session_id)` | Legal transitions from current state |
| `get_session_state(session_id)` | Full session debug info |
| `reset_session(session_id)` | Reset to initial state |
| `shutdown()` | Cleanup resources |

### Key Properties

| Property | Value |
|----------|-------|
| `name` | `"fsm:{workflow_name}"` |
| `engine_type` | `"fsm"` |

---

## Extension Points

### Adding Classification Methods
Extend `StateClassifier` and add new methods to the cascade in the `classify()` method.

### Adding Constraint Types
1. Add to `ConstraintType` enum in `workflow/schema.py`
2. Add validation in `Constraint.validate_constraint_params()`
3. Implement evaluation in `ConstraintEvaluator._evaluate_constraint()`
4. Add message formatting in `_format_violation_message()`

### Custom Intervention Strategies
Use `InterventionBuilder` to create interventions programmatically:

```python
from opensentinel.policy.engines.fsm import InterventionBuilder
from opensentinel.core.intervention.strategies import StrategyType

builder = InterventionBuilder()
builder.add(
    name="custom_warning",
    template="Please follow the correct procedure: {details}",
    strategy=StrategyType.SYSTEM_PROMPT_APPEND,
    max_applications=3,
    escalation=StrategyType.HARD_BLOCK,
)
configs = builder.build()
```
