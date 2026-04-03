# Evaluator Engines

Four engines, same interface (`PolicyEngine` protocol). Configure one or more as evaluators based on your enforcement needs.

## How to choose

```
Do you need deterministic, auditable enforcement?
  ├─ Yes → Is behavior defined by tool calls and sequencing?
  │         ├─ Yes → FSM engine (zero LLM cost, ~0ms)
  │         └─ No  → LLM engine (handles ambiguous workflows)
  └─ No  → Is the concern content quality / safety / rules compliance?
            ├─ Yes → Judge engine (async default, 0ms critical-path)
            └─ No  → Do you want NeMo-style guardrails and Colang flows?
                      ├─ Yes → NeMo engine
                      └─ No  → LLM engine (handles ambiguous workflows)
```

## Comparison

| Property | Judge | FSM | LLM | NeMo |
|----------|-------|-----|-----|------|
| What it does | Evaluates `rules.md`-compiled rules one at a time with a separate LLM | Enforces state machine workflows with temporal constraints | Classifies state and detects drift using a sidecar LLM | Runs NVIDIA NeMo Guardrails input/output rails |
| Deterministic | No | Yes | No | No |
| Requires LLM calls | Yes (judge model) | No (tool calls, regex, local embeddings) | Yes (classification + constraint eval) | Yes (NeMo's LLM) |
| Stateful | Per-turn with optional session context | Full FSM with state history | Full with drift tracking and evidence memory | Minimal (NeMo manages internally) |
| Latency overhead | **0ms critical-path** (async default); 200-800ms total in background | ~0ms (local computation) | 100-500ms (LLM API calls) | 200-800ms (NeMo LLM calls) |
| External deps | None beyond litellm | sentence-transformers (optional, for embedding fallback) | litellm, sentence-transformers | nemoguardrails |
| Best for | Content quality, safety screening, rules compliance | Well-defined tool-based workflows with ordering requirements | Conversational workflows where classification is ambiguous | Jailbreak detection, PII filtering, content moderation |

## Judge Engine

**Evaluator type**: `judge`

Uses an LLM to evaluate compiled rules from project `rules.md`. At startup, Open Bias compiles authored policy into runtime `_compiled_rules`. For each turn, the judge engine evaluates one compiled rule at a time against the current message and produces a binary pass/fail result. Failed aggregated rules are returned as violations and then mapped to the configured `fail_action`.

### When to use it

- Enforcing content quality standards (professional tone, accuracy, helpfulness)
- Safety screening (PII leakage, harmful content, unauthorized actions)
- Rules compliance where criteria are qualitative rather than structural
- Cases where you want human-readable reasoning for every decision

### How it works

1. Open Bias compiles project `rules.md` into runtime `_compiled_rules`
2. On each evaluation, the engine picks the current message under review: latest user message for `phase: pre_call`, latest assistant response for `phase: post_call`
3. For each compiled rule, one or more judge models evaluate that rule independently and return binary pass/fail results
4. If multiple judges are configured, their results are aggregated per rule with `aggregation_mode: majority` by default
5. Each failed aggregated rule is recorded as a violation, and the evaluator's `fail_action` determines whether Open Bias intervenes, blocks, or shadows

### Evaluation phases

- **`phase: pre_call`**: Evaluates the latest user message before the upstream model runs.
- **`phase: post_call`**: Evaluates the latest assistant response after the upstream model returns.

### Sync vs async modes

- **Async** (default): The agent's response reaches the user immediately. The judge runs in the background. Violations are applied as interventions on the next turn. Zero latency impact.
- **Sync**: The response is held until the judge finishes. Violations are applied immediately. Adds one LLM round-trip of latency.

### Multi-judge

When multiple judge models are configured, they evaluate the same compiled rule in parallel. The engine aggregates those binary rule results with `aggregation_mode`, which defaults to `majority` and also supports `all` and `any`. This keeps multi-judge behavior per-rule and binary.

### Minimal config

```yaml
evaluators:
  - name: content-policy
    type: judge
```

Full configuration reference: [docs/configuration.md](configuration.md#judge-engine)

`openbias serve` compiles project `rules.md` into judge runtime config automatically at startup.

Deep dive: [openbias/policy/engines/judge/README.md](../openbias/policy/engines/judge/README.md)

---

## FSM Engine

**Evaluator type**: `fsm`

Models allowed agent behavior as a finite state machine compiled internally from project `rules.md`. It classifies each LLM response to a workflow state using a three-tier cascade (tool call matching, regex, semantic embeddings), evaluates temporal constraints based on LTL-lite, and triggers interventions on violations.

### When to use it

- Multi-step workflows with defined ordering (onboarding, support tickets, refund flows)
- Processes where certain steps must precede others (verify identity before refund)
- Cases where you need deterministic, auditable enforcement with zero LLM overhead
- Tool-heavy agents where state is clearly indicated by which tools are called

### How it works

1. `openbias serve` compiles your authored `rules.md` into an internal workflow definition
2. On each agent response, the classifier determines which workflow state the response belongs to
3. The constraint evaluator checks all active temporal constraints against the state history
4. If a constraint is violated, the intervention handler schedules a correction for the next turn
5. The state machine records the transition

### Classification cascade

The classifier tries three methods in order, stopping at the first confident match:

| Method | Signal | Confidence | Latency |
|--------|--------|------------|---------|
| Tool call matching | Function/tool names in the response | 1.0 | ~0ms |
| Regex patterns | `re.search()` against state patterns | 0.85 | ~1ms |
| Semantic embeddings | Cosine similarity via sentence-transformers | Proportional to similarity | ~50ms |

### Constraint types (LTL-lite)

| Type | Semantics | Example |
|------|-----------|---------|
| `precedence` | B must occur before A | Verify identity before refund |
| `never` | State must never occur | Never share internal info |
| `eventually` | State must eventually be reached | Must reach resolution |
| `response` | If A occurs, B must eventually follow | If complaint, must acknowledge |

### Intervention strategies

Intervention templates in the compiled workflow support strategy prefixes:

| Prefix | Strategy | Effect |
|--------|----------|--------|
| *(none)* | `SYSTEM_PROMPT_APPEND` | Appends guidance to the system message |
| `inject:` | `USER_MESSAGE_INJECT` | Inserts as a user message |

### Minimal config

```yaml
evaluators:
  - name: workflow-guard
    type: fsm
```

Full configuration reference: [docs/configuration.md](configuration.md#fsm-engine)

`openbias serve` compiles project `rules.md` into FSM runtime workflow config automatically at startup.

Deep dive: [openbias/policy/engines/fsm/README.md](../openbias/policy/engines/fsm/README.md)

---

## LLM Engine

**Evaluator type**: `llm`

Uses a lightweight sidecar LLM for state classification, drift detection, and soft constraint evaluation. Like the other engines, it compiles from project-local `rules.md` at startup, then runs against the generated runtime workflow internally. It trades determinism for the ability to handle ambiguous, conversational workflows where tool calls and regex are insufficient.

### When to use it

- Conversational workflows where state boundaries are fuzzy
- Cases where constraints are qualitative ("the agent should acknowledge the customer's frustration")
- When you want drift detection that considers semantic similarity, not just state transitions
- As a drop-in upgrade from FSM when classification accuracy matters more than latency

### How it works

1. On each response, the sidecar LLM classifies the response to a workflow state with a confidence score
2. Confidence is bucketed into three tiers: CONFIDENT (>= 0.8), UNCERTAIN (0.5-0.8), LOST (< 0.5)
3. The drift detector computes a composite score from temporal drift (weighted Levenshtein distance between expected and actual state sequences) and semantic drift (cosine similarity of recent messages against an on-policy centroid)
4. Active constraints are batched and sent to the LLM for evaluation, with evidence from previous evaluations included for context
5. The intervention handler maps violations and drift levels to strategies, with cooldown to prevent repeated interventions

### Key differences from FSM

| Aspect | FSM | LLM |
|--------|-----|-----|
| Classification | Tool calls, regex, embeddings (local) | LLM-based with confidence tiers |
| Constraints | Deterministic evaluation | LLM-evaluated with evidence memory |
| Drift detection | Binary (legal/illegal transition) | Continuous composite score (temporal + semantic) |
| Latency | ~0ms | 100-500ms per turn |
| Cost | Free | Per-token LLM cost |
| Ambiguity handling | Falls back to embeddings | Reasons about context |

### Minimal config

```yaml
evaluators:
  - name: llm-guard
    type: llm
    model: anthropic/claude-sonnet-4-5
```

`openbias serve` compiles project `rules.md` into a `WorkflowDefinition` automatically at startup.

Full configuration reference: [docs/configuration.md](configuration.md#llm-engine)

Deep dive: [openbias/policy/engines/llm/README.md](../openbias/policy/engines/llm/README.md)

---

## NeMo Guardrails Engine

**Evaluator type**: `nemo`

Wraps NVIDIA's [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) as an evaluator engine. Runs requests through input rails (pre-call) and responses through output rails (post-call) for jailbreak detection, PII filtering, toxicity checks, and programmable dialog flows via Colang, while keeping `rules.md` as the only authored policy source.

### When to use it

- You need jailbreak detection or content moderation
- You want PII masking or toxicity filtering
- You want programmable dialog flows using Colang

### How it works

1. On request: messages are passed through NeMo's input rails. If NeMo generates a refusal response, the request is blocked.
2. On response: the full conversation (including the agent's response) is passed through NeMo's output rails. If NeMo blocks it, the response is denied.
3. Block detection works by matching NeMo's output against known refusal markers ("i cannot", "i'm not able to", etc.).

### Prerequisites

```bash
pip install 'openbias[nemo]'
```

### Fail-open vs fail-closed

- **Fail-open** (default): If NeMo evaluation errors, the request/response passes through with a warning logged.
- **Fail-closed** (`fail_closed: true`): If NeMo evaluation errors, the request/response is blocked.

### Bridge actions

The engine registers two custom NeMo actions for use in Colang flows:
- `openbias_log_violation`: Logs a violation through OpenBias's logging system
- `openbias_request_intervention`: Requests an OpenBias intervention from within a Colang flow

### Minimal config

```yaml
evaluators:
  - name: nemo-rails
    type: nemo
```

Full configuration reference: [docs/configuration.md](configuration.md#nemo-guardrails-engine)

`openbias serve` compiles these rules into NeMo runtime rails config automatically at startup.

Deep dive: [openbias/policy/engines/nemo/README.md](../openbias/policy/engines/nemo/README.md)


