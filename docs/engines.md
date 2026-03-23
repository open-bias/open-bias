# Policy Engines

Four engines, same interface (`PolicyEngine` protocol). Pick one based on your enforcement needs.

## How to choose

```
Do you need deterministic, auditable enforcement?
  ├─ Yes → Is behavior defined by tool calls and sequencing?
  │         ├─ Yes → FSM engine (zero LLM cost, ~0ms)
  │         └─ No  → LLM engine (handles ambiguous workflows)
  └─ No  → Is the concern content quality / safety / policy compliance?
            ├─ Yes → Judge engine (async default, 0ms critical-path)
            └─ No  → Do you already have NeMo Guardrails configs?
                      ├─ Yes → NeMo engine
                      └─ No  → LLM engine (handles ambiguous workflows)
```

## Comparison

| Property | Judge | FSM | LLM | NeMo |
|----------|-------|-----|-----|------|
| What it does | Scores responses against rubrics using a separate LLM | Enforces state machine workflows with temporal constraints | Classifies state and detects drift using a sidecar LLM | Runs NVIDIA NeMo Guardrails input/output rails |
| Deterministic | No | Yes | No | No |
| Requires LLM calls | Yes (judge model) | No (tool calls, regex, local embeddings) | Yes (classification + constraint eval) | Yes (NeMo's LLM) |
| Stateful | Per-turn + periodic conversation eval | Full FSM with state history | Full with drift tracking and evidence memory | Minimal (NeMo manages internally) |
| Latency overhead | **0ms critical-path** (async default); 200-800ms total in background | ~0ms (local computation) | 100-500ms (LLM API calls) | 200-800ms (NeMo LLM calls) |
| External deps | None beyond litellm | sentence-transformers (optional, for embedding fallback) | litellm, sentence-transformers | nemoguardrails |
| Best for | Content quality, safety screening, policy compliance | Well-defined tool-based workflows with ordering requirements | Conversational workflows where classification is ambiguous | Jailbreak detection, PII filtering, content moderation |

## Judge Engine

**Config key**: `engine: judge`

Uses an LLM to evaluate every agent response against configurable rubrics. The judge sees the conversation history and scores the response on multiple criteria (tone, safety, instruction following, etc.), then maps the aggregate score to an action: pass, warn, intervene, or block.

### When to use it

- Enforcing content quality standards (professional tone, accuracy, helpfulness)
- Safety screening (PII leakage, harmful content, unauthorized actions)
- Policy compliance where rules are qualitative rather than structural
- Cases where you want human-readable reasoning for every decision

### How it works

1. After the agent responds, the judge LLM receives the conversation history and the response
2. It scores the response on each criterion in the active rubric (binary pass/fail or 5-point Likert)
3. Binary rubrics: any criterion failure triggers the configured `fail_action` (block or intervene)
4. Likert rubrics: scores are normalized and aggregated; the aggregate maps to an action based on thresholds
5. Optionally, a conversation-level rubric runs every N turns to catch gradual drift

### Evaluation scopes

- **Turn scope**: Scores the latest response only. Runs every turn. Checks instruction following, safety, tool use correctness.
- **Conversation scope**: Scores the full conversation trajectory. Runs periodically (default: every 5 turns). Catches drift, inconsistency, goal abandonment.

### Built-in rubrics

| Rubric | Scope | Scale | What it checks |
|--------|-------|-------|----------------|
| `agent_behavior` | turn | 5-point | Instruction following, tool use, hallucinations (default) |
| `safety` | turn | binary | Harm, PII, unauthorized actions |
| `conversation_policy` | conversation | 5-point | Goal progression, consistency, drift |

Custom rubrics are defined as YAML files. See the [judge engine README](../openbias/policy/engines/judge/README.md) for the rubric schema.

### Sync vs async modes

- **Async** (default): The agent's response reaches the user immediately. The judge runs in the background. Violations are applied as interventions on the next turn. Zero latency impact.
- **Sync**: The response is held until the judge finishes. Violations are applied immediately. Adds one LLM round-trip of latency.

### Ensemble

When `ensemble_enabled: true`, multiple judge models evaluate the same response. Results are aggregated by `mean_score` or `conservative` (takes the lowest score). Useful for reducing single-model bias at the cost of additional LLM calls.

### Minimal config

```yaml
engine: judge
policy:
  - "No financial advice"
  - "Be professional"
judge:
  model: anthropic/claude-sonnet-4-5
```

Full configuration reference: [docs/configuration.md](configuration.md#judge-engine)

### Policy Generation

You can generate judge rubrics from natural language using the CLI:
```bash
openbias compile "be professional, never leak PII" --engine judge -o policy.yaml
```

Deep dive: [openbias/policy/engines/judge/README.md](../openbias/policy/engines/judge/README.md)

---

## FSM Engine

**Config key**: `engine: fsm`

Models allowed agent behavior as a finite state machine defined in YAML. Classifies each LLM response to a workflow state using a three-tier cascade (tool call matching, regex, semantic embeddings), evaluates temporal constraints based on LTL-lite, and triggers interventions on violations.

### When to use it

- Multi-step workflows with defined ordering (onboarding, support tickets, refund flows)
- Processes where certain steps must precede others (verify identity before refund)
- Cases where you need deterministic, auditable enforcement with zero LLM overhead
- Tool-heavy agents where state is clearly indicated by which tools are called

### How it works

1. You define states, transitions, constraints, and interventions in a workflow YAML
2. On each agent response, the classifier determines which state the response belongs to
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

Intervention templates in the workflow YAML support strategy prefixes:

| Prefix | Strategy | Effect |
|--------|----------|--------|
| *(none)* | `SYSTEM_PROMPT_APPEND` | Appends guidance to the system message |
| `inject:` | `USER_MESSAGE_INJECT` | Inserts as a user message |

### Minimal config

```yaml
engine: fsm
policy: ./workflow.yaml
```

Full configuration reference: [docs/configuration.md](configuration.md#fsm-engine)

### Policy Generation

You can generate FSM workflows from sequence descriptions:
```bash
openbias compile "verify identity before refunds" --engine fsm -o workflow.yaml
```

Deep dive: [openbias/policy/engines/fsm/README.md](../openbias/policy/engines/fsm/README.md)

---

## LLM Engine

**Config key**: `engine: llm`

Uses a lightweight sidecar LLM for state classification, drift detection, and soft constraint evaluation. Reads the same workflow YAML as the FSM engine, so you can swap between them without rewriting policies. Trades determinism for the ability to handle ambiguous, conversational workflows where tool calls and regex are insufficient.

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
engine: llm
llm:
  model: anthropic/claude-sonnet-4-5
```

Full configuration reference: [docs/configuration.md](configuration.md#llm-engine)

Deep dive: [openbias/policy/engines/llm/README.md](../openbias/policy/engines/llm/README.md)

---

## NeMo Guardrails Engine

**Config key**: `engine: nemo`

Wraps NVIDIA's [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) as a policy engine. Runs requests through input rails (pre-call) and responses through output rails (post-call) for jailbreak detection, PII filtering, toxicity checks, and programmable dialog flows via Colang.

### When to use it

- You need jailbreak detection or content moderation
- You want PII masking or toxicity filtering
- You have existing NeMo Guardrails configurations to reuse
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
engine: nemo
policy: ./nemo_config/
```

Full configuration reference: [docs/configuration.md](configuration.md#nemo-guardrails-engine)

### Policy Generation

You can generate NeMo configurations (Colang flows + config.yml):
```bash
openbias compile "block hacking requests" --engine nemo -o ./nemo_config
```

Deep dive: [openbias/policy/engines/nemo/README.md](../openbias/policy/engines/nemo/README.md)


