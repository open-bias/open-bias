# Examples

Each example is a self-contained directory with an `openbias.yaml` config, a project-local `RULES.md`, and a Python client script. The examples pin explicit configs so each scenario is reproducible, but the product itself does not require YAML for a first run. If you are just trying Open Bias, you can start in any directory with `RULES.md` plus `openbias serve`.

**Provider-agnostic**: every example auto-detects the model from whichever API key you have set. Set exactly one of `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `ANTHROPIC_API_KEY`.

```
Your App  ──►  Open Bias (:4000)  ──►  LLM Provider
                     │
              pre_call_hook    → apply deferred interventions (μs)
              LLM call         → forwarded unmodified via LiteLLM
              post_call_hook   → evaluator engine evaluates async
                                 violations queued for next turn
```

Every example below triggers this pipeline. The interesting part is what the evaluator engine does in step 3.

---

## Quickstart — Customer Support Agent

[`examples/quickstart/`](quickstart/)

A customer support agent for Acme Corp with 3 rules. Two turns: a normal password reset question passes cleanly, then a refund request on a 3-month-old order is **blocked immediately**. Runs in sync mode so the violation surfaces right in your terminal — no waiting for a second turn.

```bash
cd examples/quickstart
export OPENAI_API_KEY=...    # or GEMINI_API_KEY, ANTHROPIC_API_KEY
openbias serve              # terminal 1
python quickstart.py         # terminal 2
```

---

## Sales Agent — async judge + deferred intervention

[`examples/judge/`](judge/)

An AI sales rep that starts to offer a 40% discount — violating the pricing policy. The judge engine catches this asynchronously (zero latency on your call) and injects a system prompt amendment on the next turn, steering the agent back to approved pricing.

**What's happening under the hood**: Open Bias compiles the example's `RULES.md`, the judge LLM evaluates one rule at a time with binary pass/fail, and any violation is mapped to `intervene`, `block`, or `shadow`.

```bash
cd examples/judge
export OPENAI_API_KEY=...    # or GEMINI_API_KEY, ANTHROPIC_API_KEY
openbias serve
python sales_agent.py
```

### Enforcement modes

Use these copy-paste configs in `examples/judge/` for explicit behavior:

- `openbias.shadow.yaml`: `fail_action: shadow` (monitor-only, allow unchanged)
- `openbias.block.sync.yaml`: `mode: sync` + `fail_action: block` (immediate deny path)
- `openbias.yaml`: default `fail_action: intervene` (steer/repair flow)

Important: when `mode: async`, `fail_action: block` is normalized to `intervene`, because async evaluation cannot block a response that was already sent.

---

## Financial Services Chatbot — NeMo Guardrails engine

[`examples/nemo_guardrails/`](nemo_guardrails/)

A bank customer service chatbot that must never provide investment advice. When a customer asks about buying Tesla stock, NeMo's **input rail blocks the request before it reaches the LLM** — the model never sees it. On-topic banking questions pass through normally.

```bash
pip install 'openbias[nemo]'   # extra dependency
cd examples/nemo_guardrails
export OPENAI_API_KEY=...    # or GEMINI_API_KEY, ANTHROPIC_API_KEY
openbias serve
python content_safety.py
```

---

## Common patterns

**Session tracking**: Pass `X-OpenBias-Session-ID` header (or set `x-openbias-session-id` in `default_headers`) to group requests into a conversation. Without it, Open Bias falls back to: `metadata.session_id` → `metadata.run_id` → `user` field → `thread_id` → hash of first message → random UUID.

**Model strings**: Use [LiteLLM format](https://docs.litellm.ai/docs/providers) — `gpt-4o-mini`, `gemini/gemini-2.5-flash`, `anthropic/claude-sonnet-4-5`, etc. The proxy routes to the right provider based on the prefix.

**Fail-open**: All hooks are wrapped in `safe_hook()` with a 30s timeout. If a hook throws or times out, the request passes through unmodified. Only `WorkflowViolationError` (intentional hard blocks) propagates. The proxy never becomes the bottleneck.
