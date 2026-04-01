# Examples

Each example is a self-contained directory with an `openbias.yaml` config and a Python client script. Start the proxy in one terminal, run the client in another.

**Provider-agnostic**: every example auto-detects the model from whichever API key you have set. Set exactly one of `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`.

```
Your App  ──►  Open Bias (:4000)  ──►  LLM Provider
                     │
              pre_call_hook    → apply deferred interventions (μs)
              LLM call         → forwarded unmodified via LiteLLM
              post_call_hook   → policy engine evaluates async
                                 violations queued for next turn
```

Every example below triggers this pipeline. The interesting part is what the policy engine does in step 3.

---

## Quickstart — 60 seconds to running

[`examples/quickstart/`](quickstart/)

The smallest possible demo. ~30 lines of client code, 3 policy rules. Shows the judge engine evaluating responses in the background with zero critical-path latency. Start here.

```bash
cd examples/quickstart
export OPENAI_API_KEY=...    # or GEMINI_API_KEY, ANTHROPIC_API_KEY
openbias serve              # terminal 1
python quickstart.py         # terminal 2
```

---

## Prompt Injection Defense — async judge + deferred intervention

[`examples/judge/`](judge/)

A coding assistant that gets hit with a prompt injection attack. The judge engine evaluates the response asynchronously (zero latency on your call), catches the violation, and injects a system prompt amendment on the next turn. Watch the agent reassert its boundaries.

**What's happening under the hood**: The judge LLM scores responses on built-in rubrics (safety, instruction_following, tone) and maps the verdict to an action (pass/intervene/block/shadow).

```bash
cd examples/judge
export OPENAI_API_KEY=...    # or GEMINI_API_KEY, ANTHROPIC_API_KEY
openbias serve
python prompt_injection.py
```

---

## Workflow Enforcement — deterministic FSM with LTL constraints

[`examples/fsm_workflow/`](fsm_workflow/)

A customer support agent with a precedence constraint: identity verification must happen before any account action. The agent tries to process a refund without verifying — the FSM catches the violation and injects corrective guidance.

**What's happening under the hood**: Classification uses a three-tier cascade — tool call name matching (confidence 1.0, ~0ms), regex patterns (0.85, ~1ms), semantic embeddings (proportional to cosine similarity, ~50ms). First confident match wins. Constraints are evaluated as LTL-lite temporal logic over the state history.

```bash
cd examples/fsm_workflow
export OPENAI_API_KEY=...    # or GEMINI_API_KEY, ANTHROPIC_API_KEY
openbias serve
python workflow_enforcement.py
```

---

## Content Safety Rails — NeMo Guardrails engine

[`examples/nemo_guardrails/`](nemo_guardrails/)

Wraps NVIDIA NeMo Guardrails as a policy engine. Input rails run pre-call (jailbreak detection, PII filtering), output rails run post-call (toxicity, topical control). Fail-open by default — if NeMo throws, the request passes through with a warning.

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
