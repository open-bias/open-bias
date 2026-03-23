# Architecture

Open Bias is a transparent proxy between your application and LLM providers. It intercepts every call, evaluates it against policy engines, and intervenes when violations are detected.

```
┌─────────────┐    ┌───────────────────────────────────────────┐    ┌─────────────┐
│  Your App   │───▶│              OPEN BIAS                    │───▶│ LLM Provider│
│             │    │  ┌────────-┐  ┌─────────────┐             │    │             │
│             │◀───│  │ Hooks   │─▶│ Interceptor │             │◀───│             │
└─────────────┘    │  │safe_hook│  │ ┌─────────┐ │             │    └─────────────┘
                   │  └────────-┘  │ │Checkers │ │             │
                   │      │        │ └─────────┘ │             │
                   │      ▼        └─────────────┘             │
                   │  ┌────────────────────────────────────┐   │
                   │  │         Policy Engines             │   │
                   │  │  ┌───────┐ ┌─────┐ ┌─────┐ ┌────┐  │   │
                   │  │  │ Judge │ │ FSM │ │ LLM │ │NeMo│  │   │
                   │  │  └───────┘ └─────┘ └─────┘ └────┘  │   │
                   │  └────────────────────────────────────┘   │
                   │      │                                    │
                   │      ▼                                    │
                   │  ┌────────────────────────────────────┐   │
                   │  │      OpenTelemetry Tracing         │   │
                   │  └────────────────────────────────────┘   │
                   └───────────────────────────────────────────┘
```

## Components

### Proxy Layer (`openbias/proxy/`)

Wraps LiteLLM to intercept all LLM traffic.

- **`server.py`** -- `Proxy`. Main entry point. Wraps LiteLLM Router with Open Bias callbacks.
- **`hooks.py`** -- `Callback`. Implements LiteLLM's `CustomLogger` interface. Four hooks:

| Hook | Timing | Purpose |
|------|--------|---------|
| `async_pre_call_hook` | Before LLM call | Apply pending interventions, run PRE_CALL checkers, start trace |
| `async_moderation_hook` | Parallel with LLM | Reserved (unused) |
| `async_post_call_success_hook` | After LLM response | Run POST_CALL checkers, start async checkers, complete trace |
| `async_post_call_failure_hook` | After LLM error | Log failure |

- **`middleware.py`** -- Session ID extraction. Priority: `x-openbias-session-id` header > `metadata.session_id` > `metadata.run_id` (LangChain) > `user` field > `thread_id` > hash of first message > random UUID.

### Interceptor (`openbias/core/interceptor/`)

Orchestration layer between hooks and policy engines. Runs checkers in two phases (PRE_CALL, POST_CALL) with two execution modes (SYNC, ASYNC).

`run_pre_call`: collects async results from the previous request, runs sync PRE_CALL checkers, starts async PRE_CALL checkers in background.

`run_post_call`: runs sync POST_CALL checkers, starts async POST_CALL checkers (results applied on next request).

Policy engines are passed directly to the `Interceptor` — no adapter layer.

### Policy Engines (`openbias/policy/`)

All engines implement the `PolicyEngine` protocol (`protocols.py`): `initialize`, `evaluate_request`, `evaluate_response`, `get_session_state`, `reset_session`, `shutdown`. Engines register via `@register_engine("type")` and are created through `PolicyEngineRegistry`.

Engine-specific docs: [engines.md](engines.md). Engine-specific READMEs live in each engine's source directory under `openbias/policy/engines/`.

### Intervention Strategies (`openbias/core/intervention/`)

| Strategy | Mechanism |
|----------|-----------|
| `SYSTEM_PROMPT_APPEND` | Appends guidance to system message |
| `USER_MESSAGE_INJECT` | Inserts a `[System Note]` as user message |
| `RESPONSE_MODIFICATION` | Modifies current response content or tool calls |

### Tracing (`openbias/tracing/`)

`Tracer` provides session-aware OpenTelemetry tracing. Spans are grouped by session. Uses GenAI semantic conventions (`gen_ai.request.model`, `gen_ai.usage.prompt_tokens`). Supports OTLP and Langfuse backends.

## Data Flows

### No violation

```
Client request
  → pre_call_hook: extract session, run PRE_CALL checkers
  → LLM call
  → post_call_hook: run POST_CALL checkers (all pass)
  → response returned unmodified
```

### Violation with deferred intervention

```
Call N:
  → post_call_hook: POST_CALL checker detects violation, schedules intervention
  → response returned unmodified (violation is deferred)

Call N+1:
  → pre_call_hook: collects async results, merges intervention into request
  → LLM receives corrected prompt, responds accordingly
```

### Fail-open

```
Hook throws or times out
  → safe_hook() catches it
  → WorkflowViolationError? re-raise (intentional block)
  → anything else? log warning, increment counter, pass through unmodified
```

## Design Decisions

**Fail-open over fail-closed.** A monitoring layer that takes down production is worse than one that misses a violation. All hooks have timeout and exception handling. Only explicit `WorkflowViolationError` blocks requests.

**Deferred intervention.** Violations detected in POST_CALL are applied on the next request, not retroactively. This preserves the current response and avoids race conditions with streaming.

**Engine-agnostic interceptor.** The interceptor knows about checkers and phases, not about FSMs or rubrics. Engines are wrapped as checkers via an adapter. Adding a new engine requires zero changes to the proxy layer.

**Async by default.** The judge engine runs in ASYNC mode -- evaluation happens in the background after the response is sent. This adds zero latency to the critical path. Sync mode is available when blocking evaluation is required.
