# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open Sentinel is a transparent LLM proxy that monitors and enforces policies on AI agent behavior. It sits between applications and LLM providers (OpenAI, Google, Anthropic, etc.), intercepting calls and evaluating responses against configurable policy engines. When violations are detected, it intervenes by modifying prompts or blocking responses.

## Common Commands

```bash
make install-dev          # Install with dev dependencies
make test                 # Run all tests (pytest)
make test-cov             # Tests with coverage report
make lint                 # Lint with ruff
make typecheck            # Type-check with mypy (strict)
make format               # Auto-format with ruff

# Run specific tests
pytest tests/policy/engines/fsm/test_state_machine.py
pytest tests/policy/engines/fsm/test_state_machine.py::TestWorkflowStateMachine::test_create_session

# Start the proxy server
osentinel serve
```

## Architecture

### Request Flow

```
Client → Proxy (LiteLLM Router) → Interceptor → Policy Engines → LLM Provider
```

The proxy wraps LiteLLM and uses callback hooks (`async_pre_call_hook`, `async_post_call_success_hook`) to intercept requests. The Interceptor orchestrates Checkers across PRE_CALL and POST_CALL phases, in both SYNC and ASYNC modes. Async checker results are collected at the start of the next request (deferred intervention pattern).

### Key Design Decisions

- **Fail-open**: All hook exceptions are caught by `safe_hook()` and logged. Only `WorkflowViolationError` propagates (intentional blocks). Everything else passes through.
- **Deferred interventions**: Async checkers (like the Judge engine) evaluate in the background with zero latency impact. Interventions are applied on the next request.
- **Session tracking**: Session IDs extracted with priority: `x-sentinel-session-id` header > `metadata.session_id` > `metadata.run_id` > `user` field > `thread_id` > hash of first message > random UUID.

### Core Components

| Directory | Purpose |
|-----------|---------|
| `opensentinel/proxy/` | LiteLLM wrapper, callback hooks, session extraction middleware |
| `opensentinel/core/interceptor/` | Checker orchestration pipeline (phases, modes, decisions) |
| `opensentinel/core/intervention/` | Intervention strategies (system prompt append, user message inject, context reminder, hard block) |
| `opensentinel/policy/protocols.py` | Core protocol definitions: `Decision` enum, `EngineResult`, `PolicyEngine` ABC, `StatefulPolicyEngine` ABC |
| `opensentinel/policy/` | Engine registry (`@register_engine` decorator), engine implementations |
| `opensentinel/policy/compiler/` | Natural-language-to-config compilers: `PolicyCompiler` ABC, `LLMPolicyCompiler` base, per-engine compilers |
| `opensentinel/policy/engines/judge/` | LLM-as-a-judge with rubrics and session tracking |
| `opensentinel/policy/engines/fsm/` | Deterministic state machine with classification cascade (tool call → regex → embeddings) |
| `opensentinel/policy/engines/llm/` | LLM-based state classification and drift detection |
| `opensentinel/policy/engines/nemo/` | NVIDIA NeMo Guardrails wrapper |
| `opensentinel/eval/` | Offline eval framework: `EvalRunner` replays conversation JSON through engines, `EvalMetrics`, `EvalReporter` |
| `opensentinel/config/` | Pydantic BaseSettings, reads from `osentinel.yaml` / env vars (`OSNTL_*`) |
| `opensentinel/tracing/` | OpenTelemetry tracing with session-aware span grouping |

### Extension Points

- **New policy engine**: Create package under `opensentinel/policy/engines/`, use `@register_engine("type")` decorator, implement `PolicyEngine` ABC from `opensentinel/policy/protocols.py`, import in `opensentinel/policy/engines/__init__.py`.
- **New policy compiler**: Use `@register_compiler("type")` decorator, subclass `LLMPolicyCompiler` (or `PolicyCompiler` ABC directly), implement `_build_compilation_prompt`, `_parse_compilation_response`, and `export`.
- **New checker**: Implement `Checker` protocol, register in `SentinelCallback._get_interceptor()` in `hooks.py`. Use `PolicyEngineChecker` adapter (`core/interceptor/adapters.py`) to wrap a `PolicyEngine` as a checker.
- **New intervention strategy**: Add to `StrategyType` enum and `STRATEGY_REGISTRY` in `opensentinel/core/intervention/strategies.py`.
- **New FSM constraint type**: Add to `ConstraintType` enum in `schema.py`, implement in `constraints.py`.

## Code Conventions

- One class per file for major components
- Type hints on all function signatures (mypy strict: `disallow_untyped_defs = true`)
- Ruff linting: line length 100, rules E/F/I/N/W/UP (E501 ignored)
- Python 3.10+ target
- Async tests use `pytest-asyncio` with `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed)
- No backward-compatibility shims: when refactoring, remove old code paths entirely. Do not keep aliases, deprecated methods, or fallback logic for old formats.

## Commit Style

Format: `type<scope>: description` — scope is optional.

**Types:**
- `feat` — new feature
- `refactor` — code restructure without behavior change
- `patch` — small fix or correction
- `test` — adding or updating tests
- `docs` — documentation only
- `update` — general updates that don't fit above

**Scope** (optional): component or area in angle brackets, e.g. `<eval>`, `<fsm>`, `<judge>`, `<tracer>`

**Rules:**
- Description is lowercase, brief, no period
- No body or footer
- When tests accompany a refactor: `test<refactor>: ...`
- When staging and committing, skip verbose explanations — just run the commands silently

Examples: `feat<eval>: runner`, `patch<tracer>: shutdown`, `refactor: interceptor`, `test: add coverage for eval`

## Configuration

Settings loaded from `osentinel.yaml` with env var overrides using `OSNTL_` prefix. Nested vars use double underscore: `OSNTL_JUDGE__MODE=safe`. Debug mode: `OSNTL_DEBUG=true`.
