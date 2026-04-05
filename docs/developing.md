# Developing

## Setup

```bash
git clone https://github.com/open-bias/open-bias.git
cd open-bias
pip install -e ".[dev]"
```

For NeMo Guardrails support:

```bash
pip install -e ".[dev,nemo]"
```

Verify the install:

```bash
pytest
openbias --help
```

## Running Locally

```bash
# Initialize config (creates openbias.yaml)
openbias init

# Start the proxy
openbias serve

# Point any OpenAI-compatible client at the proxy
# base_url="http://localhost:4000/v1"
```

Override engine selection at startup:

```bash
openbias serve --config custom_config.yaml
```

For the trace-backed improvement loop, keep `openbias.yaml`, `rules.md`, repo-owned eval suites under `evals/suites/`, and replayable JSONL traces under `.openbias/traces/`.

## Code Conventions

### File Organization

- One class per file for major components.
- `__init__.py` exports the public API of each package.
- Type hints required on all function signatures.
- Docstrings on all public classes and methods.

### Naming

| Entity | Convention | Example |
|--------|-----------|---------|
| Files | `snake_case.py` | `state_machine.py` |
| Classes | `PascalCase` | `PolicyEngine` |
| Functions | `snake_case` | `process_response` |
| Constants | `UPPER_SNAKE_CASE` | `STRATEGY_REGISTRY` |
| Private | `_prefix` | `_sessions`, `_extract_content` |

### Import Order

```python
# 1. Standard library
import logging
from typing import Optional, Dict, Any

# 2. Third-party
from pydantic import BaseModel
import litellm

# 3. Local
from openbias.policy.protocols import PolicyEngine
from openbias.config.settings import Settings
```

## Testing

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific file
pytest tests/policy/engines/fsm/test_state_machine.py

# Specific test
pytest tests/policy/engines/fsm/test_state_machine.py::TestWorkflowStateMachine::test_create_session

# Verbose
pytest -v
```

### Trace Replay

```bash
# Replay one trace dataset against the configured engine
openbias replay --trace .openbias/traces/2026-04-05.jsonl

# Export replay results for later comparison/reporting
openbias replay --trace .openbias/traces/2026-04-05.jsonl --json-output replay.json
```

### Policy Improvement

```bash
# Generate variants from rules.md and replay them across captured traces
openbias improve \
  --trace .openbias/traces/2026-04-05.jsonl \
  --instruction "Tighten the policy around refund abuse while preserving benign support workflows."
```

### Nightly Improvement Workflow

The OSS repo ships a copyable GitHub Actions example at [`examples/github-actions/nightly-improvement.yml`](../examples/github-actions/nightly-improvement.yml).

That workflow intentionally lives under `examples/` instead of `.github/workflows/` because this repository does not ship a runnable project-local `rules.md`, `openbias.yaml`, and trace history for itself.

The example assumes your application repo has:

- a committed `openbias.yaml`
- a committed baseline `rules.md`
- repo-owned eval suites under `evals/suites/`
- replayable trace files under `.openbias/traces/`
- an improvement instruction that describes how policy variants should differ from the baseline

It runs `openbias eval`, `openbias replay`, and `openbias improve`, then uploads the resulting artifacts.

Adjust the install step to match your repo. The example defaults to `pip install openbias` so it is easy to copy into an application repository that depends on Open Bias.

End-to-end walkthrough: [continuous-improvement.md](continuous-improvement.md)

### Test Layout

```
tests/
├── conftest.py                    # Shared fixtures
├── test_cli.py                    # CLI commands
├── config/                        # Config/settings tests
├── core/
│   ├── interceptor/
│   │   └── test_interceptor.py    # Interceptor orchestration
│   ├── test_intervention.py       # Strategy tests
│   └── test_session.py
├── eval/                          # Offline scenario-based evaluation
├── proxy/
│   ├── test_hooks.py              # Callback tests
│   ├── test_middleware.py         # Session extraction
│   └── test_server_shutdown.py    # Graceful shutdown
├── tracing/
│   └── test_otel_tracer.py
└── policy/
    ├── compiler/                  # Per-engine compiler tests
    └── engines/
        ├── fsm/                   # Classifier, constraints, state machine
        ├── judge/                 # Evaluator, per-rule verdicts, rules, tool calls
        ├── llm/                   # Drift, constraints, classification
        └── nemo/
```

### Key Fixtures

Defined in `tests/conftest.py`:

| Fixture | Description |
|---------|-------------|
| `sample_workflow` | Full customer support workflow from `examples/` |
| `simple_workflow` | Minimal 3-state workflow for basic tests |
| `mock_llm_response` | Factory for creating mock LLM responses |
| `mock_tool_call` | Factory for creating mock tool calls |

### Writing a Test

```python
import pytest
from openbias.policy.engines.fsm import WorkflowStateMachine, TransitionResult

class TestMyFeature:
    @pytest.fixture
    def machine(self, simple_workflow):
        return WorkflowStateMachine(simple_workflow)

    @pytest.mark.asyncio
    async def test_transition_succeeds(self, machine):
        session = await machine.get_or_create_session("test")
        result, error = await machine.transition("test", "middle")

        assert result == TransitionResult.SUCCESS
        assert error is None
```

## Linting and Type Checking

```bash
make lint        # ruff check
make typecheck   # mypy
make format      # ruff fix + format
```

## Extension Points

### Adding a Policy Engine

Create a package under `openbias/policy/engines/`:

```python
# openbias/policy/engines/my_engine/engine.py
from openbias.policy.protocols import PolicyEngine, EvaluationResult, EvaluationStatus
from openbias.policy.registry import register_engine

@register_engine("my_engine")
class MyPolicyEngine(PolicyEngine):
    @property
    def name(self) -> str:
        return "my_engine"

    @property
    def engine_type(self) -> str:
        return "my_engine"

    async def initialize(self, config):
        ...

    async def evaluate_request(self, session_id, request_data, context=None):
        return EvaluationResult(status=EvaluationStatus.ALLOW)

    async def evaluate_response(self, session_id, response_data, request_data, context=None):
        return EvaluationResult(status=EvaluationStatus.ALLOW)

    async def get_session_state(self, session_id):
        return None

    async def reset_session(self, session_id):
        pass
```

Import the engine in `__init__.py` to trigger registration:

```python
# openbias/policy/engines/my_engine/__init__.py
from .engine import MyPolicyEngine
```

The `Interceptor` accepts registered engines directly via `Interceptor(pre_call_evaluators=[], post_call_evaluators=[engine])`. No adapter layer needed.

### Adding a Constraint Type (FSM Engine)

1. Add to `ConstraintType` enum in `openbias/policy/engines/fsm/workflow/schema.py`.
2. Add validation in `Constraint.validate_constraint_params()`.
3. Implement evaluation in `ConstraintEvaluator._evaluate_constraint()` in `openbias/policy/engines/fsm/workflow/constraints.py`.
4. Add message formatting in `_format_violation_message()`.

### Adding an Intervention Strategy

Two request-time strategies exist as standalone classes in `openbias/core/intervention/strategies.py` (no base class):

- **`SystemPromptAppendStrategy`** — appends guidance to the system message via `merge(messages, value)`.
- **`UserMessageInjectStrategy`** — injects a user message with guidance via `merge(messages, value)`.

The `StrategyType` enum maps to these classes. To add a new strategy:

1. Add a variant to `StrategyType` enum in `openbias/core/intervention/strategies.py`.
2. Create a standalone class with `merge(messages, value)` and `cleanup_rules()` so it behaves like the existing request-time strategies.
3. Add handling for the new type in `Interceptor._apply_intervention()` in `openbias/core/interceptor/interceptor.py`.

### Adding a Classification Method (FSM Engine)

The FSM classifier uses a cascade: tool calls -> regex patterns -> embeddings. To add a method, extend `StateClassifier.classify()` in `openbias/policy/engines/fsm/classifier.py` and insert your method at the appropriate priority in the cascade.

### Adding a Policy Compiler

```python
# openbias/policy/engines/my_engine/compiler.py
from openbias.policy.compiler.base import LLMPolicyCompiler
from openbias.policy.compiler.protocol import CompilationResult
from openbias.policy.compiler.registry import register_compiler

@register_compiler("my_engine")
class MyEngineCompiler(LLMPolicyCompiler):
    @property
    def engine_type(self) -> str:
        return "my_engine"

    def _build_compilation_prompt(self, natural_language, context=None):
        return f"Convert to my engine format:\n{natural_language}"

    def _parse_compilation_response(self, response, natural_language):
        config = MyEngineConfig(**response)
        return CompilationResult(success=True, config=config)

    def export(self, result, output_path):
        with open(output_path, "w") as f:
            yaml.dump(result.config.to_dict(), f)
```

Import in the engine's `__init__.py` to trigger registration.

### Adding a CLI Command

Edit `openbias/cli.py`:

```python
@main.command()
@click.option("--option", "-o", help="Description")
@click.argument("arg")
def mycommand(option: str, arg: str):
    """Short description."""
    click.echo(f"Running with {option} and {arg}")
```

### Adding Configuration Options

Engine-specific config keys are set directly in the evaluator entry in `openbias.yaml`:

```yaml
# openbias.yaml
evaluators:
  - name: my-evaluator
    type: my_engine
    option_a: "value"
    option_b: 42
```

These keys are collected into the evaluator's `config` dict and passed to the engine's `initialize()` method. No changes to `settings.py` are needed for engine-specific options.

## Debugging

Enable debug logging:

```bash
openbias serve --debug
```

Or set `debug: true` in `openbias.yaml`.

Target specific loggers:

```python
import logging
logging.getLogger("openbias.policy.engines.fsm.classifier").setLevel(logging.DEBUG)
logging.getLogger("openbias.policy.engines.llm.engine").setLevel(logging.DEBUG)
logging.getLogger("openbias.core.interceptor").setLevel(logging.DEBUG)
```

Monitor fail-open activations:

```python
from openbias.proxy.hooks import get_fail_open_counts
counts = get_fail_open_counts()
# {"pre_call": 0, "post_call": 1, ...}
```

### OpenTelemetry Tracing

```bash
Set `tracing:` section in `openbias.yaml`.

# Local Jaeger instance
docker run -d -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one:latest
```

Traces are grouped by session. View them at `http://localhost:16686`.

Tracing contract for async evaluators:

- `openbias.async.phase` is canonical for async lifecycle (`dispatched`, `executing`, `applied`, and `dropped` for eviction logs/telemetry).
- Apply-time spans use `openbias.evaluator.phase=async_applied` (not `pre_call`) to avoid query ambiguity.
- Async execution causality is links-canonical: execution spans carry origin attributes (`openbias.origin.trace_id`, `openbias.origin.span_id`) and OTEL links to dispatch context.
- Judge verdict metadata is written onto the current evaluator span; no fallback standalone judge span is emitted when no active span is available.

## Performance Reference

### Classification Latency (FSM Engine)

| Method | Latency | When used |
|--------|---------|-----------|
| Tool call match | ~0ms | Response contains `tool_calls` |
| Regex patterns | ~1ms | Patterns defined, no tool match |
| Embedding similarity | ~50ms | Exemplars defined, no pattern match |

### Classification Latency (LLM Engine)

| Method | Latency | When used |
|--------|---------|-----------|
| LLM classification | 200-500ms | Every response |

### Memory

| Component | Footprint |
|-----------|-----------|
| Embedding model (sentence-transformers) | ~100MB |
| Compiled regex patterns | Cached per workflow |
| State embeddings | Cached after first computation |
| Session state (FSM) | ~1KB per active session |
| Session state (LLM) | ~5KB per active session (turn window) |

### Concurrency

The state machine uses `asyncio.Lock` per session. Sessions are stored in memory (not persistent across restarts). For high-concurrency deployments, consider external state storage.

## Troubleshooting

**"No workflow configured - running in pass-through mode"**
Ensure the project has a `rules.md` file and that your evaluator is declared in `openbias.yaml`.

**"Failed to load embedding model"**
Install `sentence-transformers` and check disk space. The model downloads ~100MB on first use.

**"Unknown intervention: ..."**
The intervention name in a constraint must match a key in the internal workflow definition produced by compilation.

**"Workflow has no initial state"**
Update `rules.md` so the compiled workflow has a clear starting step or state.

**"Constraint references unknown state"**
Check the corresponding rules in `rules.md`; every referenced trigger/target must compile to a real workflow state.

**"Unknown policy engine type: '...'"**
The `type` field on an evaluator entry in `openbias.yaml` must match a registered engine key (e.g. `judge`, `fsm`, `llm`, `nemo`). Ensure the engine module is imported in `openbias/policy/engines/__init__.py` so its `@register_engine` decorator runs.

### NeMo-Specific

**"API key not valid (400)"**
Check the model configured in `openbias.yaml` and ensure the matching API key is present in your environment. NeMo runtime artifacts are compiled internally from `rules.md`.

**"TypeError: 'function' object is not subscriptable"**
Pydantic v1/v2 conflict between LangChain and OpenBias. Pin compatible versions.

### LLM Engine-Specific

**Low classification confidence** -- Add more descriptive state descriptions and exemplars. Try a more capable model (e.g., GPT-4o or Claude 3.5 Sonnet). Check `turn_window` size.

**Excessive interventions** -- Increase `cooldown_turns`. Lower `self_correction_margin` to detect self-correction more aggressively.

**High drift scores** -- Adjust `temporal_weight` to rebalance temporal vs. semantic drift. Verify workflow states reflect expected agent behavior.

### Interceptor

**Async evaluator results not applied** -- Async results are collected at the start of the next `run_pre_call`. Ensure session IDs are consistent across requests.

**Evaluator errors not propagating** -- By default, evaluator errors produce a `FAIL` decision. Under `safe_hook`, only `WorkflowViolationError` propagates; all other errors trigger pass-through.

### Session ID

If workflows aren't tracked correctly:

1. Add debug logging to `openbias/proxy/middleware.py` to see extracted session IDs.
2. Ensure consistent session identifiers across calls.
3. Use the `x-openbias-session-id` header for explicit control.
