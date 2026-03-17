# Developing

## Setup

```bash
git clone https://github.com/open-sentinel/open-sentinel.git
cd open-sentinel
pip install -e ".[dev]"
```

For NeMo Guardrails support:

```bash
pip install -e ".[dev,nemo]"
```

Verify the install:

```bash
pytest
osentinel --help
```

## Running Locally

```bash
# Initialize config files (creates osentinel.yaml and policy.yaml)
osentinel init

# Start the proxy
osentinel serve

# Point any OpenAI-compatible client at the proxy
# base_url="http://localhost:4000/v1"
```

Override engine selection at startup:

```bash
osentinel serve --config custom_config.yaml
```

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
from opensentinel.policy.protocols import PolicyEngine
from opensentinel.config.settings import SentinelSettings
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

### Test Layout

```
tests/
├── conftest.py                    # Shared fixtures
├── config/                        # Config/settings tests
├── tracing/
│   └── test_otel_tracer.py
├── core/
│   ├── interceptor/
│   │   ├── test_interceptor.py    # Interceptor orchestration
│   │   └── test_adapters.py       # PolicyEngineChecker adapter
│   └── test_intervention.py       # Strategy tests
├── proxy/
│   ├── test_hooks.py              # SentinelCallback tests
│   └── test_middleware.py         # Session extraction
└── policy/
    ├── compiler/
    └── engines/
        ├── fsm/
        │   ├── test_classifier.py
        │   ├── test_constraints.py
        │   ├── test_state_machine.py
        │   └── test_workflow_parser.py
        ├── llm/
        ├── nemo/
        └── composite/
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
from opensentinel.policy.engines.fsm import WorkflowStateMachine, TransitionResult

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

Create a package under `opensentinel/policy/engines/`:

```python
# opensentinel/policy/engines/my_engine/engine.py
from opensentinel.policy.protocols import PolicyEngine, Decision, EngineResult
from opensentinel.policy.registry import register_engine

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
        return EngineResult(decision=Decision.ALLOW)

    async def evaluate_response(self, session_id, response_data, request_data, context=None):
        return EngineResult(decision=Decision.ALLOW)

    async def get_session_state(self, session_id):
        return None

    async def reset_session(self, session_id):
        pass
```

Import the engine in `__init__.py` to trigger registration:

```python
# opensentinel/policy/engines/my_engine/__init__.py
from .engine import MyPolicyEngine
```

The `Interceptor` automatically wraps registered engines as `PolicyEngineChecker` instances. No changes to the proxy layer needed.

### Adding a Checker

All checkers are `PolicyEngineChecker` instances wrapping a `PolicyEngine`. Create a new engine (see above) and register it via `PolicyEngineChecker`:

```python
from opensentinel.core.interceptor.adapters import PolicyEngineChecker
from opensentinel.core.interceptor.types import CheckPhase, CheckerMode

checker = PolicyEngineChecker(
    engine=my_engine,
    phase=CheckPhase.POST_CALL,
    mode=CheckerMode.ASYNC,  # Runs in background, results applied next request
)
```

Register it in `SentinelCallback._get_interceptor()` in `opensentinel/proxy/hooks.py`.

### Adding a Constraint Type (FSM Engine)

1. Add to `ConstraintType` enum in `opensentinel/policy/engines/fsm/workflow/schema.py`.
2. Add validation in `Constraint.validate_constraint_params()`.
3. Implement evaluation in `ConstraintEvaluator._evaluate_constraint()` in `opensentinel/policy/engines/fsm/workflow/constraints.py`.
4. Add message formatting in `_format_violation_message()`.

### Adding an Intervention Strategy

Two strategies are supported: `SYSTEM_PROMPT_APPEND` and `USER_MESSAGE_INJECT`. The `Interceptor` selects the strategy based on `default_strategy` in `InterventionConfig`. To add a new strategy:

1. Add to `StrategyType` enum in `opensentinel/core/intervention/strategies.py`.
2. Create a class extending `InterventionStrategy` with a `merge()` method.
3. Add handling for the new type in `Interceptor._apply_intervention()` in `opensentinel/core/interceptor/interceptor.py`.

### Adding a Classification Method (FSM Engine)

The FSM classifier uses a cascade: tool calls -> regex patterns -> embeddings. To add a method, extend `StateClassifier.classify()` in `opensentinel/policy/engines/fsm/classifier.py` and insert your method at the appropriate priority in the cascade.

### Adding a Policy Compiler

```python
# opensentinel/policy/engines/my_engine/compiler.py
from opensentinel.policy.compiler.base import LLMPolicyCompiler
from opensentinel.policy.compiler.protocol import CompilationResult
from opensentinel.policy.compiler.registry import register_compiler

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

Edit `opensentinel/cli.py`:

```python
@main.command()
@click.option("--option", "-o", help="Description")
@click.argument("arg")
def mycommand(option: str, arg: str):
    """Short description."""
    click.echo(f"Running with {option} and {arg}")
```

### Adding Configuration Options

Edit `opensentinel/config/settings.py`:

```python
class MyComponentConfig(BaseModel):
    option_a: str = "default"
    option_b: int = 42

class SentinelSettings(BaseSettings):
    # ... existing fields ...
    my_component: MyComponentConfig = Field(default_factory=MyComponentConfig)
```

Configuration is primarily handled via `osentinel.yaml`.

## Debugging

Enable debug logging:

```bash
DEBUG=true osentinel serve
```

(If using standard env vars) or set `debug: true` in `osentinel.yaml`.

Target specific loggers:

```python
import logging
logging.getLogger("opensentinel.policy.engines.fsm.classifier").setLevel(logging.DEBUG)
logging.getLogger("opensentinel.policy.engines.llm.engine").setLevel(logging.DEBUG)
logging.getLogger("opensentinel.core.interceptor").setLevel(logging.DEBUG)
```

Monitor fail-open activations:

```python
from opensentinel.proxy.hooks import get_fail_open_counts
counts = get_fail_open_counts()
# {"pre_call": 0, "post_call": 1, ...}
```

### OpenTelemetry Tracing

```bash
Set `tracing:` section in `osentinel.yaml`.

# Local Jaeger instance
docker run -d -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one:latest
```

Traces are grouped by session. View them at `http://localhost:16686`.

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
Ensure your `osentinel.yaml` has the correct `policy:` path or inline rules.

**"Failed to load embedding model"**
Install `sentence-transformers` and check disk space. The model downloads ~100MB on first use.

**"Unknown intervention: ..."**
The intervention name in a constraint must match a key in the `interventions` dict of your workflow YAML.

**"Workflow has no initial state"**
At least one state needs `is_initial: true`.

**"Constraint references unknown state"**
Check that `trigger` and `target` values in constraints match state names exactly.

**"Unknown policy engine type: '...'"**
Valid types: `judge`, `fsm`, `llm`, `nemo`, `composite`. Ensure the engine module is imported in `opensentinel/policy/engines/__init__.py`.

### NeMo-Specific

**"API key not valid (400)"**
Check that your model configuration in `config.yml` maps to a valid API key in your `.env` file.

**"TypeError: 'function' object is not subscriptable"**
Pydantic v1/v2 conflict between LangChain and OpenSentinel. Pin compatible versions.

### LLM Engine-Specific

**Low classification confidence** -- Add more descriptive state descriptions and exemplars. Try a more capable model (e.g., GPT-4o or Claude 3.5 Sonnet). Check `turn_window` size.

**Excessive interventions** -- Increase `cooldown_turns`. Lower `self_correction_margin` to detect self-correction more aggressively.

**High drift scores** -- Adjust `temporal_weight` to rebalance temporal vs. semantic drift. Verify workflow states reflect expected agent behavior.

### Interceptor

**Async checker results not applied** -- Async results are collected at the start of the next `run_pre_call`. Ensure session IDs are consistent across requests.

**Checker errors not propagating** -- By default, checker errors produce a `FAIL` decision. Under `safe_hook`, only `WorkflowViolationError` propagates; all other errors trigger pass-through.

### Session ID

If workflows aren't tracked correctly:

1. Add debug logging to `opensentinel/proxy/middleware.py` to see extracted session IDs.
2. Ensure consistent session identifiers across calls.
3. Use the `x-sentinel-session-id` header for explicit control.
