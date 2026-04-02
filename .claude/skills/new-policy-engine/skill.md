---
name: new-policy-engine
description: Guide for creating a new policy engine under openbias/policy/engines/
---

# Creating a New Policy Engine

## Decision: PolicyEngine vs StatefulPolicyEngine

- **`PolicyEngine`** — Use when each request/response is evaluated independently against policy. No state transitions between turns. Examples: Judge (compiled-rule evaluation), NeMo (guardrails).
- **`StatefulPolicyEngine`** (from `openbias.policy.engines.stateful`) — Use when you need to track state transitions across turns (e.g., FSM workflows). Adds `classify_response`, `get_current_state`, `get_state_history`, `get_valid_next_states`.

If you're unsure, start with `PolicyEngine`. You can always extend later.

## Checklist

1. **Create the engine package**

   ```
   openbias/policy/engines/<name>/
   ├── __init__.py    # Export engine class
   ├── engine.py      # Engine implementation
   └── compiler.py    # (Optional) NL-to-config compiler
   ```

2. **Implement the engine class** in `engine.py`

   ```python
   from typing import Any, Dict, Optional

   from openbias.policy.protocols import (
       EvaluationResult,
       EvaluationStatus,
       ViolationRecord,
       PolicyEngine,
       require_initialized,
   )
   from openbias.policy.registry import register_engine

   @register_engine("<name>")
   class MyPolicyEngine(PolicyEngine):
       def __init__(self) -> None:
           self._initialized = False
           self._config: Dict[str, Any] = {}
           self._session_data: Dict[str, Dict[str, Any]] = {}

       @property
       def name(self) -> str:
           return "<name>:<variant>"

       @property
       def engine_type(self) -> str:
           return "<name>"

       async def initialize(self, config: Dict[str, Any]) -> None:
           self._config = config
           # Setup resources, load models, etc.
           self._initialized = True  # CRITICAL: must set this

       @require_initialized
       async def evaluate_request(
           self,
           session_id: str,
           request_data: Dict[str, Any],
           context: Optional[Dict[str, Any]] = None,
       ) -> EvaluationResult:
           # Return EvaluationResult(status=EvaluationStatus.ALLOW)
           ...

       @require_initialized
       async def evaluate_response(
           self,
           session_id: str,
           response_data: Any,
           request_data: Dict[str, Any],
           context: Optional[Dict[str, Any]] = None,
       ) -> EvaluationResult:
           # This is where most evaluation logic lives
           ...

       async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
           return self._session_data.get(session_id)

       async def reset_session(self, session_id: str) -> None:
           self._session_data.pop(session_id, None)

       async def shutdown(self) -> None:
           self._session_data.clear()
           self._initialized = False
   ```

3. **Export in the engine's `__init__.py`**

   ```python
   from openbias.policy.engines.<name>.engine import MyPolicyEngine

   __all__ = ["MyPolicyEngine"]
   ```

4. **Register the import** in `openbias/policy/engines/__init__.py`

   Add a line alongside existing imports:
   ```python
   from openbias.policy.engines import fsm, nemo, llm, judge, <name>
   ```

5. **(Optional) Create a compiler** — see `openbias/policy/compiler/` for the `PolicyCompiler` ABC and `LLMPolicyCompiler` base class. Use `@register_compiler("<name>")`. Wire it via `get_compiler()` on your engine.

6. **Add a config example** for `openbias.yaml`

   ```yaml
   engine: <name>
   <name>:
     some_option: value
   ```

7. **Write tests** in `tests/policy/engines/<name>/`

   At minimum: initialization, evaluate_request with ALLOW result, evaluate_response with INTERVENE/BLOCK result, session reset.

## Anti-patterns

- **Forgetting `self._initialized = True`** in `initialize()` — the `@require_initialized` decorator will reject all evaluate calls.
- **Blocking in evaluate methods** — All evaluate methods are `async`. Use `await` for I/O. Never block the event loop.
- **Not cleaning up sessions** — Implement `reset_session` properly. The interceptor calls this for TTL-expired sessions.
- **Mutating `request_data`** — Always work on copies. The interceptor may retry with the original.
- **Returning bare strings instead of `EvaluationResult`** — Always return `EvaluationResult(status=..., violations=[...])`.

## Reference Files

| File | What to look at |
|------|----------------|
| `openbias/policy/protocols.py` | `PolicyEngine` ABC, `EvaluationResult`, `EvaluationStatus`, `ViolationRecord` |
| `openbias/policy/engines/stateful.py` | `StatefulPolicyEngine` ABC, `StateClassificationResult` dataclass |
| `openbias/policy/registry.py` | `@register_engine` decorator, `PolicyEngineRegistry` |
| `openbias/policy/engines/__init__.py` | Where to add your import |
| `openbias/policy/engines/fsm/` | Stateful engine example (smallest engine at ~340 lines) |
| `openbias/policy/engines/nemo/` | Stateless engine wrapping an external library |
| `openbias/policy/engines/judge/` | Judge engine example with LLM calls and compiled-rule evaluation |
