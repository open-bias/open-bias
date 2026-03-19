---
name: new-intervention-strategy
description: Guide for adding a new intervention strategy to opensentinel/core/intervention/
---

# Adding a New Intervention Strategy

Intervention strategies modify the request message list when a policy violation is detected. The interceptor applies them to steer the LLM back on track (or block it entirely).

## Existing Strategies

| Strategy | What it does |
|----------|-------------|
| `system_prompt_append` | Appends `[WORKFLOW GUIDANCE]: {value}` to the system message |
| `user_message_inject` | Inserts a `[System Note]: {value}` user message after the last user message |
| `response_modification` | Modifies the LLM response directly (append warning or replace content) |

## Checklist

1. **Add to `StrategyType` enum** in `opensentinel/core/intervention/strategies.py`

   ```python
   class StrategyType(Enum):
       SYSTEM_PROMPT_APPEND = "system_prompt_append"
       USER_MESSAGE_INJECT = "user_message_inject"
       RESPONSE_MODIFICATION = "response_modification"
       MY_STRATEGY = "my_strategy"  # Add here
   ```

2. **Create the strategy class** in `strategies.py`

   ```python
   class MyStrategy(InterventionStrategy):
       @staticmethod
       def merge(
           messages: list[dict[str, Any]],
           value: str,
       ) -> list[dict[str, Any]]:
           # CRITICAL: Copy the list, never mutate the input
           result = [dict(m) for m in messages]
           # Modify result as needed
           return result

       def apply(
           self,
           data: dict,
           config: InterventionConfig,
           context: dict[str, Any],
       ) -> dict:
           message = self.format_message(config.message_template, context)
           result = dict(data)
           result["messages"] = self.merge(result.get("messages", []), message)
           return result
   ```

3. **Export in `__init__.py`** — add your class to `opensentinel/core/intervention/__init__.py`

4. **Wire into the interceptor** — in `opensentinel/core/interceptor/interceptor.py`, update `_apply_intervention`:

   ```python
   def _apply_intervention(self, request_data, message, strategy=None):
       result = dict(request_data)
       messages = result.get("messages", [])
       effective_strategy = strategy or self._default_strategy

       if effective_strategy == "user_message_inject":
           result["messages"] = UserMessageInjectStrategy.merge(messages, message)
       elif effective_strategy == "my_strategy":
           result["messages"] = MyStrategy.merge(messages, message)
       else:
           result["messages"] = SystemPromptAppendStrategy.merge(messages, message)

       return result
   ```

5. **Add to config** — in `opensentinel/config/settings.py`, update the `InterventionConfig.default_strategy` Literal:

   ```python
   default_strategy: Literal[
       "system_prompt_append",
       "user_message_inject",
       "my_strategy",
   ] = "user_message_inject"
   ```

6. **Write tests** in `tests/core/test_intervention.py`

   Test both `merge()` (static, operates on message lists) and `apply()` (uses config + context). Verify:
   - Original messages list is not mutated
   - Strategy inserts/modifies the right messages
   - Template formatting works with context variables
   - Edge cases: empty messages list, missing system message, no user messages

## Key Constraint

**`merge()` must never mutate the input.** Always copy the messages list and individual message dicts before modifying. The interceptor may call merge multiple times with the same input (e.g., retries, multiple violations).

```python
# WRONG — mutates input
def merge(messages, value):
    messages.append({"role": "user", "content": value})
    return messages

# RIGHT — copies first
def merge(messages, value):
    result = [dict(m) for m in messages]
    result.append({"role": "user", "content": value})
    return result
```

## Anti-patterns

- **Mutating the input messages list** — Will cause bugs when the same list is reused.
- **Forgetting to wire into the interceptor** — Your strategy exists but is never called.
- **Not handling missing system/user messages** — Don't assume the messages list contains any particular roles.
- **Ignoring `format_message`** — Use `self.format_message(template, context)` for template variable substitution. It handles missing keys gracefully.

## Reference Files

| File | What to look at |
|------|----------------|
| `opensentinel/core/intervention/strategies.py` | `InterventionStrategy` ABC, existing implementations, `StrategyType` enum |
| `opensentinel/core/interceptor/interceptor.py` | `_apply_intervention()` method — where strategies are dispatched |
| `opensentinel/config/settings.py` | `InterventionConfig` — `default_strategy` Literal type |
| `tests/core/test_intervention.py` | Existing tests to follow as patterns |
