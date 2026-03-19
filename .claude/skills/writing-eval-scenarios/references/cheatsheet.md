# Eval Cheatsheet

## Mock Response Formats

### Judge Engine

**ALLOW (pass):**
```json
{"scores": [{"criterion": "policy_compliance", "score": 1, "max_score": 1, "reasoning": "Response follows all policies"}], "summary": "All policies followed"}
```

**BLOCK (fail):**
```json
{"scores": [{"criterion": "policy_compliance", "score": 0, "max_score": 1, "reasoning": "Specific reason for failure"}], "summary": "Policy violation: brief description"}
```

### NeMo Engine

Mock responses replace `_rails.generate_async`. Return the string the guardrails would produce:

**ALLOW:** Return the original assistant message content.
**BLOCK:** Return `"I'm sorry, I can't respond to that."` (or similar refusal).

## Assertion Patterns

```python
from opensentinel.policy.protocols import Decision

# Single turn check
assert result.turns[0].response_eval.decision == Decision.ALLOW
assert result.turns[1].response_eval.decision == Decision.BLOCK

# All turns should pass
assert all(t.response_eval.decision == Decision.ALLOW for t in result.turns)

# At least one violation detected
assert any(t.response_eval.decision != Decision.ALLOW for t in result.turns)

# Check violation message
assert "policy violation" in result.turns[0].response_eval.message.lower()

# No errors
assert result.error is None
```

## Message Role Reference

| Role | When to use |
|------|------------|
| `system` | First message, sets agent instructions |
| `user` | Human input |
| `assistant` | LLM response (marks turn boundary) |
| `tool` | Tool execution result, must follow `tool_calls` |

## Tool Call Format

```json
{
  "role": "assistant",
  "content": "Optional text before tool use",
  "tool_calls": [
    {
      "id": "call_<unique_id>",
      "type": "function",
      "function": {
        "name": "function_name",
        "arguments": "{\"key\": \"value\"}"
      }
    }
  ]
}
```

`arguments` is a **JSON string**, not an object.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Missing `tool` message after `tool_calls` | Add `{"role": "tool", "tool_call_id": "call_XXX", "content": "..."}` |
| `arguments` as object | Wrap in `JSON.stringify` / `json.dumps` — must be a string |
| Mock count mismatch | Count turns across ALL scenarios (alphabetical order) |
| Non-deterministic assertions | Always use `apply_mock_provider` |
