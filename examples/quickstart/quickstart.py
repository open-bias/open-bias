"""
Open Bias — Quickstart

A customer support agent with 3 rules. Two turns:
  - Turn 1: normal question → passes, agent answers.
  - Turn 2: refund request on an old order → blocked immediately.

The proxy runs in sync mode so violations surface as a blocked response
right here in the terminal — no background evaluation to wait for.

Provider-agnostic:
  Set exactly ONE of these env vars. The example auto-detects the model.
    export OPENAI_API_KEY=...      → uses gpt-4o-mini
    export GEMINI_API_KEY=...      → uses gemini/gemini-2.5-flash
    export ANTHROPIC_API_KEY=...   → uses anthropic/claude-sonnet-4-5

Run:
  cd examples/quickstart
  export <PROVIDER>_API_KEY=...    # pick one
  openbias serve                  # terminal 1
  python quickstart.py             # terminal 2
"""

import os
import sys
from openai import OpenAI


def detect_model():
    """Auto-detect model from whichever API key is set."""
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini", os.environ["OPENAI_API_KEY"]
    if os.getenv("GEMINI_API_KEY"):
        return "gemini/gemini-2.5-flash", os.environ["GEMINI_API_KEY"]
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini/gemini-2.5-flash", os.environ["GOOGLE_API_KEY"]
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-5", os.environ["ANTHROPIC_API_KEY"]
    return None, None


model, api_key = detect_model()
if not model:
    print("Set one of: OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY")
    sys.exit(1)

client = OpenAI(
    base_url="http://localhost:4000/v1",  # ← only change vs. calling the LLM directly
    api_key=api_key,
)

print(f"Using model: {model}\n")

messages = [
    {"role": "system", "content": (
        "You are a helpful customer support agent for Acme Corp. "
        "You assist with product questions, account issues, and order status."
    )}
]

turns = [
    # Turn 1: normal support question — passes all rules
    "How do I reset my password?",

    # Turn 2: refund on a 3-month-old order — blocked by the
    # "no refunds older than 30 days" rule. Sync mode means the
    # violation surfaces immediately as a blocked response.
    "I want a refund for an order I placed 3 months ago.",
]

for i, user_input in enumerate(turns, 1):
    print(f"\n{'━' * 60}")
    print(f"  Turn {i}")
    print(f"{'━' * 60}")
    print(f"\n  → Customer: {user_input}\n")

    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers={"X-OpenBias-Session-ID": "quickstart-001"},
        )
        reply = response.choices[0].message
        print(f"  ← Agent: {reply.content}")
        messages.append(reply)

    except Exception as e:
        if "violation" in str(e).lower() or "blocked" in str(e).lower():
            print(f"  🚫 Blocked by rules: {e}")
        else:
            print(f"  ✗ Error: {e}")

print(f"\n{'━' * 60}")
print("  Done. Check the openbias server logs for judge evaluation details.")
print(f"{'━' * 60}\n")
