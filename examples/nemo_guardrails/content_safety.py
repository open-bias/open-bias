"""
Open Bias — Content Safety Rails (NeMo Engine)

Demonstrates NVIDIA NeMo Guardrails as an evaluator engine. NeMo provides
content safety rails (jailbreak detection, PII filtering, toxicity) and
programmable dialog flows via Colang.

Architecture (what happens on each call):
  1. pre_call_hook: messages are passed through NeMo's INPUT rails.
     If NeMo generates a refusal response (matched against known refusal
     markers: "i cannot", "i'm not able to", etc.), the request is blocked
     before it ever reaches the LLM.
  2. LLM call: if input rails pass, forwarded to provider. Unmodified.
  3. post_call_hook: the full conversation (including the agent's response)
     runs through NeMo's OUTPUT rails. If NeMo blocks it, the response
     is denied.

Fail-open by default:
  If NeMo evaluation throws an exception, the request/response passes
  through with a warning logged. Set `nemo.fail_closed: true` in config
  to block on errors instead. Rationale: a monitoring layer that takes
  down production is worse than one that misses a violation.

Prerequisites:
  pip install 'openbias[nemo]'

Note on providers:
  Open Bias compiles the example's `rules.md` into NeMo runtime artifacts
  internally. The MODEL below is for the agent's LLM — it goes through
  Open Bias's proxy. The sidecar rails evaluation model is configured by
  the compiled runtime, not as a separate user-authored input surface.

Run:
  cd examples/nemo_guardrails
  export <PROVIDER>_API_KEY=...
  openbias serve
  python content_safety.py
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


MODEL, API_KEY = detect_model()
if not MODEL:
    print("Set one of: OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY")
    sys.exit(1)

# -- Config ------------------------------------------------------------------
PROXY_URL = os.getenv("OBIAS_URL", "http://localhost:4000/v1")
SESSION_ID = "nemo-demo-001"

client = OpenAI(
    base_url=PROXY_URL,
    api_key=API_KEY,
    default_headers={"x-openbias-session-id": SESSION_ID},
)

print(f"Using model: {MODEL}\n")

messages = [
    {"role": "system", "content": (
        "You are a helpful customer support agent for TechCo. "
        "You help with refunds, subscriptions, and general questions."
    )}
]

# -- Conversation turns designed to trigger NeMo rails -------------------------
turns = [
    # Turn 1: benign — passes both input and output rails
    "Hi, I need help with my subscription.",

    # Turn 2: off-topic — should be caught by NeMo's topical rails
    # (if configured in the Colang flows under config/)
    "What stocks should I invest in right now?",

    # Turn 3: back on-topic — should pass
    "OK nevermind. Can I get a refund for last month?",
]

for i, user_input in enumerate(turns, 1):
    print(f"\n{'━' * 70}")
    print(f"  Turn {i}")
    print(f"{'━' * 70}")
    print(f"\n  → Customer: {user_input}\n")

    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        reply = response.choices[0].message
        print(f"  ← Agent: {reply.content}")
        messages.append(reply)

    except Exception as e:
        if "blocked" in str(e).lower() or "violation" in str(e).lower():
            print(f"  🚫 Blocked by NeMo rail: {e}")
        else:
            print(f"  ✗ Error: {e}")

print(f"\n{'━' * 70}")
print("  Done. Check openbias server logs for NeMo rail evaluations.")
print(f"{'━' * 70}\n")
