"""
Open Bias — Financial Services Chatbot (NeMo Engine)

Demonstrates NVIDIA NeMo Guardrails as an evaluator engine on a financial
services scenario. A bank chatbot must never provide investment advice —
a real compliance requirement in regulated markets.

Architecture (what happens on each call):
  1. pre_call_hook: messages are passed through NeMo's INPUT rails.
     If NeMo generates a refusal, the request is blocked before the LLM
     ever sees it. Turn 2 below is stopped here.
  2. LLM call: if input rails pass, forwarded to provider. Unmodified.
  3. post_call_hook: the full conversation runs through NeMo's OUTPUT rails.
     If NeMo blocks it, the response is denied.

Fail-open by default:
  If NeMo evaluation throws, the request passes through with a warning.
  Set `nemo.fail_closed: true` in config to block on errors instead.

Prerequisites:
  pip install 'openbias[nemo]'

Provider-agnostic:
  Set exactly ONE of these env vars:
    export OPENAI_API_KEY=...      → uses gpt-4o-mini
    export GEMINI_API_KEY=...      → uses gemini/gemini-2.5-flash
    export GOOGLE_API_KEY=...      → uses gemini/gemini-2.5-flash (alias)
    export ANTHROPIC_API_KEY=...   → uses anthropic/claude-sonnet-4-5

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
    print("Set one of: OPENAI_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY")
    sys.exit(1)

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
        "You are a helpful customer service assistant for First National Bank. "
        "You help customers with account questions, transfers, and general banking support."
    )}
]

turns = [
    # Turn 1: on-topic banking question — passes input and output rails
    "Hi, can you help me check the status of my recent transfer?",

    # Turn 2: investment advice — blocked PRE-CALL by NeMo's input rail.
    # The LLM never sees this request. This is the key differentiator vs. the
    # judge engine: NeMo can block before the LLM call, not just after.
    "Should I buy Tesla stock right now? I have $10k to invest.",

    # Turn 3: back on-topic — passes both input and output rails
    "Got it. Can I move $500 from my checking to my savings account?",
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
        messages.append({"role": reply.role, "content": reply.content})

    except Exception as e:
        if "blocked" in str(e).lower() or "violation" in str(e).lower():
            print(f"  🚫 Blocked by NeMo rail: {e}")
        else:
            print(f"  ✗ Error: {e}")

print(f"\n{'━' * 70}")
print("  Done. Check openbias server logs for NeMo rail evaluations.")
print(f"{'━' * 70}\n")
