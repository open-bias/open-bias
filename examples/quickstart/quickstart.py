"""
Open Bias — Quickstart

The smallest possible example. 3 policy rules, ~30 lines of client code.
Takes 60 seconds to run if you already have an API key for any provider.

What's happening:
  Your OpenAI-compatible client talks to Open Bias (localhost:4000)
  instead of the LLM provider directly. Open Bias forwards the call
  via LiteLLM, then asynchronously evaluates the response against your
  policy rules using a sidecar LLM (the "judge"). If the response
  violates a rule, Open Bias queues an intervention for the next turn.

  The proxy adds zero latency to the critical path — the response comes
  back to your app immediately while the judge evaluates in the background.

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

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain how DNS resolution works, step by step."},
    ],
    extra_headers={"X-OpenBias-Session-ID": "quickstart-001"},
)

print(response.choices[0].message.content)

# That's it. Every call now runs through your policy rules (see openbias.yaml).
# Check the openbias server logs to see the judge evaluation results.
