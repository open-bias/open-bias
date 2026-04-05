"""
Open Bias — Workflow Enforcement (FSM Engine)

Demonstrates deterministic workflow enforcement using a finite state machine
with LTL-lite temporal constraints. A customer support agent is required to
verify identity before processing account actions.

Architecture (what happens on each call):
  1. pre_call_hook: checks for pending interventions from previous violations.
     If the FSM queued one, it's applied here (system prompt amendment, context
     reminder, or hard block — configurable in the workflow compiled from
     `RULES.md`).
  2. LLM call: forwarded to provider. Unmodified.
  3. post_call_hook: the FSM engine classifies the response to a workflow state
     using a three-tier cascade:
       a. Tool call matching: if the response includes a function call, match
          the function name against state definitions. Confidence = 1.0. ~0ms.
       b. Regex patterns: run re.search() against state patterns. Conf = 0.85. ~1ms.
       c. Semantic embeddings: cosine similarity via sentence-transformers against
          state exemplars. Confidence ∝ similarity. ~50ms on CPU. ONNX available.
     First confident match wins — no unnecessary computation.
  4. The constraint evaluator checks all active LTL-lite constraints against
     the state history. If a precedence constraint is violated (e.g., "identity
     verification must happen before account_action"), the intervention handler
     schedules a correction.
  5. The state machine records the transition.

What to watch for:
  - The agent tries to process a refund WITHOUT verifying identity first.
  - The FSM catches the precedence violation and injects a system prompt amendment.
  - Next turn, the agent course-corrects and asks for verification.

Provider-agnostic:
  Set exactly ONE of these env vars:
    export OPENAI_API_KEY=...      → uses gpt-4o-mini
    export GEMINI_API_KEY=...      → uses gemini/gemini-2.5-flash
    export ANTHROPIC_API_KEY=...   → uses anthropic/claude-sonnet-4-5

Run:
  cd examples/fsm_workflow
  export <PROVIDER>_API_KEY=...
  openbias serve                      # terminal 1
  python workflow_enforcement.py       # terminal 2
"""

import os
import sys
import json
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
SESSION_ID = "fsm-demo-001"

client = OpenAI(base_url=PROXY_URL, api_key=API_KEY)

print(f"Using model: {MODEL}\n")

# -- Tools -------------------------------------------------------------------
# These function names are what the FSM's tool-call classifier keys on.
# The compiled FSM workflow includes an "account_action"-style state with
# `classification.tool_calls` entries such as `process_refund` and
# `update_subscription`. When the LLM emits one of those tool calls, the
# classifier immediately assigns that state with confidence 1.0 — no regex or
# embeddings.

tools = [
    {
        "type": "function",
        "function": {
            "name": "verify_identity",
            "description": "Verify customer identity before performing account actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "verification_method": {"type": "string", "enum": ["email", "payment_last_four"]},
                    "verification_value": {"type": "string"},
                },
                "required": ["customer_id", "verification_method", "verification_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the knowledge base for help articles.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_refund",
            "description": "Process a refund for a customer. Requires identity verification first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["customer_id", "amount", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate to a human support agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
                },
                "required": ["reason"],
            },
        },
    },
]

# -- Simulated tool execution ------------------------------------------------
# In production, these would be real API calls. Here we simulate them
# to show the FSM enforcement — what matters is the function NAMES,
# which the classifier uses for state assignment.

VERIFIED = False

def execute_tool(name: str, args: dict) -> str:
    global VERIFIED
    if name == "verify_identity":
        VERIFIED = True
        return f"Identity verified for {args.get('customer_id', 'unknown')}."
    elif name == "search_knowledge_base":
        return "Refunds are processed within 5-7 business days."
    elif name == "process_refund":
        if not VERIFIED:
            return "ERROR: Identity not verified."
        return f"Refund of ${args.get('amount', 0):.2f} processed."
    elif name == "escalate_to_human":
        return f"Escalated: {args.get('reason', '')}"
    return f"Unknown tool: {name}"

# -- Conversation ------------------------------------------------------------
# Designed to trigger the precedence constraint:
#   "identity_verification must precede account_action"
# The customer immediately asks for a refund without identifying themselves.

SYSTEM = (
    "You are a customer support agent for TechCo. "
    "Customer ID: customer-123. "
    "Always verify identity before processing refunds or account changes."
)

turns = [
    # Turn 1: classified as "greeting" → "identify_issue" via regex
    "Hi, I was charged twice last month. I need a refund of $29.99 right away.",

    # Turn 2: if the agent tries to call process_refund here, the FSM catches
    # the precedence violation (no identity_verification in state history)
    # and queues an intervention for the next turn.
    "Yes, just process the refund please. My customer ID is customer-123.",

    # Turn 3: the intervention fires — system prompt gets amended with
    # "you must verify identity before account actions". Agent should ask
    # for verification now.
    "Fine, my email is alice@example.com",

    # Turn 4: after verify_identity tool call, the FSM allows process_refund.
    "OK, now please process my refund.",
]

messages = [{"role": "system", "content": SYSTEM}]

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
            tools=tools,
            tool_choice="auto",
            extra_headers={"X-OpenBias-Session-ID": SESSION_ID},
        )
        msg = response.choices[0].message
        messages.append(msg)

        # Handle tool call loops — the agent may chain multiple tools per turn.
        while msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                print(f"  🔧 {tc.function.name}({json.dumps(args)})")
                result = execute_tool(tc.function.name, args)
                print(f"     → {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                extra_headers={"X-OpenBias-Session-ID": SESSION_ID},
            )
            msg = response.choices[0].message
            messages.append(msg)

        print(f"  ← Agent: {msg.content}")

    except Exception as e:
        if "violation" in str(e).lower():
            print(f"  🚫 Blocked: {e}")
        else:
            print(f"  ✗ Error: {e}")

print(f"\n{'━' * 70}")
print("  Done. Check openbias server logs for FSM state transitions + constraint checks.")
print(f"{'━' * 70}\n")
