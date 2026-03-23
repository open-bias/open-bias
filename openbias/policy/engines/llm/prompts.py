"""
LLM prompt templates for the LLM Policy Engine.

Contains structured prompts for:
- State classification: Identifying which workflow state the agent is in
- Constraint evaluation: Checking soft constraints with evidence collection
"""

# =============================================================================
# STATE CLASSIFICATION PROMPTS
# =============================================================================

STATE_CLASSIFICATION_SYSTEM = """\
You are a workflow state classifier. Your task is to analyze an LLM agent's response \
and determine which workflow state it corresponds to.

The workflow "{workflow_name}" has the following states:

{states_block}

Current state: {current_state}

Instructions:
1. Analyze the assistant's message and any tool calls
2. Determine which state(s) the response could represent
3. Return a JSON array of candidates sorted by confidence (highest first)

Each candidate should have:
- "state_id": The state name (must match one of the defined states)
- "confidence": A number between 0.0 and 1.0
- "reasoning": Brief explanation of why this state matches

Consider:
- Tool calls are strong indicators of specific states
- Message content and intent
- The transition from the current state
- Exemplar phrases defined for each state

Return ONLY valid JSON, no other text."""

STATE_CLASSIFICATION_USER = """\
Classify this response:

Assistant message:
{assistant_message}

Tool calls:
{tool_calls_block}

Recent conversation context (last {window_size} turns):
{conversation_block}

Return a JSON array of state candidates sorted by confidence."""

# =============================================================================
# CONSTRAINT EVALUATION PROMPTS
# =============================================================================

CONSTRAINT_EVALUATION_SYSTEM = """\
You are a workflow constraint evaluator. Your task is to check whether an LLM agent's \
behavior violates any defined constraints.

Workflow: "{workflow_name}"
Current state: {current_state}
State history: {state_history}

Active constraints to evaluate:

{constraints_block}

Instructions:
1. Analyze the assistant's message and actions against each constraint
2. Determine if any constraints are violated
3. Look for contradictions in the agent's statements
4. Provide evidence for your evaluations

Return a JSON array with one object per constraint:
- "constraint_id": The constraint name
- "violated": true or false
- "confidence": How confident you are in this evaluation (0.0-1.0)
- "evidence": Quote or describe the specific content that supports your evaluation
- "severity": The severity if violated ("warning", "error", "critical")

Also check for:
- Contradictions with previous statements (add as special "contradiction_detected" constraint)
- Inconsistent information provided to the user

Return ONLY valid JSON, no other text."""

CONSTRAINT_EVALUATION_USER = """\
Evaluate constraints for this response:

Assistant message:
{assistant_message}

Tool calls:
{tool_calls_block}

Recent conversation:
{conversation_block}

Previous evidence for these constraints:
{evidence_block}

Return a JSON array of constraint evaluations."""
