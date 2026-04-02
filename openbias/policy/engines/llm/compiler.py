"""
LLM Rules Compiler - Rules Text to WorkflowDefinition.

Converts normalized rules text into a WorkflowDefinition
(the same schema used by the FSM engine) via an LLM call.

Example:
    ```python
    compiler = LLMCompiler()
    result = await compiler.compile(
        "Greet the customer. Verify identity before account actions. "
        "Never share internal system information."
    )

    if result.success:
        compiler.export(result, Path("workflow.yaml"))
    ```
"""

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from openbias.policy.compiler.base import LLMPolicyCompiler
from openbias.policy.compiler.protocol import CompilationResult
from openbias.policy.compiler.registry import register_compiler
from openbias.policy.engines.fsm.workflow.schema import WorkflowDefinition

logger = logging.getLogger(__name__)

LLM_WORKFLOW_SCHEMA = """
Generate a JSON object representing a WorkflowDefinition with this structure:

{
  "name": "workflow-name",
  "description": "Brief description of the workflow",
  "states": [
    {
      "name": "state_name",
      "description": "What happens in this state",
      "is_initial": true/false,
      "is_terminal": true/false,
      "is_error": false,
      "classification": {
        "tool_calls": ["tool_name"],
        "patterns": ["regex pattern"],
        "exemplars": ["example phrase the LLM might say in this state"]
      }
    }
  ],
  "transitions": [
    {
      "from_state": "state_a",
      "to_state": "state_b",
      "description": "When this transition happens"
    }
  ],
  "constraints": [
    {
      "name": "constraint_name",
      "type": "precedence|never|eventually|response",
      "description": "What this constraint enforces",
      "trigger": "state_name",
      "target": "state_name",
      "message": "Violation message shown when constraint is broken"
    }
  ]
}

## State names
- Use lowercase snake_case identifiers (e.g. "greeting", "verify_identity")
- Exactly one state must have "is_initial": true
- Terminal states represent successful completion

## Classification hints
- tool_calls: Function/tool names the agent calls in this state
- patterns: Regex patterns to match in the agent's response text
- exemplars: 2-3 example phrases the agent might produce in this state

## Constraint types
- precedence: target state must be visited BEFORE trigger state (use for "verify X before doing Y")
- never: target state must NEVER be reached (use for prohibitions like "never share internal info")
- eventually: target state must be reached at some point (use for requirements like "must eventually close the ticket")
- response: if trigger state is reached, target must eventually follow (use for conditional requirements)

## Conversion heuristics
- Ordering language ("before", "first", "prior to") → PRECEDENCE constraint
- Prohibition language ("never", "block", "do not", "must not") → NEVER constraint
- Requirement language ("must", "always", "ensure") → EVENTUALLY constraint
- Conditional language ("if X then Y", "when X ensure Y") → RESPONSE constraint
"""

LLM_WORKFLOW_EXAMPLES = """
Example 1 - Customer support rules:

Input: "Greet the customer. Verify identity before any account action. Never share internal system information. Must resolve the issue before closing."

Output:
{
  "name": "customer-support",
  "description": "Customer support workflow with identity verification and information controls",
  "states": [
    {"name": "greeting", "description": "Initial customer greeting", "is_initial": true, "classification": {"exemplars": ["Hello! How can I help you today?", "Welcome, how may I assist you?"]}},
    {"name": "identify_issue", "description": "Understanding the customer's problem", "classification": {"exemplars": ["I understand you're having trouble with...", "Let me look into that for you"]}},
    {"name": "verify_identity", "description": "Verifying customer identity", "classification": {"tool_calls": ["verify_identity"], "exemplars": ["I'll need to verify your identity first", "Can you confirm your account details?"]}},
    {"name": "account_action", "description": "Performing account-level actions", "classification": {"tool_calls": ["update_account", "reset_password"], "exemplars": ["I've updated your account", "Your password has been reset"]}},
    {"name": "resolve_issue", "description": "Resolving the customer's issue", "classification": {"exemplars": ["That should fix the issue", "The problem has been resolved"]}},
    {"name": "closing", "description": "Closing the conversation", "is_terminal": true, "classification": {"exemplars": ["Is there anything else I can help with?", "Thank you for contacting us"]}}
  ],
  "transitions": [
    {"from_state": "greeting", "to_state": "identify_issue"},
    {"from_state": "identify_issue", "to_state": "verify_identity"},
    {"from_state": "verify_identity", "to_state": "account_action"},
    {"from_state": "identify_issue", "to_state": "resolve_issue"},
    {"from_state": "account_action", "to_state": "resolve_issue"},
    {"from_state": "resolve_issue", "to_state": "closing"}
  ],
  "constraints": [
    {"name": "verify_before_account", "type": "precedence", "trigger": "account_action", "target": "verify_identity", "message": "You must verify the customer's identity before performing account actions."},
    {"name": "no_internal_info", "type": "never", "target": "share_internal_info", "message": "Never share internal system information with the customer."},
    {"name": "must_resolve", "type": "eventually", "target": "resolve_issue", "message": "The customer's issue must be resolved before closing."}
  ]
}

Example 2 - Simple content moderation:

Input: "Block requests about hacking. Ensure responses are professional."

Output:
{
  "name": "content-moderation",
  "description": "Content moderation with topic blocking and tone requirements",
  "states": [
    {"name": "processing", "description": "Processing user request", "is_initial": true, "is_terminal": true, "classification": {"exemplars": ["Let me help you with that", "Here's the information you requested"]}}
  ],
  "transitions": [],
  "constraints": [
    {"name": "block_hacking", "type": "never", "target": "hacking_discussion", "message": "Requests about hacking must be blocked."},
    {"name": "professional_tone", "type": "eventually", "target": "processing", "message": "All responses must maintain a professional tone."}
  ]
}
"""


@register_compiler("llm")
class LLMCompiler(LLMPolicyCompiler):
    """Compiler that converts normalized rules text to a WorkflowDefinition.

    Produces the same WorkflowDefinition schema used by the FSM engine,
    allowing the LLM engine to accept rules/rules_file just like other engines.
    """

    SYSTEM_PROMPT = (
        "You are a rules compiler that converts normalized rules text "
        "into WorkflowDefinition configurations for an LLM-based policy engine.\n\n"
        "Your task is to:\n"
        "1. Identify workflow states (phases of the agent's operation)\n"
        "2. Define transitions between states\n"
        "3. Extract temporal constraints (precedence, never, eventually, response)\n"
        "4. Provide classification hints (tool calls, patterns, exemplars) for each state\n\n"
        "Respond ONLY with valid JSON matching the requested schema. "
        "Do not include explanations or markdown formatting."
    )

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("system_prompt", self.SYSTEM_PROMPT)
        super().__init__(**kwargs)

    @property
    def engine_type(self) -> str:
        return "llm"

    def _build_compilation_prompt(
        self,
        rules_text: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        prompt_parts = [
            "Convert the following rules into a WorkflowDefinition.",
            "",
            LLM_WORKFLOW_SCHEMA,
            "",
            "Here are examples of well-formed output:",
            LLM_WORKFLOW_EXAMPLES,
            "",
        ]

        if context:
            if context.get("domain"):
                prompt_parts.append(f"Domain: {context['domain']}")
            prompt_parts.append("")

        prompt_parts.extend([
            "Rules:",
            "---",
            rules_text,
            "---",
            "",
            "Generate the JSON WorkflowDefinition:",
        ])

        return "\n".join(prompt_parts)

    def _parse_compilation_response(
        self,
        response: dict[str, Any],
        rules_text: str,
    ) -> CompilationResult:
        warnings: list[str] = []
        errors: list[str] = []

        # Basic structure checks
        if "states" not in response or not response["states"]:
            errors.append("Response missing 'states' or states list is empty")
            return CompilationResult.failure(errors, warnings)

        # Defensive normalization
        self._normalize_response(response, warnings)

        # Ensure at least one initial state
        states = response.get("states", [])
        has_initial = any(s.get("is_initial") for s in states)
        if not has_initial and states:
            states[0]["is_initial"] = True
            warnings.append(
                f"No initial state found; marked '{states[0].get('name', 'first')}' as initial"
            )

        # Validate via Pydantic
        try:
            workflow = WorkflowDefinition.model_validate(response)
        except Exception as e:
            errors.append(f"WorkflowDefinition validation failed: {e}")
            return CompilationResult.failure(errors, warnings)

        return CompilationResult(
            success=True,
            config=workflow,
            warnings=warnings,
            metadata={
                "source": rules_text[:200],
                "state_count": len(workflow.states),
                "constraint_count": len(workflow.constraints),
            },
        )

    def _normalize_response(
        self, response: dict[str, Any], warnings: list[str]
    ) -> None:
        """Defensively normalize LLM output before Pydantic validation."""
        # Slugify state names
        for state in response.get("states", []):
            if "name" in state:
                state["name"] = self._slugify(state["name"])

        # Slugify state references in transitions
        for transition in response.get("transitions", []):
            if "from_state" in transition:
                transition["from_state"] = self._slugify(transition["from_state"])
            if "to_state" in transition:
                transition["to_state"] = self._slugify(transition["to_state"])

        # Slugify state references in constraints
        for constraint in response.get("constraints", []):
            if "name" in constraint:
                constraint["name"] = self._slugify(constraint["name"])
            if "trigger" in constraint and constraint["trigger"]:
                constraint["trigger"] = self._slugify(constraint["trigger"])
            if "target" in constraint and constraint["target"]:
                constraint["target"] = self._slugify(constraint["target"])

        # Strip unknown top-level keys
        allowed_keys = {"name", "description", "states", "transitions", "constraints", "metadata"}
        unknown = set(response.keys()) - allowed_keys
        for key in unknown:
            del response[key]
            warnings.append(f"Stripped unknown top-level key: '{key}'")

        # Strip unknown keys from states
        allowed_state_keys = {
            "name", "description", "is_initial", "is_terminal", "is_error",
            "classification", "max_duration_seconds",
        }
        for state in response.get("states", []):
            state_unknown = set(state.keys()) - allowed_state_keys
            for key in state_unknown:
                del state[key]

        # Strip unknown keys from classification hints
        allowed_cls_keys = {"tool_calls", "patterns", "exemplars", "min_similarity"}
        for state in response.get("states", []):
            cls = state.get("classification", {})
            if isinstance(cls, dict):
                cls_unknown = set(cls.keys()) - allowed_cls_keys
                for key in cls_unknown:
                    del cls[key]

        # Ensure name exists
        if "name" not in response or not response["name"]:
            response["name"] = "compiled-workflow"

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a state name to a valid slug identifier."""
        slug = name.lower().strip()
        slug = re.sub(r"[^a-z0-9_\-]", "_", slug)
        slug = re.sub(r"_+", "_", slug)
        slug = slug.strip("_")
        return slug or "unnamed"

    def validate_result(self, result: CompilationResult) -> list[str]:
        errors = super().validate_result(result)
        if errors:
            return errors

        config = result.config
        if not isinstance(config, WorkflowDefinition):
            return ["Config must be a WorkflowDefinition instance"]

        if not config.states:
            errors.append("Workflow has no states")

        if not config.get_initial_states():
            errors.append("Workflow has no initial state")

        return errors

    def export(self, result: CompilationResult, output_path: Path) -> None:
        """Export WorkflowDefinition to YAML file.

        Args:
            result: Successful compilation result
            output_path: Path to write the YAML file

        Raises:
            ValueError: If result was not successful
        """
        if not result.success:
            raise ValueError("Cannot export failed compilation result")

        workflow: WorkflowDefinition = result.config
        workflow_dict: dict[str, Any] = {
            "name": workflow.name,
        }
        if workflow.description:
            workflow_dict["description"] = workflow.description

        workflow_dict["states"] = []
        for state in workflow.states:
            sd: dict[str, Any] = {"name": state.name}
            if state.description:
                sd["description"] = state.description
            if state.is_initial:
                sd["is_initial"] = True
            if state.is_terminal:
                sd["is_terminal"] = True
            if state.is_error:
                sd["is_error"] = True

            cls: dict[str, Any] = {}
            if state.classification.tool_calls:
                cls["tool_calls"] = state.classification.tool_calls
            if state.classification.patterns:
                cls["patterns"] = state.classification.patterns
            if state.classification.exemplars:
                cls["exemplars"] = state.classification.exemplars
            if state.classification.min_similarity != 0.7:
                cls["min_similarity"] = state.classification.min_similarity
            if cls:
                sd["classification"] = cls
            workflow_dict["states"].append(sd)

        if workflow.transitions:
            workflow_dict["transitions"] = [
                {"from_state": t.from_state, "to_state": t.to_state}
                | ({"description": t.description} if t.description else {})
                for t in workflow.transitions
            ]

        if workflow.constraints:
            workflow_dict["constraints"] = []
            for c in workflow.constraints:
                cd: dict[str, Any] = {"name": c.name, "type": c.type.value}
                if c.description:
                    cd["description"] = c.description
                if c.trigger:
                    cd["trigger"] = c.trigger
                if c.target:
                    cd["target"] = c.target
                if c.message:
                    cd["message"] = c.message
                workflow_dict["constraints"].append(cd)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            yaml.dump(workflow_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info("Exported workflow to %s", output)
