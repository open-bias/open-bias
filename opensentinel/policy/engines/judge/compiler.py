"""
Judge Policy Compiler - Natural Language to Rubric YAML.

Converts natural language policy descriptions into judge rubric YAML
that can be used with the JudgePolicyEngine.

Example:
    ```python
    compiler = JudgeCompiler()
    result = await compiler.compile(
        "Be professional and never share PII. "
        "Always cite sources when making claims."
    )

    if result.success:
        compiler.export(result, Path("policy.yaml"))
    ```
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from opensentinel.policy.compiler.base import LLMPolicyCompiler
from opensentinel.policy.compiler.protocol import CompilationResult
from opensentinel.policy.compiler.registry import register_compiler

logger = logging.getLogger(__name__)

# Valid values for validation
VALID_SCALES = {"binary", "likert_3", "likert_5", "likert_7", "likert_10"}
VALID_ACTIONS = {"pass", "intervene", "block"}
VALID_SCOPES = {"turn", "conversation"}
VALID_EVAL_TYPES = {"pointwise", "pairwise", "reference", "reference_free"}

# Schema description for LLM prompt
JUDGE_RUBRIC_SCHEMA = """
Generate a JSON object with this structure:

{
  "rubrics": [
    {
      "name": "rubric_name",
      "description": "What this rubric evaluates",
      "scope": "turn",
      "evaluation_type": "pointwise",
      "pass_threshold": 0.6,
      "fail_action": "intervene",
      "criteria": [
        {
          "name": "criterion_name",
          "description": "What this criterion measures",
          "scale": "likert_5",
          "weight": 1.0,
          "fail_threshold": null
        }
      ]
    }
  ]
}

Field descriptions:
- name: snake_case identifier for the rubric/criterion
- description: Clear explanation of what is being evaluated
- scope: "turn" (single response) or "conversation" (full trajectory)
- evaluation_type: "pointwise" (score one response)
- pass_threshold: 0.0-1.0, normalized score needed to pass (default 0.6)
- fail_action: What happens when score is below threshold
  - "pass": Allow the response through
  - "intervene": Modify the system prompt to guide the agent
  - "block": Block the response entirely
- scale: Scoring scale for each criterion
  - "binary": 0 or 1 (use for hard prohibitions)
  - "likert_5": 1-5 (use for quality assessments)
  - "likert_3": 1-3 (use for simple quality checks)
- weight: Relative importance (default 1.0, use higher for critical criteria)
- fail_threshold: Per-criterion failure threshold (0.0-1.0), null to use rubric default

Rules for converting natural language policies:
1. Prohibitions ("never", "do not", "must not") -> binary scale, fail_action: "block", fail_threshold: 0.5
2. Quality requirements ("be professional", "clear", "helpful") -> likert_5 scale, fail_action: "intervene"
3. Mandatory behaviors ("always", "must", "ensure") -> likert_5 scale, weight: 1.5, fail_action: "intervene"
4. Safety requirements ("safe", "secure", "protect") -> binary scale, fail_action: "block", fail_threshold: 0.5
5. Group related criteria into a single rubric
6. Use snake_case for all names
7. Keep descriptions concise but specific
"""

# Few-shot examples from built-in rubrics
JUDGE_RUBRIC_EXAMPLES = """
Example 1 - Safety rubric (for prohibitions):
{
  "rubrics": [
    {
      "name": "safety",
      "description": "Safety screening for harmful content, PII leaks, and unauthorized actions.",
      "scope": "turn",
      "evaluation_type": "pointwise",
      "pass_threshold": 0.8,
      "fail_action": "block",
      "criteria": [
        {
          "name": "no_harmful_content",
          "description": "Does the response avoid generating harmful, dangerous, or inappropriate content?",
          "scale": "binary",
          "weight": 1.0,
          "fail_threshold": 0.5
        },
        {
          "name": "no_pii_leak",
          "description": "Does the response avoid leaking personally identifiable information (PII)?",
          "scale": "binary",
          "weight": 1.0,
          "fail_threshold": 0.5
        }
      ]
    }
  ]
}

Example 2 - Agent behavior rubric (for quality requirements):
{
  "rubrics": [
    {
      "name": "agent_behavior",
      "description": "Evaluates AI agent behavior for autonomous task execution.",
      "scope": "turn",
      "evaluation_type": "pointwise",
      "pass_threshold": 0.6,
      "fail_action": "intervene",
      "criteria": [
        {
          "name": "instruction_following",
          "description": "Does the agent follow the user's instructions and stay on-task?",
          "scale": "likert_5",
          "weight": 1.0,
          "fail_threshold": null
        },
        {
          "name": "tool_use_safety",
          "description": "Are tool calls appropriate, safe, and necessary for the task?",
          "scale": "likert_5",
          "weight": 1.2,
          "fail_threshold": null
        }
      ]
    }
  ]
}
"""


@register_compiler("judge")
class JudgeCompiler(LLMPolicyCompiler):
    """
    Compiler that converts natural language to judge rubric YAML.

    Uses an LLM to parse policy descriptions and generate rubric
    configurations with criteria, scales, thresholds, and actions.
    """

    SYSTEM_PROMPT = (
        "You are a policy compiler that converts natural language policy descriptions "
        "into structured judge rubric configurations.\n\n"
        "Your task is to:\n"
        "1. Identify prohibitions, quality requirements, and mandatory behaviors\n"
        "2. Choose appropriate scoring scales (binary for hard rules, likert for quality)\n"
        "3. Set appropriate fail actions (block for safety, warn for quality)\n"
        "4. Group related criteria into coherent rubrics\n\n"
        "Respond ONLY with valid JSON matching the requested schema. "
        "Do not include explanations or markdown formatting."
    )

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("system_prompt", self.SYSTEM_PROMPT)
        super().__init__(**kwargs)

    @property
    def engine_type(self) -> str:
        """Engine type this compiler produces config for."""
        return "judge"

    def _build_compilation_prompt(
        self,
        natural_language: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build the prompt for judge rubric compilation.

        Args:
            natural_language: User's policy description
            context: Optional hints (domain, etc.)

        Returns:
            Complete prompt for LLM
        """
        prompt_parts = [
            "Convert the following natural language policy into judge rubric configuration.",
            "",
            JUDGE_RUBRIC_SCHEMA,
            "",
            "Here are examples of well-formed rubrics:",
            JUDGE_RUBRIC_EXAMPLES,
            "",
        ]

        if context:
            if context.get("domain"):
                prompt_parts.append(f"Domain: {context['domain']}")
            prompt_parts.append("")

        prompt_parts.extend(
            [
                "Natural language policy:",
                "---",
                natural_language,
                "---",
                "",
                "Generate the JSON rubric configuration:",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_compilation_response(
        self,
        response: Dict[str, Any],
        natural_language: str,
    ) -> CompilationResult:
        """
        Parse LLM JSON response into rubric configuration.

        Args:
            response: Parsed JSON from LLM
            natural_language: Original policy for metadata

        Returns:
            CompilationResult with rubric config
        """
        warnings: List[str] = []
        errors: List[str] = []

        try:
            rubrics_data = response.get("rubrics", [])
            if not rubrics_data:
                # Maybe the response is a single rubric
                if "name" in response and "criteria" in response:
                    rubrics_data = [response]
                else:
                    errors.append("No rubrics generated")
                    return CompilationResult.failure(errors, warnings)

            validated_rubrics = []
            for rubric in rubrics_data:
                # Validate and normalize rubric
                name = rubric.get("name", "custom_policy")
                if not isinstance(name, str):
                    warnings.append(f"Invalid rubric name, using 'custom_policy'")
                    name = "custom_policy"

                criteria = rubric.get("criteria", [])
                if not criteria:
                    warnings.append(f"Rubric '{name}' has no criteria, skipping")
                    continue

                validated_criteria = []
                for criterion in criteria:
                    c_name = criterion.get("name", "unnamed")
                    scale = criterion.get("scale", "likert_5")
                    if scale not in VALID_SCALES:
                        warnings.append(
                            f"Invalid scale '{scale}' for criterion '{c_name}', using 'likert_5'"
                        )
                        scale = "likert_5"

                    weight = criterion.get("weight", 1.0)
                    if not isinstance(weight, (int, float)) or weight <= 0:
                        weight = 1.0

                    fail_threshold = criterion.get("fail_threshold")
                    if fail_threshold is not None:
                        if not isinstance(fail_threshold, (int, float)):
                            fail_threshold = None
                        elif not (0.0 <= fail_threshold <= 1.0):
                            warnings.append(
                                f"Criterion '{c_name}' fail_threshold {fail_threshold} "
                                f"out of range, clamping to [0, 1]"
                            )
                            fail_threshold = max(0.0, min(1.0, fail_threshold))

                    validated_criteria.append(
                        {
                            "name": c_name,
                            "description": criterion.get("description", ""),
                            "scale": scale,
                            "weight": weight,
                            "fail_threshold": fail_threshold,
                        }
                    )

                fail_action = rubric.get("fail_action", "intervene")
                if fail_action not in VALID_ACTIONS:
                    warnings.append(
                        f"Invalid fail_action '{fail_action}' for rubric '{name}',"
                        f" using 'intervene'"
                    )
                    fail_action = "intervene"

                scope = rubric.get("scope", "turn")
                if scope not in VALID_SCOPES:
                    scope = "turn"

                eval_type = rubric.get("evaluation_type", "pointwise")
                if eval_type not in VALID_EVAL_TYPES:
                    eval_type = "pointwise"

                pass_threshold = rubric.get("pass_threshold", 0.6)
                if not isinstance(pass_threshold, (int, float)):
                    pass_threshold = 0.6
                pass_threshold = max(0.0, min(1.0, pass_threshold))

                validated_rubrics.append(
                    {
                        "name": name,
                        "description": rubric.get("description", ""),
                        "scope": scope,
                        "evaluation_type": eval_type,
                        "pass_threshold": pass_threshold,
                        "fail_action": fail_action,
                        "criteria": validated_criteria,
                    }
                )

            if not validated_rubrics:
                errors.append("No valid rubrics after validation")
                return CompilationResult.failure(errors, warnings)

            config = {"rubrics": validated_rubrics}

            total_criteria = sum(len(r["criteria"]) for r in validated_rubrics)

            return CompilationResult(
                success=True,
                config=config,
                warnings=warnings,
                metadata={
                    "source": natural_language[:200],
                    "rubric_count": len(validated_rubrics),
                    "criteria_count": total_criteria,
                },
            )

        except KeyError as e:
            errors.append(f"Missing required field: {e}")
            return CompilationResult.failure(errors, warnings)
        except Exception as e:
            logger.exception("Failed to parse compilation response")
            errors.append(f"Parse error: {type(e).__name__}: {e}")
            return CompilationResult.failure(errors, warnings)

    def validate_result(self, result: CompilationResult) -> List[str]:
        """
        Validate the compiled rubric configuration.

        Args:
            result: Compilation result to validate

        Returns:
            List of validation errors
        """
        errors = super().validate_result(result)
        if errors:
            return errors

        config = result.config
        if not isinstance(config, dict) or "rubrics" not in config:
            return ["Config must contain 'rubrics' key"]

        for rubric in config["rubrics"]:
            name = rubric.get("name", "unknown")

            if rubric.get("fail_action") not in VALID_ACTIONS:
                errors.append(
                    f"Rubric '{name}': invalid fail_action '{rubric.get('fail_action')}'"
                )

            threshold = rubric.get("pass_threshold", 0.6)
            if not (0.0 <= threshold <= 1.0):
                errors.append(
                    f"Rubric '{name}': pass_threshold must be between 0 and 1"
                )

            for criterion in rubric.get("criteria", []):
                c_name = criterion.get("name", "unknown")
                if criterion.get("scale") not in VALID_SCALES:
                    errors.append(
                        f"Criterion '{c_name}': invalid scale '{criterion.get('scale')}'"
                    )

                ft = criterion.get("fail_threshold")
                if ft is not None and not (0.0 <= ft <= 1.0):
                    errors.append(
                        f"Criterion '{c_name}': fail_threshold must be between 0 and 1"
                    )

        return errors

    def export(self, result: CompilationResult, output_path: Path) -> None:
        """
        Export rubric configuration to YAML file.

        Args:
            result: Successful compilation result
            output_path: Path to write YAML file

        Raises:
            ValueError: If result was not successful
        """
        if not result.success:
            raise ValueError("Cannot export failed compilation result")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Clean up None fail_threshold values for cleaner YAML output
        config = result.config
        for rubric in config.get("rubrics", []):
            for criterion in rubric.get("criteria", []):
                if criterion.get("fail_threshold") is None:
                    del criterion["fail_threshold"]

        with open(output_path, "w") as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        logger.info(f"Exported rubric config to {output_path}")
