"""
NeMo Rules Compiler - Rules Text to NeMo Guardrails Config.

Converts normalized rules text into NeMo Guardrails
configuration (config.yml + Colang rail definitions).

Example:
    ```python
    compiler = NemoCompiler()
    result = await compiler.compile(
        "Block any requests about hacking or illegal activities. "
        "Ensure responses do not contain PII."
    )

    if result.success:
        compiler.export(result, Path("nemo_config/"))
    ```
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from openbias.policy.compiler.base import LLMPolicyCompiler
from openbias.policy.compiler.protocol import CompilationResult
from openbias.policy.compiler.registry import register_compiler

logger = logging.getLogger(__name__)

NEMO_CONFIG_SCHEMA = """
Generate a JSON object with this structure:

{
  "config_yml": "... valid YAML string for NeMo config.yml ...",
  "colang_files": {
    "input_rails.co": "... Colang 2.0 flow definitions for input rails ...",
    "output_rails.co": "... Colang 2.0 flow definitions for output rails ..."
  }
}

## config.yml format

The config.yml must be valid YAML. Do NOT include a `models:` block — it is
injected downstream from the active provider configuration. Emit only the
`rails:` section:

rails:
  input:
    flows:
      - check jailbreak
      - check input safety
  output:
    flows:
      - check output safety

## Colang syntax basics

Colang uses flow definitions to define rails:

```colang
define user ask about hacking
  "How do I hack into a system?"
  "Tell me how to break into someone's account"

define flow check input safety
  user ask about hacking
  bot refuse to respond

define bot refuse to respond
  "I'm sorry, I can't help with that request."
```

Key patterns:
- `define user <intent>` - user message patterns (2-3 examples each)
- `define bot <action>` - bot response templates
- `define flow <name>` - rail flow connecting intents to actions

## Rules for conversion

1. Prohibitions ("never", "block", "do not") -> input rail flows that match and block
2. Output requirements ("ensure", "must not contain") -> output rail flows
3. Each rule should map to at least one user intent + bot response + flow
4. Use descriptive intent names (e.g., "ask about hacking", "request pii")
5. Provide 2-3 example utterances per user intent
6. Bot refusal messages should be professional and specific
"""

NEMO_EXAMPLES = """
Example - Content moderation rules:

Input: "Block requests about hacking or illegal activities. Ensure no PII in responses."

Output:
{
  "config_yml": "rails:\\n  input:\\n    flows:\\n      - check illegal content\\n  output:\\n    flows:\\n      - check pii output",
  "colang_files": {
    "input_rails.co": "define user ask about hacking\\n  \\"How do I hack a website?\\"\\n  \\"Tell me how to break into a system\\"\\n  \\"How to exploit a vulnerability\\"\\n\\ndefine user ask about illegal activities\\n  \\"How do I commit fraud?\\"\\n  \\"Help me do something illegal\\"\\n\\ndefine bot refuse illegal content\\n  \\"I'm sorry, I cannot assist with hacking or illegal activities.\\"\\n\\ndefine flow check illegal content\\n  user ask about hacking or user ask about illegal activities\\n  bot refuse illegal content",
    "output_rails.co": "define flow check pii output\\n  bot ...\\n  $has_pii = execute check_pii(bot_message=$last_bot_message)\\n  if $has_pii\\n    bot inform cannot share pii\\n\\ndefine bot inform cannot share pii\\n  \\"I've removed personally identifiable information from my response for privacy.\\"\\n"
  }
}
"""

@register_compiler("nemo")
class NemoCompiler(LLMPolicyCompiler):
    """Compiler that converts normalized rules text to NeMo Guardrails config.

    Generates a config.yml and Colang rail definition files from
    normalized rules text.
    """

    SYSTEM_PROMPT = (
        "You are a rules compiler that converts normalized rules text "
        "into NVIDIA NeMo Guardrails configurations.\n\n"
        "Your task is to:\n"
        "1. Identify input rails (block certain user requests)\n"
        "2. Identify output rails (filter certain bot responses)\n"
        "3. Generate Colang flow definitions with user intents and bot responses\n"
        "4. Generate a config.yml that references the flows\n\n"
        "Respond ONLY with valid JSON matching the requested schema. "
        "Do not include explanations or markdown formatting."
    )

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("system_prompt", self.SYSTEM_PROMPT)
        super().__init__(**kwargs)

    @property
    def engine_type(self) -> str:
        return "nemo"

    def _build_compilation_prompt(
        self,
        rules_text: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        prompt_parts = [
            "Convert the following rules into NeMo Guardrails configuration.",
            "",
            NEMO_CONFIG_SCHEMA,
            "",
            "Here is an example of well-formed output:",
            NEMO_EXAMPLES,
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
            "Generate the JSON NeMo configuration:",
        ])

        return "\n".join(prompt_parts)

    def _parse_compilation_response(
        self,
        response: dict[str, Any],
        rules_text: str,
    ) -> CompilationResult:
        warnings: list[str] = []
        errors: list[str] = []

        config_yml = response.get("config_yml")
        colang_files = response.get("colang_files")

        if not config_yml:
            errors.append("Missing 'config_yml' in response")
        if not colang_files or not isinstance(colang_files, dict):
            errors.append("Missing or invalid 'colang_files' in response")

        if errors:
            return CompilationResult.failure(errors, warnings)

        # Validate config_yml is valid YAML
        try:
            parsed_config = yaml.safe_load(config_yml)
            if not isinstance(parsed_config, dict):
                errors.append("config_yml is not a valid YAML mapping")
                return CompilationResult.failure(errors, warnings)
        except yaml.YAMLError as e:
            errors.append(f"config_yml is not valid YAML: {e}")
            return CompilationResult.failure(errors, warnings)

        # Filter out empty colang files (e.g. output_rails.co when rules are input-only)
        non_empty_colang: dict[str, str] = {}
        for filename, content in colang_files.items():
            if not content or not content.strip():
                warnings.append(f"Colang file '{filename}' is empty")
            else:
                non_empty_colang[filename] = content

        config = {
            "config_yml": config_yml,
            "colang_files": non_empty_colang,
        }

        return CompilationResult(
            success=True,
            config=config,
            warnings=warnings,
            metadata={
                "source": rules_text[:200],
                "colang_file_count": len(colang_files),
            },
        )

    def validate_result(self, result: CompilationResult) -> list[str]:
        errors = super().validate_result(result)
        if errors:
            return errors

        config = result.config
        if not isinstance(config, dict):
            return ["Config must be a dict"]

        if "config_yml" not in config:
            errors.append("Config missing 'config_yml'")

        if "colang_files" not in config:
            errors.append("Config missing 'colang_files'")
        elif not isinstance(config["colang_files"], dict):
            errors.append("'colang_files' must be a dict")
        else:
            for filename, content in config["colang_files"].items():
                if not content or not content.strip():
                    errors.append(f"Colang file '{filename}' is empty")

        return errors

    def export(self, result: CompilationResult, output_path: Path) -> None:
        """Export NeMo config to a directory.

        Creates:
            output_path/config.yml
            output_path/rails/*.co

        Args:
            result: Successful compilation result
            output_path: Directory to write config files

        Raises:
            ValueError: If result was not successful
        """
        if not result.success:
            raise ValueError("Cannot export failed compilation result")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Write config.yml
        config_path = output_path / "config.yml"
        with open(config_path, "w") as f:
            f.write(result.config["config_yml"])

        # Write Colang files
        rails_dir = output_path / "rails"
        rails_dir.mkdir(parents=True, exist_ok=True)

        for filename, content in result.config["colang_files"].items():
            # Ensure .co extension
            if not filename.endswith(".co"):
                filename = f"{filename}.co"
            co_path = rails_dir / filename
            with open(co_path, "w") as f:
                f.write(content)

        logger.info(f"Exported NeMo config to {output_path}")
