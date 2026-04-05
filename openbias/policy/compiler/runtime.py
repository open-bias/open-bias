"""Runtime rules-to-engine compilation used by `openbias serve`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openbias.policy.compiler import PolicyCompilerRegistry
from openbias.policy.rules import POLICY_FILENAME, resolve_project_rules_path, resolve_rules_payload


async def compile_runtime_config_for_evaluator(
    evaluator_name: str,
    evaluator_type: str,
    evaluator_config: dict[str, Any],
    *,
    default_model: str | None,
    base_dir: Path,
    rules_path: Path | None = None,
) -> dict[str, Any]:
    """Compile canonical rules inputs into engine-native runtime config."""
    normalized_rules = resolve_rules_payload(
        base_dir=base_dir,
        auto_discover_rules_md=True,
        rules_path=rules_path,
    )
    if not normalized_rules:
        expected_path = resolve_project_rules_path(base_dir)
        raise ValueError(
            f"Evaluator '{evaluator_name}' requires project {POLICY_FILENAME}. "
            f"Expected file at: {expected_path}"
        )

    cleaned = dict(evaluator_config)

    compiler_class = PolicyCompilerRegistry.get(evaluator_type)
    if compiler_class is None:
        raise ValueError(
            f"No compiler registered for evaluator type '{evaluator_type}'."
        )
    compiler = compiler_class()
    if default_model and hasattr(compiler, "model"):
        compiler.model = default_model

    compilation_context: dict[str, Any] | None = None
    if evaluator_type == "fsm":
        compilation_context = {
            "simple_config": {
                "name": evaluator_name,
                "steps": normalized_rules,
                "rules": normalized_rules,
            }
        }

    source_text = "\n".join(normalized_rules)
    result = await compiler.compile(source_text, context=compilation_context)
    if not result.success:
        raise ValueError(
            f"Failed to compile rules for evaluator '{evaluator_name}': "
            + "; ".join(result.errors)
        )

    if evaluator_type == "fsm":
        cleaned["workflow"] = result.config
        return cleaned

    if evaluator_type == "llm":
        cleaned["workflow"] = result.config
        return cleaned

    if evaluator_type == "nemo":
        runtime_dir = base_dir / ".openbias_runtime" / "nemo" / evaluator_name
        compiler.export(result, runtime_dir)
        cleaned["config_path"] = str(runtime_dir)
        return cleaned

    if isinstance(result.config, dict):
        cleaned.update(result.config)
        return cleaned

    return cleaned
