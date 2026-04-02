"""Runtime rules-to-engine compilation used by `openbias serve`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openbias.policy.compiler import PolicyCompilerRegistry
from openbias.policy.registry import PolicyEngineRegistry
from openbias.policy.rules import resolve_rules_payload


async def compile_runtime_config_for_evaluator(
    evaluator_name: str,
    evaluator_type: str,
    evaluator_config: dict[str, Any],
    *,
    default_model: str | None,
    base_dir: Path,
) -> dict[str, Any]:
    """Compile canonical rules inputs into engine-native runtime config."""
    normalized_rules = resolve_rules_payload(
        evaluator_config,
        base_dir=base_dir,
        auto_discover_rules_md=True,
    )
    if not normalized_rules:
        return evaluator_config

    engine_cls = PolicyEngineRegistry.get(evaluator_type)
    compiler = None
    if engine_cls is not None:
        compiler = engine_cls().get_compiler(model=default_model)
    if compiler is None:
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

    cleaned = dict(evaluator_config)
    cleaned.pop("rules", None)
    cleaned.pop("rules_file", None)

    if evaluator_type == "judge":
        compiled_rubrics = []
        if isinstance(result.config, dict):
            compiled_rubrics = result.config.get("rubrics", [])
        cleaned["inline_rules"] = compiled_rubrics or normalized_rules
        return cleaned

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

    return cleaned
