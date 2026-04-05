"""Helpers for constructing initialized policy engines from project config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openbias.config.settings import Settings
from openbias.policy.compiler.runtime import compile_runtime_config_for_evaluator
from openbias.policy.registry import PolicyEngineRegistry


async def build_engine_for_policy(
    *,
    settings: Settings,
    config_path: Path | None,
    rules_path: Path,
) -> Any:
    """Build and initialize one policy engine against a specific rules file."""

    if not settings.evaluators:
        raise ValueError("Policy execution requires at least one configured evaluator.")

    evaluator = settings.evaluators[0]
    evaluator_config = dict(evaluator.config)
    settings.inject_default_model(
        evaluator.type, evaluator_config, settings.proxy.default_model
    )
    base_dir = (config_path.parent if config_path else Path.cwd()).resolve()
    compiled = await compile_runtime_config_for_evaluator(
        evaluator_name=evaluator.name,
        evaluator_type=evaluator.type,
        evaluator_config=evaluator_config,
        default_model=settings.proxy.default_model,
        base_dir=base_dir,
        rules_path=rules_path,
    )
    return await PolicyEngineRegistry.create_and_initialize(evaluator.type, compiled)
