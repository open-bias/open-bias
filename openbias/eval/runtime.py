"""Helpers for deriving eval runtime behavior from active settings."""

from __future__ import annotations

from openbias.config.settings import Settings
from openbias.eval.runner import EvalRuntimeConfig


def runtime_config_from_settings(settings: Settings) -> EvalRuntimeConfig:
    """Build eval runtime behavior from active settings."""

    evaluators = getattr(settings, "evaluators", [])
    if isinstance(evaluators, list) and evaluators:
        phase = evaluators[0].phase
        request_phase_enabled = phase == "pre_call"
        response_phase_enabled = phase == "post_call"
    else:
        request_phase_enabled = True
        response_phase_enabled = True

    return EvalRuntimeConfig(
        request_phase_enabled=request_phase_enabled,
        response_phase_enabled=response_phase_enabled,
        mode=settings.mode,
        fail_action=settings.fail_action,
        strategy=settings.strategy,
    )
