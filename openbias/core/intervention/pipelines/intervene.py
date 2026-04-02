"""Intervene action pipeline."""

from __future__ import annotations

import logging
from typing import Any

from openbias.core.interceptor.types import InterceptionResult

from .base import PostCallPipelineContext, PreCallPipelineContext

logger = logging.getLogger(__name__)


class IntervenePipeline:
    """Applies request/response intervention behavior."""

    def handle_pre_call(self, context: PreCallPipelineContext) -> InterceptionResult | None:
        if not context.message:
            return None

        logger.info(
            "Applying pre-call intervention from evaluator '%s'",
            context.evaluator_name,
        )
        applied = context.apply_intervention(
            context.modified_data, context.message, context.default_strategy
        )
        if applied is not None:
            context.modified_data = applied
            context.has_modifications = True
        return None

    def handle_post_call(self, context: PostCallPipelineContext) -> InterceptionResult | None:
        logger.info(
            "Sync POST_CALL evaluator '%s' returned INTERVENE: %s",
            context.evaluator_name,
            context.message,
        )
        context.all_metadata.setdefault("interventions", []).append(
            {
                "evaluator": context.evaluator_name,
                "message": context.message,
            }
        )
        if context.modified_data is None:
            context.modified_data = {}
        context.modified_data.setdefault("_interventions", []).append(
            {
                "evaluator": context.evaluator_name,
                "message": context.message,
                "metadata": context.mapped_metadata,
            }
        )
        return None
