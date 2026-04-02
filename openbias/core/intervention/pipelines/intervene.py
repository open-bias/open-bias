"""Intervene action pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import PostCallPipelineContext, PreCallPipelineContext

if TYPE_CHECKING:
    from openbias.core.interceptor.types import InterceptionResult

logger = logging.getLogger(__name__)


class IntervenePipeline:
    """Applies request/response intervention behavior."""

    def handle_pre_call(self, context: PreCallPipelineContext) -> InterceptionResult | None:
        if not context.message:
            return None

        logger.info(
            "Collecting pre-call intervention signal from evaluator '%s'",
            context.evaluator_name,
        )
        context.all_metadata.setdefault("_pending_pre_call_interventions", []).append(
            {
                "evaluator": context.evaluator_name,
                "message": context.message,
                "metadata": context.mapped_metadata,
                "source": context.mapped_metadata.get("_openbias_source", "sync_pre_call"),
            }
        )
        return None

    def handle_post_call(self, context: PostCallPipelineContext) -> InterceptionResult | None:
        logger.info(
            "Sync POST_CALL evaluator '%s' returned INTERVENE: %s",
            context.evaluator_name,
            context.message,
        )
        context.all_metadata.setdefault("_pending_post_call_interventions", []).append(
            {
                "evaluator": context.evaluator_name,
                "message": context.message,
                "metadata": context.mapped_metadata,
            }
        )
        return None
