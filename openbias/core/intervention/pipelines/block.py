"""Block action pipeline."""

from __future__ import annotations

import logging

from openbias.core.interceptor.types import InterceptionResult

from .base import PostCallPipelineContext, PreCallPipelineContext

logger = logging.getLogger(__name__)


class BlockPipeline:
    """Terminates the request/response when policy decides block."""

    def handle_pre_call(self, context: PreCallPipelineContext) -> InterceptionResult | None:
        logger.warning(
            "Request blocked by evaluator '%s': %s",
            context.evaluator_name,
            context.message,
        )
        return InterceptionResult(
            allowed=False,
            user_message=context.message,
            internal_metadata=context.all_metadata,
        )

    def handle_post_call(self, context: PostCallPipelineContext) -> InterceptionResult | None:
        logger.warning(
            "Response blocked by evaluator '%s': %s",
            context.evaluator_name,
            context.message,
        )
        return InterceptionResult(
            allowed=False,
            user_message=context.message,
            internal_metadata=context.all_metadata,
        )
