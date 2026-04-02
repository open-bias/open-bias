"""Block action pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import PostCallPipelineContext, PreCallPipelineContext

if TYPE_CHECKING:
    from openbias.core.interceptor.types import InterceptionResult

logger = logging.getLogger(__name__)


class BlockPipeline:
    """Terminates the request/response when policy decides block."""

    def handle_pre_call(self, context: PreCallPipelineContext) -> InterceptionResult | None:
        from openbias.core.interceptor.types import InterceptionResult

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
        from openbias.core.interceptor.types import InterceptionResult

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
