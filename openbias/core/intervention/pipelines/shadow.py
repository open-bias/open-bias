"""Shadow action pipeline: observe-only, never mutates payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import PostCallPipelineContext, PreCallPipelineContext

if TYPE_CHECKING:
    from openbias.core.interceptor.types import InterceptionResult


class ShadowPipeline:
    """No-op handler used when violations are downgraded to shadow mode."""

    def handle_pre_call(self, context: PreCallPipelineContext) -> InterceptionResult | None:
        return None

    def handle_post_call(self, context: PostCallPipelineContext) -> InterceptionResult | None:
        return None
