"""Shadow action pipeline: observe-only, never mutates payloads."""

from __future__ import annotations

from openbias.core.interceptor.types import InterceptionResult

from .base import PostCallPipelineContext, PreCallPipelineContext


class ShadowPipeline:
    """No-op handler used when violations are downgraded to shadow mode."""

    def handle_pre_call(self, context: PreCallPipelineContext) -> InterceptionResult | None:
        return None

    def handle_post_call(self, context: PostCallPipelineContext) -> InterceptionResult | None:
        return None
