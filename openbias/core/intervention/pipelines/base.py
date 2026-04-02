"""Shared contracts for interceptor action pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from openbias.core.interceptor.types import InterceptionResult


@dataclass
class PreCallPipelineContext:
    session_id: str
    evaluator_name: str
    message: str | None
    mapped_metadata: dict[str, Any]
    modified_data: dict[str, Any]
    all_metadata: dict[str, Any]
    default_strategy: str
    apply_intervention: Any
    has_modifications: bool = False


@dataclass
class PostCallPipelineContext:
    session_id: str
    evaluator_name: str
    message: str | None
    mapped_metadata: dict[str, Any]
    modified_data: dict[str, Any] | None
    all_metadata: dict[str, Any]


class ActionPipeline(Protocol):
    """Minimal contract for action-specific handling."""

    def handle_pre_call(self, context: PreCallPipelineContext) -> InterceptionResult | None:
        ...

    def handle_post_call(self, context: PostCallPipelineContext) -> InterceptionResult | None:
        ...
