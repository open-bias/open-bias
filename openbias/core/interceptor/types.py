"""Core result types shared between interceptor, pipelines, and hooks."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(init=False)
class InterceptionResult:
    """Normalized action result consumed by proxy hooks.

    Canonical fields:
    - ``user_message``: optional user-facing message when blocked
    - ``internal_metadata``: internal diagnostics and evaluator details

    Backward-compat aliases are supported for existing call sites:
    - ``message`` -> ``user_message``
    - ``metadata`` -> ``internal_metadata``
    """

    allowed: bool
    modified_data: dict[str, Any] | None = None
    user_message: str | None = None
    internal_metadata: dict[str, Any] = field(default_factory=dict)
    pending_intervention: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        allowed: bool,
        modified_data: dict[str, Any] | None = None,
        user_message: str | None = None,
        internal_metadata: dict[str, Any] | None = None,
        pending_intervention: dict[str, Any] | None = None,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.allowed = allowed
        self.modified_data = modified_data
        self.user_message = user_message if user_message is not None else message
        self.internal_metadata = (
            dict(internal_metadata) if internal_metadata is not None else dict(metadata or {})
        )
        self.pending_intervention = pending_intervention

    @property
    def message(self) -> str | None:
        return self.user_message

    @message.setter
    def message(self, value: str | None) -> None:
        self.user_message = value

    @property
    def metadata(self) -> dict[str, Any]:
        return self.internal_metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        self.internal_metadata = value
