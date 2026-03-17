"""
Adapters to wrap PolicyEngine instances for use by the Interceptor.

PolicyEngineChecker delegates to the appropriate engine method based on phase.
"""

import logging
from typing import Any

from opensentinel.policy.protocols import EngineResult, PolicyEngine

from .types import CheckerMode, CheckPhase

logger = logging.getLogger(__name__)


class PolicyEngineChecker:
    """
    Wraps a PolicyEngine for use by the Interceptor.

    Routes to evaluate_request() for PRE_CALL or evaluate_response() for POST_CALL.
    Returns EngineResult directly — no mapping layer.
    """

    def __init__(
        self,
        engine: PolicyEngine,
        phase: CheckPhase,
        mode: CheckerMode = CheckerMode.SYNC,
    ):
        self.engine = engine
        self.phase = phase
        self.mode = mode

    @property
    def name(self) -> str:
        return f"{self.engine.name}_{self.phase.value}"

    async def evaluate(
        self,
        session_id: str,
        request_data: dict[str, Any],
        response_data: Any = None,
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        """Evaluate by delegating to the engine's phase-appropriate method."""
        if self.phase == CheckPhase.PRE_CALL:
            return await self.engine.evaluate_request(session_id, request_data, context)
        return await self.engine.evaluate_response(
            session_id, response_data, request_data, context
        )
