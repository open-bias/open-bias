"""
Composite policy engine implementation.

Combines multiple policy engines to run in parallel,
merging their results with configurable strategies.

This enables using multiple policy mechanisms together,
e.g., FSM workflow enforcement + NeMo content moderation.
"""

from typing import Optional, Dict, Any, List
import logging
import asyncio

from opensentinel.policy.protocols import (
    PolicyEngine,
    Decision,
    EngineResult,
    require_initialized,
)
from opensentinel.policy.registry import register_engine, PolicyEngineRegistry

logger = logging.getLogger(__name__)

# Decision priority for merging (higher = more restrictive)
DECISION_PRIORITY = {
    Decision.BLOCK: 3,
    Decision.INTERVENE: 2,
    Decision.ALLOW: 1,
}


@register_engine("composite")
class CompositePolicyEngine(PolicyEngine):
    """
    Combines multiple policy engines.

    Evaluation strategy:
    - All engines evaluate in parallel
    - Most restrictive decision wins: BLOCK > INTERVENE > ALLOW
    - First INTERVENE message wins
    - All metadata merged under engine names

    Configuration:
        - engines: list - List of engine configurations
          Each entry: {"type": "fsm|nemo|...", "config": {...}}
        - strategy: str - Merge strategy: "all" (run all) or "first_block" (stop on first block)
        - parallel: bool - Run engines in parallel (default: True)

    Example:
        ```python
        engine = CompositePolicyEngine()
        await engine.initialize({
            "engines": [
                {"type": "fsm", "config": {"workflow_path": "./workflow.yaml"}},
                {"type": "nemo", "config": {"config_path": "./nemo_config/"}}
            ],
            "strategy": "all"
        })
        ```
    """

    def __init__(self):
        self._engines: List[PolicyEngine] = []
        self._engine_configs: List[Dict[str, Any]] = []
        self._strategy = "all"
        self._parallel = True
        self._initialized = False

    @property
    def name(self) -> str:
        """Unique name showing all combined engines."""
        if not self._engines:
            return "composite:empty"
        names = [e.name for e in self._engines]
        return f"composite:[{','.join(names)}]"

    @property
    def engine_type(self) -> str:
        """Type identifier for this engine."""
        return "composite"

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize with list of engine configurations.

        Args:
            config: Configuration dict with:
                - engines: List of {"type": str, "config": dict}
                - strategy: "all" or "first_block" (optional)
                - parallel: bool (optional, default True)

        Raises:
            ValueError: If engines list is empty or invalid
        """
        engine_configs = config.get("engines", [])
        if not engine_configs:
            raise ValueError("Composite engine requires at least one engine in 'engines' list")

        self._strategy = config.get("strategy", "all")
        self._parallel = config.get("parallel", True)
        self._engine_configs = engine_configs

        # Create and initialize all engines
        for engine_config in engine_configs:
            engine_type = engine_config.get("type")
            if not engine_type:
                raise ValueError("Each engine config must have a 'type' field")

            engine = PolicyEngineRegistry.create(engine_type)
            await engine.initialize(engine_config.get("config", {}))
            self._engines.append(engine)

            logger.debug(f"Composite: Added engine '{engine.name}'")

        self._initialized = True
        logger.info(
            f"CompositePolicyEngine initialized with {len(self._engines)} engines: "
            f"{[e.engine_type for e in self._engines]}"
        )

    @require_initialized
    async def evaluate_request(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EngineResult:
        """
        Evaluate request through all engines.

        Returns the most restrictive decision from all engines.
        """
        results = await self._run_engines(
            "request", session_id, request_data=request_data, context=context
        )
        return self._merge_results(results)

    @require_initialized
    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EngineResult:
        """
        Evaluate response through all engines.

        Returns the most restrictive decision from all engines.
        """
        results = await self._run_engines(
            "response", session_id,
            response_data=response_data, request_data=request_data, context=context
        )
        return self._merge_results(results)

    async def _run_engines(
        self,
        phase: str,
        session_id: str,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Any = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[EngineResult]:
        """Run all engines for the given phase, handling errors as fail-open."""

        async def run_one(engine: PolicyEngine) -> EngineResult:
            try:
                if phase == "request":
                    return await engine.evaluate_request(session_id, request_data, context)
                else:
                    return await engine.evaluate_response(
                        session_id, response_data, request_data, context
                    )
            except Exception as e:
                logger.error(f"Engine {engine.name} {phase} evaluation failed: {e}")
                return EngineResult(
                    decision=Decision.ALLOW,
                    metadata={"error": str(e), "engine": engine.name},
                )

        if self._parallel:
            results = await asyncio.gather(*[run_one(e) for e in self._engines])
            valid_results = list(results)
        else:
            valid_results = []
            for engine in self._engines:
                result = await run_one(engine)
                valid_results.append(result)
                if self._strategy == "first_block" and result.decision == Decision.BLOCK:
                    break

        return valid_results

    def _merge_results(self, results: List[EngineResult]) -> EngineResult:
        """
        Merge results from multiple engines.

        Priority: BLOCK > INTERVENE > ALLOW
        First INTERVENE message wins; first BLOCK message wins.
        All metadata merged under engine names.
        """
        if not results:
            return EngineResult(decision=Decision.ALLOW)

        final_decision = Decision.ALLOW
        intervene_message: Optional[str] = None
        block_message: Optional[str] = None
        metadata: Dict[str, Any] = {"engines": {}}

        for i, result in enumerate(results):
            engine_name = self._engines[i].name if i < len(self._engines) else f"engine_{i}"

            # Most restrictive decision wins
            if DECISION_PRIORITY[result.decision] > DECISION_PRIORITY[final_decision]:
                final_decision = result.decision

            # First INTERVENE message wins
            if result.decision == Decision.INTERVENE and intervene_message is None:
                intervene_message = result.message

            # First BLOCK message wins
            if result.decision == Decision.BLOCK and block_message is None:
                block_message = result.message

            # Merge metadata under engine name
            if result.metadata:
                metadata["engines"][engine_name] = result.metadata

        # Use block message if blocked, otherwise intervene message
        final_message = block_message if final_decision == Decision.BLOCK else intervene_message

        return EngineResult(
            decision=final_decision,
            message=final_message,
            metadata=metadata,
        )

    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session state from all engines."""
        if not self._initialized:
            return None

        states: Dict[str, Any] = {}
        for engine in self._engines:
            state = await engine.get_session_state(session_id)
            if state:
                states[engine.name] = state

        return states if states else None

    async def reset_session(self, session_id: str) -> None:
        """Reset session state in all engines."""
        if not self._initialized:
            return

        await asyncio.gather(*[
            engine.reset_session(session_id)
            for engine in self._engines
        ])
        logger.debug(f"Composite: Reset session {session_id} in all engines")

    async def shutdown(self) -> None:
        """Cleanup all engines."""
        await asyncio.gather(*[
            engine.shutdown()
            for engine in self._engines
        ])
        self._engines.clear()
        self._initialized = False
        logger.info("CompositePolicyEngine shutdown")

    def get_engines(self) -> List[PolicyEngine]:
        """Get list of child engines (for debugging)."""
        return list(self._engines)

    def get_engine_by_type(self, engine_type: str) -> Optional[PolicyEngine]:
        """Get a specific engine by type."""
        for engine in self._engines:
            if engine.engine_type == engine_type:
                return engine
        return None
