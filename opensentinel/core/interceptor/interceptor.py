"""
Interceptor orchestrator.

Manages the execution of checkers at PRE_CALL and POST_CALL phases,
handles async checker task management, and applies interventions.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from opensentinel.core.intervention.strategies import (
    SystemPromptAppendStrategy,
    UserMessageInjectStrategy,
)
from opensentinel.policy.protocols import Decision, EngineResult

from .adapters import PolicyEngineChecker
from .types import (
    CheckerMode,
    CheckPhase,
    InterceptionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class _PendingResult:
    """Internal: pairs an EngineResult with the checker name that produced it."""

    checker_name: str
    result: EngineResult


class Interceptor:
    """
    Orchestrator for running checkers during LLM request lifecycle.

    Manages:
    - Sync checkers that block and must complete (ALLOW or BLOCK only)
    - Async checkers that run in background with results applied next request
    - Pending async results per session
    - Modification merging for deferred INTERVENE decisions
    """

    def __init__(
        self,
        checkers: list[PolicyEngineChecker],
        default_strategy: str = "system_prompt_append",
    ):
        self._sync_pre_call: list[PolicyEngineChecker] = []
        self._sync_post_call: list[PolicyEngineChecker] = []
        self._async_pre_call: list[PolicyEngineChecker] = []
        self._async_post_call: list[PolicyEngineChecker] = []

        for checker in checkers:
            if checker.mode == CheckerMode.ASYNC:
                if checker.phase == CheckPhase.PRE_CALL:
                    self._async_pre_call.append(checker)
                else:
                    self._async_post_call.append(checker)
            elif checker.phase == CheckPhase.PRE_CALL:
                self._sync_pre_call.append(checker)
            else:
                self._sync_post_call.append(checker)

        # Intervention strategy
        self._default_strategy = default_strategy

        # session_id -> pending async results
        self._pending_async: dict[str, list[_PendingResult]] = {}

        # session_id -> running async tasks
        self._running_tasks: dict[str, list[asyncio.Task[_PendingResult]]] = {}

        logger.info(
            f"Interceptor initialized: {len(self._sync_pre_call)} sync pre-call, "
            f"{len(self._sync_post_call)} sync post-call, "
            f"{len(self._async_pre_call)} async pre-call, "
            f"{len(self._async_post_call)} async post-call"
        )

    async def run_pre_call(
        self,
        session_id: str,
        request_data: dict[str, Any],
        user_request_id: str = "",
    ) -> InterceptionResult:
        """
        Run PRE_CALL phase.

        1. Apply pending async results from previous request
           - BLOCK -> block this request
           - INTERVENE -> merge modified_data into request
        2. Run sync PRE_CALL checkers (gate only: ALLOW or BLOCK)
        3. Start async PRE_CALL checkers in background
        4. Return result with possibly modified request_data
        """
        modified_data = dict(request_data)
        all_metadata: dict[str, Any] = {"results": []}

        # Step 1: Apply pending async results
        pending_results = self._collect_completed_async(session_id)
        for pending in pending_results:
            result = pending.result
            all_metadata["results"].append(
                {"checker": pending.checker_name, "decision": result.decision.value}
            )

            if result.decision == Decision.BLOCK:
                logger.warning(
                    f"Request blocked by async checker '{pending.checker_name}': "
                    f"{result.message}"
                )
                return InterceptionResult(
                    allowed=False,
                    message=result.message,
                    metadata=all_metadata,
                )

            if result.decision == Decision.INTERVENE and result.message:
                logger.info(
                    f"Applying async intervention from '{pending.checker_name}'"
                )
                modified_data = self._apply_intervention(modified_data, result.message)

        # Step 2: Run sync PRE_CALL checkers
        for checker in self._sync_pre_call:
            try:
                result = await checker.evaluate(
                    session_id=session_id,
                    request_data=modified_data,
                    context={"user_request_id": user_request_id},
                )
                all_metadata["results"].append(
                    {"checker": checker.name, "decision": result.decision.value}
                )

                if result.decision == Decision.BLOCK:
                    logger.warning(
                        f"Request blocked by sync checker '{checker.name}': "
                        f"{result.message}"
                    )
                    return InterceptionResult(
                        allowed=False,
                        message=result.message,
                        metadata=all_metadata,
                    )

                if result.decision == Decision.INTERVENE and result.message:
                    logger.info(
                        f"Applying sync intervention from '{checker.name}'"
                    )
                    modified_data = self._apply_intervention(
                        modified_data, result.message
                    )

            except Exception as e:
                logger.error(f"Checker '{checker.name}' failed: {e}")
                # Fail-open: log and continue instead of blocking
                all_metadata["results"].append(
                    {"checker": checker.name, "decision": "error", "error": str(e)}
                )

        # Step 3: Start async PRE_CALL checkers in background
        for checker in self._async_pre_call:
            self._start_async_checker(checker, session_id, modified_data, None)

        return InterceptionResult(
            allowed=True,
            modified_data=modified_data if modified_data != request_data else None,
            metadata=all_metadata,
        )

    async def run_post_call(
        self,
        session_id: str,
        request_data: dict[str, Any],
        response_data: Any,
        user_request_id: str = "",
    ) -> InterceptionResult:
        """
        Run POST_CALL phase.

        1. Run sync POST_CALL checkers (gate only: ALLOW or BLOCK)
        2. Start async POST_CALL checkers in background (don't wait)
        3. Return result
        """
        all_metadata: dict[str, Any] = {"results": []}

        # Step 1: Run sync POST_CALL checkers
        for checker in self._sync_post_call:
            try:
                result = await checker.evaluate(
                    session_id=session_id,
                    request_data=request_data,
                    response_data=response_data,
                    context={"user_request_id": user_request_id},
                )
                all_metadata["results"].append(
                    {"checker": checker.name, "decision": result.decision.value}
                )

                if result.decision == Decision.BLOCK:
                    logger.warning(
                        f"Response blocked by sync checker '{checker.name}': "
                        f"{result.message}"
                    )
                    return InterceptionResult(
                        allowed=False,
                        message=result.message,
                        metadata=all_metadata,
                    )

            except Exception as e:
                logger.error(f"Checker '{checker.name}' failed: {e}")
                all_metadata["results"].append(
                    {"checker": checker.name, "decision": "error", "error": str(e)}
                )

        # Step 2: Start async POST_CALL checkers in background
        for checker in self._async_post_call:
            self._start_async_checker(
                checker, session_id, request_data, response_data
            )

        return InterceptionResult(
            allowed=True,
            metadata=all_metadata,
        )

    def _collect_completed_async(self, session_id: str) -> list[_PendingResult]:
        """Collect results from completed async tasks for a session."""
        results: list[_PendingResult] = []

        # Get pending results stored from previous collection
        if session_id in self._pending_async:
            results.extend(self._pending_async.pop(session_id))

        # Check running tasks
        if session_id in self._running_tasks:
            tasks = self._running_tasks[session_id]
            still_running: list[asyncio.Task[_PendingResult]] = []

            for task in tasks:
                if task.done():
                    try:
                        result = task.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Async checker task failed: {e}")
                        results.append(
                            _PendingResult(
                                checker_name="async_task_error",
                                result=EngineResult(
                                    decision=Decision.ALLOW,
                                    message=f"Async task error: {e}",
                                    metadata={"error": str(e)},
                                ),
                            )
                        )
                else:
                    still_running.append(task)

            if still_running:
                self._running_tasks[session_id] = still_running
            else:
                del self._running_tasks[session_id]

        return results

    def _start_async_checker(
        self,
        checker: PolicyEngineChecker,
        session_id: str,
        request_data: dict[str, Any],
        response_data: Any,
    ) -> None:
        """Start an async checker task in the background."""

        async def run_checker() -> _PendingResult:
            try:
                result = await checker.evaluate(
                    session_id=session_id,
                    request_data=request_data,
                    response_data=response_data,
                )
                return _PendingResult(checker_name=checker.name, result=result)
            except Exception as e:
                logger.error(f"Async checker '{checker.name}' failed: {e}")
                return _PendingResult(
                    checker_name=checker.name,
                    result=EngineResult(
                        decision=Decision.ALLOW,
                        message=f"Async checker error: {e}",
                        metadata={"error": str(e)},
                    ),
                )

        task = asyncio.create_task(run_checker())

        if session_id not in self._running_tasks:
            self._running_tasks[session_id] = []
        self._running_tasks[session_id].append(task)

        logger.debug(f"Started async checker '{checker.name}' for session {session_id}")

    def _apply_intervention(
        self,
        request_data: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        """
        Apply an intervention message to request data using the configured strategy.

        Args:
            request_data: The LLM request data to modify.
            message: The intervention guidance text.

        Returns:
            New request data dict with the intervention applied.
        """
        result = dict(request_data)
        messages = result.get("messages", [])

        if self._default_strategy == "user_message_inject":
            result["messages"] = UserMessageInjectStrategy.merge(messages, message)
        else:
            result["messages"] = SystemPromptAppendStrategy.merge(messages, message)

        return result

    async def cleanup_session(self, session_id: str) -> None:
        """Cancel running async tasks and clear pending results for a session."""
        if session_id in self._running_tasks:
            for task in self._running_tasks[session_id]:
                if not task.done():
                    task.cancel()
            del self._running_tasks[session_id]

        if session_id in self._pending_async:
            del self._pending_async[session_id]

        logger.debug(f"Cleaned up session {session_id}")

    async def shutdown(self) -> None:
        """Shutdown the interceptor, cancelling all running async tasks."""
        for session_id in list(self._running_tasks.keys()):
            await self.cleanup_session(session_id)

        logger.info("Interceptor shutdown complete")
