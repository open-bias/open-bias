"""
Interceptor orchestrator.

Manages the execution of checkers at PRE_CALL and POST_CALL phases,
handles async checker task management, and applies interventions.
"""

import asyncio
import copy
import logging
from dataclasses import dataclass
from typing import Any, Literal

from opensentinel.core.intervention.strategies import (
    StrategyType,
    SystemPromptAppendStrategy,
    UserMessageInjectStrategy,
)
from opensentinel.core.session import SessionStore
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

    # Defaults for session memory management
    DEFAULT_SESSION_TTL = 3600  # 1 hour
    DEFAULT_MAX_SESSIONS = 10_000
    DEFAULT_MAX_ASYNC_TASKS_PER_SESSION = 50

    def __init__(
        self,
        checkers: list[PolicyEngineChecker],
        default_strategy: str = "user_message_inject",
        fail_action: Literal["intervene", "block"] = "intervene",
        session_ttl: int | None = None,
        max_sessions: int | None = None,
        max_async_tasks_per_session: int | None = None,
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
        valid_strategies = {s.value for s in StrategyType}
        if default_strategy not in valid_strategies:
            raise ValueError(
                f"Unknown default_strategy '{default_strategy}'. "
                f"Valid values: {sorted(valid_strategies)}"
            )
        self._default_strategy = default_strategy
        self._fail_action = fail_action

        # Session memory management
        self._max_async_tasks = (
            max_async_tasks_per_session
            if max_async_tasks_per_session is not None
            else self.DEFAULT_MAX_ASYNC_TASKS_PER_SESSION
        )

        # Temporary storage for collected-but-not-yet-confirmed async results
        self._last_collected: dict[str, list[asyncio.Task[_PendingResult]]] = {}

        self._sessions: SessionStore[list[asyncio.Task[_PendingResult]]] = SessionStore(
            ttl=session_ttl if session_ttl is not None else self.DEFAULT_SESSION_TTL,
            max_sessions=max_sessions if max_sessions is not None else self.DEFAULT_MAX_SESSIONS,
            on_evict=self._on_session_evict,
        )

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
        self._sessions.touch(session_id)
        self._sessions.evict_stale()

        modified_data = copy.deepcopy(request_data)
        all_metadata: dict[str, Any] = {"results": []}

        # Step 1: Apply pending async results
        pending_results = self._collect_completed_async(session_id)
        for processed_count, pending in enumerate(pending_results, start=1):
            result = pending.result
            decision = self._effective_decision(result.decision)
            all_metadata["results"].append(
                {"checker": pending.checker_name, "decision": decision.value}
            )

            if decision == Decision.BLOCK:
                logger.warning(
                    f"Request blocked by async checker '{pending.checker_name}': "
                    f"{result.message}"
                )
                # Only confirm tasks up to and including the blocking one.
                # Any remaining completed-but-unprocessed tasks stay in the
                # session so they are picked up on the next request.
                self._confirm_collected(session_id, count=processed_count)
                return InterceptionResult(
                    allowed=False,
                    message=result.message,
                    metadata=all_metadata,
                )

            if decision == Decision.INTERVENE:
                if result.modified_messages is not None:
                    logger.info(
                        f"Applying async message replacement from '{pending.checker_name}'"
                    )
                    modified_data = dict(modified_data)
                    modified_data["messages"] = result.modified_messages
                elif result.message:
                    logger.info(
                        f"Applying async intervention from '{pending.checker_name}'"
                    )
                    modified_data = self._apply_intervention(
                        modified_data, result.message, self._default_strategy
                    )

        # Async results processed successfully — remove from session store
        self._confirm_collected(session_id)

        # Step 2: Run sync PRE_CALL checkers
        # Note: INTERVENE results accumulate — each checker sees the already-modified
        # request_data from prior checkers. Order in checkers list matters.
        for checker in self._sync_pre_call:
            try:
                result = await checker.evaluate(
                    session_id=session_id,
                    request_data=modified_data,
                    context={"user_request_id": user_request_id},
                )
                decision = self._effective_decision(result.decision)
                all_metadata["results"].append(
                    {"checker": checker.name, "decision": decision.value}
                )

                if decision == Decision.BLOCK:
                    logger.warning(
                        f"Request blocked by sync checker '{checker.name}': "
                        f"{result.message}"
                    )
                    return InterceptionResult(
                        allowed=False,
                        message=result.message,
                        metadata=all_metadata,
                    )

                if decision == Decision.INTERVENE:
                    if result.modified_messages is not None:
                        logger.info(
                            f"Applying sync message replacement from '{checker.name}'"
                        )
                        modified_data = dict(modified_data)
                        modified_data["messages"] = result.modified_messages
                    elif result.message:
                        logger.info(
                            f"Applying sync intervention from '{checker.name}'"
                        )
                        modified_data = self._apply_intervention(
                            modified_data, result.message, self._default_strategy
                        )

            except Exception as e:
                logger.error(f"Checker '{checker.name}' failed: {e}")
                # Fail-open: log and continue instead of blocking
                all_metadata["results"].append(
                    {"checker": checker.name, "decision": "error", "error": str(e)}
                )

        # Step 3: Start async PRE_CALL checkers in background
        for checker in self._async_pre_call:
            self._start_async_checker(
                checker, session_id, modified_data, None,
                context={"user_request_id": user_request_id},
            )

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

        1. Run sync POST_CALL checkers (ALLOW, BLOCK, or INTERVENE)
        2. Start async POST_CALL checkers in background (don't wait)
        3. Return result with optional modified_data for response modification
        """
        self._sessions.touch(session_id)
        self._sessions.evict_stale()

        all_metadata: dict[str, Any] = {"results": [], "interventions": []}
        modified_data: dict[str, Any] | None = None

        # Step 1: Run sync POST_CALL checkers
        for checker in self._sync_post_call:
            try:
                result = await checker.evaluate(
                    session_id=session_id,
                    request_data=request_data,
                    response_data=response_data,
                    context={"user_request_id": user_request_id},
                )
                decision = self._effective_decision(result.decision)
                all_metadata["results"].append(
                    {"checker": checker.name, "decision": decision.value}
                )

                if decision == Decision.BLOCK:
                    logger.warning(
                        f"Response blocked by sync checker '{checker.name}': "
                        f"{result.message}"
                    )
                    return InterceptionResult(
                        allowed=False,
                        message=result.message,
                        metadata=all_metadata,
                    )

                if decision == Decision.INTERVENE:
                    logger.info(
                        f"Sync POST_CALL checker '{checker.name}' returned INTERVENE: "
                        f"{result.message}"
                    )
                    intervention_info: dict[str, Any] = {
                        "checker": checker.name,
                        "message": result.message,
                    }
                    if result.modified_messages is not None:
                        intervention_info["has_modified_messages"] = True
                    all_metadata["interventions"].append(intervention_info)
                    # Store the engine result for the hooks layer to act on
                    if modified_data is None:
                        modified_data = {}
                    modified_data.setdefault("_interventions", []).append(
                        {
                            "checker": checker.name,
                            "message": result.message,
                            "modified_messages": result.modified_messages,
                            "metadata": result.metadata,
                        }
                    )

            except Exception as e:
                logger.error(f"Checker '{checker.name}' failed: {e}")
                all_metadata["results"].append(
                    {"checker": checker.name, "decision": "error", "error": str(e)}
                )

        # Step 2: Start async POST_CALL checkers in background
        for checker in self._async_post_call:
            self._start_async_checker(
                checker, session_id, request_data, response_data,
                context={"user_request_id": user_request_id},
            )

        return InterceptionResult(
            allowed=True,
            modified_data=modified_data,
            metadata=all_metadata,
        )

    def _effective_decision(self, decision: Decision) -> Decision:
        """Upgrade INTERVENE → BLOCK when fail_action is 'block'."""
        if decision == Decision.INTERVENE and self._fail_action == "block":
            return Decision.BLOCK
        return decision

    def _collect_completed_async(self, session_id: str) -> list[_PendingResult]:
        """Collect results from completed async tasks for a session.

        Non-destructive: completed tasks are only removed after results are
        extracted. Call _confirm_collected() after processing to clean up.
        """
        results: list[_PendingResult] = []

        tasks = self._sessions.get(session_id)
        if tasks is None:
            return results

        completed_tasks: list[asyncio.Task[_PendingResult]] = []

        for task in tasks:
            if task.done():
                completed_tasks.append(task)
                try:
                    result = task.result()
                    results.append(result)
                except asyncio.CancelledError:
                    logger.warning("Async checker task was cancelled")
                    results.append(
                        _PendingResult(
                            checker_name="async_task_cancelled",
                            result=EngineResult(
                                decision=Decision.ALLOW,
                                message="Async task cancelled",
                                metadata={},
                            ),
                        )
                    )
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

        # Store completed tasks for later removal by _confirm_collected
        self._last_collected[session_id] = completed_tasks

        return results

    def _confirm_collected(self, session_id: str, count: int | None = None) -> None:
        """Remove previously collected async tasks after successful processing.

        Args:
            session_id: Session to confirm for.
            count: Number of tasks to confirm from the front of the collected list.
                   If None, confirm all collected tasks. Use a specific count when
                   processing was interrupted early (e.g., BLOCK) so that unprocessed
                   completed tasks remain in the session for the next request.
        """
        completed_tasks = self._last_collected.get(session_id)
        if completed_tasks is None:
            return

        tasks_to_remove = completed_tasks if count is None else completed_tasks[:count]
        remaining_collected = [] if count is None else completed_tasks[count:]

        if remaining_collected:
            # Keep unprocessed completed tasks for next request
            self._last_collected[session_id] = remaining_collected
        else:
            self._last_collected.pop(session_id, None)

        tasks = self._sessions.get(session_id)
        if tasks is None:
            return

        remove_set = set(id(t) for t in tasks_to_remove)
        still_pending = [t for t in tasks if id(t) not in remove_set]

        if still_pending:
            tasks[:] = still_pending
        else:
            self._sessions.remove(session_id)

    def _start_async_checker(
        self,
        checker: PolicyEngineChecker,
        session_id: str,
        request_data: dict[str, Any],
        response_data: Any,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Start an async checker task in the background."""

        async def run_checker() -> _PendingResult:
            try:
                result = await checker.evaluate(
                    session_id=session_id,
                    request_data=request_data,
                    response_data=response_data,
                    context=context,
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

        # Enforce per-session async task cap
        tasks = self._sessions.get(session_id)
        if tasks is None:
            tasks = []
            self._sessions.put(session_id, tasks)

        # Prune completed tasks before checking the cap
        tasks[:] = [t for t in tasks if not t.done()]

        if len(tasks) >= self._max_async_tasks:
            logger.warning(
                f"Async task cap ({self._max_async_tasks}) reached for session "
                f"{session_id}, dropping oldest task"
            )
            oldest = tasks.pop(0)
            if not oldest.done():
                oldest.cancel()

        task = asyncio.create_task(run_checker())
        tasks.append(task)

        logger.debug(f"Started async checker '{checker.name}' for session {session_id}")

    def _apply_intervention(
        self,
        request_data: dict[str, Any],
        message: str,
        strategy: str | None = None,
    ) -> dict[str, Any]:
        """
        Apply an intervention message to request data using the configured strategy.

        Args:
            request_data: The LLM request data to modify.
            message: The intervention guidance text.
            strategy: Strategy name override; falls back to self._default_strategy.

        Returns:
            New request data dict with the intervention applied.
        """
        result = dict(request_data)
        messages = result.get("messages", [])
        effective_strategy = strategy or self._default_strategy

        if effective_strategy == "user_message_inject":
            result["messages"] = UserMessageInjectStrategy.merge(messages, message)
        elif effective_strategy == "system_prompt_append":
            result["messages"] = SystemPromptAppendStrategy.merge(messages, message)
        elif effective_strategy == "response_modification":
            logger.warning(
                "response_modification is a response-time strategy and cannot be applied "
                "during request modification (PRE_CALL); intervention skipped"
            )

        return result

    @staticmethod
    def _on_session_evict(_session_id: str, tasks: list[asyncio.Task[_PendingResult]]) -> None:
        """Cancel all async tasks when a session is evicted."""
        for task in tasks:
            if not task.done():
                task.cancel()

    async def cleanup_session(self, session_id: str) -> None:
        """Cancel running async tasks and clear pending results for a session."""
        tasks = self._sessions.remove(session_id)
        if tasks:
            for task in tasks:
                if not task.done():
                    task.cancel()

        logger.debug(f"Cleaned up session {session_id}")

    async def shutdown(self) -> None:
        """Shutdown the interceptor, cancelling all running async tasks."""
        for session_id in list(self._sessions.keys()):
            await self.cleanup_session(session_id)

        logger.info("Interceptor shutdown complete")
