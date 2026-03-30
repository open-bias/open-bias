"""
Interceptor orchestrator.

Manages the execution of policy evaluators at PRE_CALL and POST_CALL phases,
handles async evaluator task management, and applies interventions.
"""

import asyncio
import copy
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

from openbias.core.intervention.strategies import (
    StrategyType,
    SystemPromptAppendStrategy,
    UserMessageInjectStrategy,
)
from openbias.core.session import SessionStore
from openbias.policy.protocols import Decision, EngineResult, PolicyEngine

from .types import InterceptionResult

logger = logging.getLogger(__name__)


@dataclass
class _PendingResult:
    """Internal: pairs an EngineResult with the evaluator name that produced it."""

    evaluator_name: str
    result: EngineResult


class Interceptor:
    """
    Orchestrator for running policy evaluators during LLM request lifecycle.

    Manages:
    - Sync evaluators that block and must complete (ALLOW or BLOCK only)
    - Async evaluators that run in background with results applied next request
    - Pending async results per session
    - Modification merging for deferred INTERVENE decisions
    - Per-session intervention count tracking with escalation to BLOCK
    """

    # Defaults for session memory management
    DEFAULT_SESSION_TTL = 3600  # 1 hour
    DEFAULT_MAX_SESSIONS = 10_000

    def __init__(
        self,
        pre_call_evaluators: list[PolicyEngine],
        post_call_evaluators: list[PolicyEngine],
        mode: str = "async",
        default_strategy: str = "user_message_inject",
        fail_action: Literal["intervene", "block", "shadow"] = "intervene",
        max_intervention_attempts: int = 3,
        session_ttl: int | None = None,
        max_sessions: int | None = None,
    ):
        # PRE_CALL is always sync
        self._sync_pre_call_evaluators: list[PolicyEngine] = list(pre_call_evaluators)

        # POST_CALL mode depends on mode param
        self._sync_post_call_evaluators: list[PolicyEngine] = []
        self._async_post_call_evaluators: list[PolicyEngine] = []

        if mode == "async":
            self._async_post_call_evaluators = list(post_call_evaluators)
        else:
            self._sync_post_call_evaluators = list(post_call_evaluators)

        # Intervention strategy
        valid_strategies = {s.value for s in StrategyType}
        if default_strategy not in valid_strategies:
            raise ValueError(
                f"Unknown default_strategy '{default_strategy}'. "
                f"Valid values: {sorted(valid_strategies)}"
            )
        self._default_strategy = default_strategy
        self._fail_action = fail_action
        self._max_intervention_attempts = max_intervention_attempts

        # Per-session intervention count tracking
        self._intervention_counts: dict[str, int] = {}

        # Temporary storage for collected-but-not-yet-confirmed async results
        self._last_collected: dict[str, list[asyncio.Task[_PendingResult]]] = {}

        self._sessions: SessionStore[list[asyncio.Task[_PendingResult]]] = SessionStore(
            ttl=session_ttl if session_ttl is not None else self.DEFAULT_SESSION_TTL,
            max_sessions=max_sessions if max_sessions is not None else self.DEFAULT_MAX_SESSIONS,
            on_evict=self._on_session_evict,
        )

        logger.info(
            f"Interceptor initialized: {len(self._sync_pre_call_evaluators)} sync pre-call, "
            f"{len(self._sync_post_call_evaluators)} sync post-call, "
            f"{len(self._async_post_call_evaluators)} async post-call"
        )

    async def run_pre_call(
        self,
        session_id: str,
        request_data: dict[str, Any],
        user_request_id: str = "",
        span_factory: Any | None = None,
        async_span_group: Any | None = None,
    ) -> InterceptionResult:
        """
        Run PRE_CALL phase.

        1. Apply pending async results from previous request
           - BLOCK -> block this request
           - INTERVENE -> merge modified_data into request
        2. Run sync PRE_CALL evaluators (gate only: ALLOW or BLOCK)
        3. Return result with possibly modified request_data
        """
        self._sessions.touch(session_id)
        self._sessions.evict_stale()

        modified_data = copy.deepcopy(request_data)
        all_metadata: dict[str, Any] = {"results": []}

        # Step 1: Apply pending async results
        pending_results = self._collect_completed_async(session_id)
        for processed_count, pending in enumerate(pending_results, start=1):
            _async_ctx = (
                async_span_group.applied(pending.evaluator_name)
                if async_span_group is not None
                else nullcontext()
            )
            with _async_ctx:
                result = pending.result
                decision = self._effective_decision(result.decision, session_id)
                all_metadata["results"].append(
                    {"evaluator": pending.evaluator_name, "decision": decision.value}
                )

                if decision == Decision.BLOCK:
                    logger.warning(
                        f"Request blocked by async evaluator '{pending.evaluator_name}': "
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
                            f"Applying async message replacement from '{pending.evaluator_name}'"
                        )
                        modified_data = dict(modified_data)
                        modified_data["messages"] = result.modified_messages
                    elif result.message:
                        logger.info(
                            f"Applying async intervention from '{pending.evaluator_name}'"
                        )
                        modified_data = self._apply_intervention(
                            modified_data, result.message, self._default_strategy
                        )

        # Async results processed successfully — remove from session store
        self._confirm_collected(session_id)

        # Step 2: Run sync PRE_CALL evaluators
        # Note: INTERVENE results accumulate — each evaluator sees the already-modified
        # request_data from prior evaluators. Order in evaluators list matters.
        for evaluator in self._sync_pre_call_evaluators:
            _eval_ctx = (
                span_factory(evaluator.name, "pre_call")
                if span_factory is not None
                else nullcontext()
            )
            with _eval_ctx as _eval_span:
                try:
                    ctx: dict[str, Any] = {"user_request_id": user_request_id}
                    if _eval_span is not None:
                        ctx["_parent_span"] = _eval_span
                    result = await evaluator.evaluate_request(
                        session_id=session_id,
                        request_data=modified_data,
                        context=ctx,
                    )
                    decision = self._effective_decision(result.decision, session_id)
                    if _eval_span is not None and hasattr(_eval_span, "set_attribute"):
                        _eval_span.set_attribute(
                            "openbias.evaluator.decision", decision.value
                        )
                    all_metadata["results"].append(
                        {"evaluator": evaluator.name, "decision": decision.value}
                    )

                    if decision == Decision.BLOCK:
                        logger.warning(
                            f"Request blocked by sync evaluator '{evaluator.name}': "
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
                                f"Applying sync message replacement from '{evaluator.name}'"
                            )
                            modified_data = dict(modified_data)
                            modified_data["messages"] = result.modified_messages
                        elif result.message:
                            logger.info(
                                f"Applying sync intervention from '{evaluator.name}'"
                            )
                            modified_data = self._apply_intervention(
                                modified_data, result.message, self._default_strategy
                            )

                except Exception as e:
                    logger.error(f"Evaluator '{evaluator.name}' failed: {e}")
                    # Fail-open: log and continue instead of blocking
                    all_metadata["results"].append(
                        {"evaluator": evaluator.name, "decision": "error", "error": str(e)}
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
        parent_span: Any | None = None,
        span_factory: Any | None = None,
        async_span_group: Any | None = None,
    ) -> InterceptionResult:
        """
        Run POST_CALL phase.

        1. Run sync POST_CALL evaluators (ALLOW, BLOCK, or INTERVENE)
        2. Start async POST_CALL evaluators in background (don't wait)
        3. Return result with optional modified_data for response modification
        """
        self._sessions.touch(session_id)
        self._sessions.evict_stale()

        all_metadata: dict[str, Any] = {"results": [], "interventions": []}
        modified_data: dict[str, Any] | None = None

        # Step 1: Run sync POST_CALL evaluators
        for evaluator in self._sync_post_call_evaluators:
            _eval_ctx = (
                span_factory(evaluator.name, "post_call")
                if span_factory is not None
                else nullcontext()
            )
            with _eval_ctx as _eval_span:
                try:
                    ctx: dict[str, Any] = {
                        "user_request_id": user_request_id,
                        "_parent_span": _eval_span if _eval_span is not None else parent_span,
                    }
                    result = await evaluator.evaluate_response(
                        session_id=session_id,
                        response_data=response_data,
                        request_data=request_data,
                        context=ctx,
                    )
                    decision = self._effective_decision(result.decision, session_id)
                    if _eval_span is not None and hasattr(_eval_span, "set_attribute"):
                        _eval_span.set_attribute(
                            "openbias.evaluator.decision", decision.value
                        )
                    all_metadata["results"].append(
                        {"evaluator": evaluator.name, "decision": decision.value}
                    )

                    if decision == Decision.BLOCK:
                        logger.warning(
                            f"Response blocked by sync evaluator '{evaluator.name}': "
                            f"{result.message}"
                        )
                        return InterceptionResult(
                            allowed=False,
                            message=result.message,
                            metadata=all_metadata,
                        )

                    if decision == Decision.INTERVENE:
                        logger.info(
                            f"Sync POST_CALL evaluator '{evaluator.name}' returned INTERVENE: "
                            f"{result.message}"
                        )
                        intervention_info: dict[str, Any] = {
                            "evaluator": evaluator.name,
                            "message": result.message,
                        }
                        if result.modified_messages is not None:
                            intervention_info["has_modified_messages"] = True
                        all_metadata["interventions"].append(intervention_info)
                        # Store the evaluator result for the hooks layer to act on
                        if modified_data is None:
                            modified_data = {}
                        modified_data.setdefault("_interventions", []).append(
                            {
                                "evaluator": evaluator.name,
                                "message": result.message,
                                "modified_messages": result.modified_messages,
                                "metadata": result.metadata,
                            }
                        )

                except Exception as e:
                    logger.error(f"Evaluator '{evaluator.name}' failed: {e}")
                    all_metadata["results"].append(
                        {"evaluator": evaluator.name, "decision": "error", "error": str(e)}
                    )

        # Step 2: Start async POST_CALL evaluators in background
        # Note: _parent_span may be ended before async tasks finish, so
        # child spans may fall outside the parent's time window in the
        # trace backend.  This is acceptable — async results are applied
        # on the *next* request, and the span still provides lineage.
        #
        # The dispatched span from async_span_group is a zero-duration
        # marker recording *that* an async evaluator was started.  It is
        # intentionally not passed as _parent_span to the evaluator
        # because the evaluator runs long after this request's spans end.
        for evaluator in self._async_post_call_evaluators:
            _dispatch_ctx = (
                async_span_group.dispatched(evaluator.name)
                if async_span_group is not None
                else nullcontext()
            )
            with _dispatch_ctx:
                self._start_async_evaluator(
                    evaluator, session_id, request_data, response_data,
                    context={"user_request_id": user_request_id, "_parent_span": parent_span},
                )

        return InterceptionResult(
            allowed=True,
            modified_data=modified_data,
            metadata=all_metadata,
        )

    def _effective_decision(self, decision: Decision, session_id: str) -> Decision:
        """Map evaluator decision based on fail_action policy and intervention count."""
        if self._fail_action == "shadow":
            if decision != Decision.ALLOW:
                logger.info("Shadow mode: downgrading %s to allow", decision.value)
            return Decision.ALLOW

        if decision == Decision.INTERVENE and self._fail_action == "block":
            return Decision.BLOCK

        # Escalate INTERVENE to BLOCK when max_intervention_attempts exceeded
        if decision == Decision.INTERVENE:
            count = self._intervention_counts.get(session_id, 0) + 1
            self._intervention_counts[session_id] = count
            if count > self._max_intervention_attempts:
                logger.warning(
                    f"Session {session_id} exceeded max intervention attempts "
                    f"({self._max_intervention_attempts}), escalating to BLOCK"
                )
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
                    logger.warning("Async evaluator task was cancelled")
                    results.append(
                        _PendingResult(
                            evaluator_name="async_evaluator_cancelled",
                            result=EngineResult(
                                decision=Decision.ALLOW,
                                message="Async evaluator cancelled",
                                metadata={},
                            ),
                        )
                    )
                except Exception as e:
                    logger.error(f"Async evaluator task failed: {e}")
                    results.append(
                        _PendingResult(
                            evaluator_name="async_evaluator_error",
                            result=EngineResult(
                                decision=Decision.ALLOW,
                                message=f"Async evaluator error: {e}",
                                metadata={"error": str(e)},
                            ),
                        )
                    )

        # Store completed tasks for later removal by _confirm_collected
        self._last_collected.setdefault(session_id, []).extend(completed_tasks)

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

    def _start_async_evaluator(
        self,
        evaluator: PolicyEngine,
        session_id: str,
        request_data: dict[str, Any],
        response_data: Any,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Start an async evaluator task in the background."""

        async def run_evaluator() -> _PendingResult:
            try:
                if response_data is None:
                    result = await evaluator.evaluate_request(
                        session_id=session_id,
                        request_data=request_data,
                        context=context,
                    )
                else:
                    result = await evaluator.evaluate_response(
                        session_id=session_id,
                        response_data=response_data,
                        request_data=request_data,
                        context=context,
                    )
                return _PendingResult(evaluator_name=evaluator.name, result=result)
            except Exception as e:
                logger.error(f"Async evaluator '{evaluator.name}' failed: {e}")
                return _PendingResult(
                    evaluator_name=evaluator.name,
                    result=EngineResult(
                        decision=Decision.ALLOW,
                        message=f"Async evaluator error: {e}",
                        metadata={"error": str(e)},
                    ),
                )

        # Enforce per-session async task cap
        tasks = self._sessions.get(session_id)
        if tasks is None:
            tasks = []
            self._sessions.put(session_id, tasks)

        # Prune completed tasks
        tasks[:] = [t for t in tasks if not t.done()]

        task = asyncio.create_task(run_evaluator())
        tasks.append(task)

        logger.debug(f"Started async evaluator '{evaluator.name}' for session {session_id}")

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

    def _on_session_evict(self, session_id: str, tasks: list[asyncio.Task[_PendingResult]]) -> None:
        """Cancel all async tasks and clean up intervention count when a session is evicted."""
        for task in tasks:
            if not task.done():
                task.cancel()
        self._intervention_counts.pop(session_id, None)

    async def cleanup_session(self, session_id: str) -> None:
        """Cancel running async tasks and clear pending results for a session."""
        tasks = self._sessions.remove(session_id)
        if tasks:
            for task in tasks:
                if not task.done():
                    task.cancel()

        # Clear intervention count for this session
        self._intervention_counts.pop(session_id, None)

        logger.debug(f"Cleaned up session {session_id}")

    async def shutdown(self) -> None:
        """Shutdown the interceptor, cancelling all running async tasks."""
        for session_id in list(self._sessions.keys()):
            await self.cleanup_session(session_id)

        logger.info("Interceptor shutdown complete")
