"""
Interceptor orchestrator.

Manages the execution of policy evaluators at PRE_CALL and POST_CALL phases,
handles async evaluator task management, and applies interventions.
"""

import asyncio
import copy
import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Literal

from openbias.core.intervention.strategies import (
    StrategyType,
    SystemPromptAppendStrategy,
    UserMessageInjectStrategy,
)
from openbias.core.intervention.pipelines import (
    ActionPipeline,
    BlockPipeline,
    IntervenePipeline,
    PostCallPipelineContext,
    PreCallPipelineContext,
    ShadowPipeline,
)
from openbias.core.intervention.pipelines.aggregation import ViolationAggregationStage
from openbias.core.intervention.pipelines.instruction_builder import (
    DeterministicRepairInstructionBuilder,
)
from openbias.core.session import SessionStore
from openbias.policy.protocols import (
    EvaluationResult,
    EvaluationStatus,
    PolicyEngine,
)

from .types import InterceptionResult

logger = logging.getLogger(__name__)


@dataclass
class _MappedResult:
    """Internal: an EvaluationResult mapped to an enforcement decision."""

    decision: Literal["allow", "shadow", "block", "intervene"]
    message: str | None
    metadata: dict[str, Any]


@dataclass
class _PendingResult:
    """Internal: pairs an EvaluationResult with the evaluator name that produced it."""

    evaluator_name: str
    result: EvaluationResult


@dataclass
class _AsyncTraceContext:
    """Serializable tracing context propagated across async evaluator boundaries."""

    session_id: str
    request_id: str
    evaluator_name: str
    origin_trace_id: str | None = None
    origin_span_id: str | None = None

    def as_dict(self) -> dict[str, str]:
        payload = {
            "session_id": self.session_id,
            "request_id": self.request_id,
            "evaluator_name": self.evaluator_name,
        }
        if self.origin_trace_id:
            payload["origin_trace_id"] = self.origin_trace_id
        if self.origin_span_id:
            payload["origin_span_id"] = self.origin_span_id
        return payload


class Interceptor:
    """
    Sole enforcement gateway for policy evaluation results.

    Engines are pure evaluators that return ALLOW or VIOLATION outcomes.
    The Interceptor is the only layer that maps evaluation outcomes to
    enforcement actions based on the ``fail_action`` policy:

    - ``fail_action="intervene"``: VIOLATION → modify request/response
    - ``fail_action="block"``: VIOLATION → reject request/response
    - ``fail_action="shadow"``: VIOLATION → log only, allow through

    Manages:
    - Sync evaluators that gate and must complete before proceeding
    - Async evaluators that run in background with results applied next request
    - Pending async results per session
    - Modification merging for deferred intervention decisions
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
        self._action_pipelines: dict[str, ActionPipeline] = {
            "shadow": ShadowPipeline(),
            "block": BlockPipeline(),
            "intervene": IntervenePipeline(),
        }
        self._aggregation_stage = ViolationAggregationStage()
        self._instruction_builder = DeterministicRepairInstructionBuilder()

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
        has_modifications = False
        all_metadata: dict[str, Any] = {"results": []}

        # Step 1: Apply pending async results
        pending_results = self._collect_completed_async(session_id)
        for processed_count, pending in enumerate(pending_results, start=1):
            _async_ctx = (
                span_factory(pending.evaluator_name, "async_applied")
                if span_factory is not None
                else nullcontext()
            )
            with _async_ctx as _async_span:
                if _async_span is not None and hasattr(_async_span, "set_attribute"):
                    _async_span.set_attribute(
                        "openbias.evaluator.source", "async_applied"
                    )
                    _async_span.set_attribute("openbias.async.phase", "applied")
                mapped = self._map_evaluation(pending.result, session_id)
                mapped.metadata["_openbias_source"] = "async_applied"
                all_metadata["results"].append(
                    {"evaluator": pending.evaluator_name, "decision": mapped.decision}
                )
                modified_data, changed, terminal = self._dispatch_pre_call_action(
                    mapped=mapped,
                    session_id=session_id,
                    evaluator_name=pending.evaluator_name,
                    modified_data=modified_data,
                    all_metadata=all_metadata,
                )
                if changed:
                    has_modifications = True
                if terminal is not None:
                    self._confirm_collected(session_id, count=processed_count)
                    return terminal

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
                    eval_result = await evaluator.evaluate_request(
                        session_id=session_id,
                        request_data=modified_data,
                        context=ctx,
                    )
                    mapped = self._map_evaluation(eval_result, session_id)
                    mapped.metadata["_openbias_source"] = "sync_pre_call"
                    if _eval_span is not None and hasattr(_eval_span, "set_attribute"):
                        _eval_span.set_attribute(
                            "openbias.evaluator.decision", mapped.decision
                        )
                    all_metadata["results"].append(
                        {"evaluator": evaluator.name, "decision": mapped.decision}
                    )
                    modified_data, changed, terminal = self._dispatch_pre_call_action(
                        mapped=mapped,
                        session_id=session_id,
                        evaluator_name=evaluator.name,
                        modified_data=modified_data,
                        all_metadata=all_metadata,
                    )
                    if changed:
                        has_modifications = True
                    if terminal is not None:
                        return terminal

                except Exception as e:
                    logger.error(f"Evaluator '{evaluator.name}' failed: {e}")
                    # Fail-open: log and continue instead of blocking
                    all_metadata["results"].append(
                        {"evaluator": evaluator.name, "decision": "error", "error": str(e)}
                    )

        modified_data, finalized = self._finalize_pre_call_interventions(
            modified_data=modified_data,
            all_metadata=all_metadata,
        )
        if finalized:
            has_modifications = True

        return InterceptionResult(
            allowed=True,
            modified_data=modified_data if has_modifications else None,
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
        async_span_factory: Any | None = None,
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
                    eval_result = await evaluator.evaluate_response(
                        session_id=session_id,
                        response_data=response_data,
                        request_data=request_data,
                        context=ctx,
                    )
                    mapped = self._map_evaluation(eval_result, session_id)
                    mapped.metadata["_openbias_source"] = "sync_post_call"
                    if _eval_span is not None and hasattr(_eval_span, "set_attribute"):
                        _eval_span.set_attribute(
                            "openbias.evaluator.decision", mapped.decision
                        )
                    all_metadata["results"].append(
                        {"evaluator": evaluator.name, "decision": mapped.decision}
                    )
                    modified_data, terminal = self._dispatch_post_call_action(
                        mapped=mapped,
                        session_id=session_id,
                        evaluator_name=evaluator.name,
                        modified_data=modified_data,
                        all_metadata=all_metadata,
                    )
                    if terminal is not None:
                        return terminal

                except Exception as e:
                    logger.error(f"Evaluator '{evaluator.name}' failed: {e}")
                    all_metadata["results"].append(
                        {"evaluator": evaluator.name, "decision": "error", "error": str(e)}
                    )

        modified_data = self._finalize_post_call_interventions(
            modified_data=modified_data,
            all_metadata=all_metadata,
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
                span_factory(evaluator.name, "post_call")
                if span_factory is not None
                else nullcontext()
            )
            with _dispatch_ctx as _dispatch_span:
                if _dispatch_span is not None and hasattr(_dispatch_span, "set_attribute"):
                    _dispatch_span.set_attribute(
                        "openbias.evaluator.source", "async_dispatched"
                    )
                    _dispatch_span.set_attribute("openbias.async.phase", "dispatched")
                    if user_request_id:
                        _dispatch_span.set_attribute("openbias.request_id", user_request_id)
                self._start_async_evaluator(
                    evaluator, session_id, request_data, response_data,
                    context={
                        "user_request_id": user_request_id,
                        "_async_trace_context": self._build_async_trace_context(
                            session_id=session_id,
                            request_id=user_request_id,
                            evaluator_name=evaluator.name,
                            dispatch_span=_dispatch_span,
                        ),
                    },
                    async_span_factory=async_span_factory,
                )

        return InterceptionResult(
            allowed=True,
            modified_data=modified_data,
            metadata=all_metadata,
        )

    def _dispatch_pre_call_action(
        self,
        mapped: _MappedResult,
        session_id: str,
        evaluator_name: str,
        modified_data: dict[str, Any],
        all_metadata: dict[str, Any],
    ) -> tuple[dict[str, Any], bool, InterceptionResult | None]:
        """Route pre-call action handling through the selected pipeline."""
        if mapped.decision == "allow":
            return modified_data, False, None
        pipeline = self._action_pipelines[mapped.decision]
        context = PreCallPipelineContext(
            session_id=session_id,
            evaluator_name=evaluator_name,
            message=mapped.message,
            mapped_metadata=mapped.metadata,
            modified_data=modified_data,
            all_metadata=all_metadata,
            default_strategy=self._default_strategy,
            apply_intervention=self._apply_intervention,
        )
        terminal = pipeline.handle_pre_call(context)
        return context.modified_data, context.has_modifications, terminal

    def _dispatch_post_call_action(
        self,
        mapped: _MappedResult,
        session_id: str,
        evaluator_name: str,
        modified_data: dict[str, Any] | None,
        all_metadata: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, InterceptionResult | None]:
        """Route post-call action handling through the selected pipeline."""
        if mapped.decision == "allow":
            return modified_data, None
        pipeline = self._action_pipelines[mapped.decision]
        context = PostCallPipelineContext(
            session_id=session_id,
            evaluator_name=evaluator_name,
            message=mapped.message,
            mapped_metadata=mapped.metadata,
            modified_data=modified_data,
            all_metadata=all_metadata,
        )
        terminal = pipeline.handle_post_call(context)
        return context.modified_data, terminal

    def _finalize_pre_call_interventions(
        self,
        *,
        modified_data: dict[str, Any],
        all_metadata: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        records = all_metadata.pop("_pending_pre_call_interventions", [])
        if not records:
            return modified_data, False

        mode: Literal["sync", "async"] = (
            "async"
            if all(record.get("source") == "async_applied" for record in records)
            else "sync"
        )
        aggregated = self._aggregation_stage.aggregate(records=records, mode=mode)
        recent_messages = self._extract_recent_message_text(modified_data)
        payload = self._instruction_builder.build(
            aggregated=aggregated,
            recent_messages=recent_messages,
        )

        all_metadata["interventions"] = [{"evaluator": "merged", "message": payload.repair_instruction}]
        all_metadata["intervention_payload"] = asdict(payload)

        applied = self._apply_intervention(
            modified_data,
            payload.repair_instruction,
            self._default_strategy,
        )
        if applied is None:
            return modified_data, False
        return applied, True

    def _finalize_post_call_interventions(
        self,
        *,
        modified_data: dict[str, Any] | None,
        all_metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        records = all_metadata.pop("_pending_post_call_interventions", [])
        if not records:
            return modified_data

        aggregated = self._aggregation_stage.aggregate(records=records, mode="sync")
        payload = self._instruction_builder.build(aggregated=aggregated, recent_messages=None)

        if modified_data is None:
            modified_data = {}
        modified_data["_interventions"] = [
            {
                "evaluator": "merged",
                "message": payload.repair_instruction,
                "metadata": {
                    "source_violations": payload.source_violations,
                    "merged_violation_summary": payload.merged_violation_summary,
                },
                "payload": asdict(payload),
            }
        ]
        all_metadata["interventions"] = [{"evaluator": "merged", "message": payload.repair_instruction}]
        all_metadata["intervention_payload"] = asdict(payload)
        return modified_data

    @staticmethod
    def _extract_recent_message_text(request_data: dict[str, Any]) -> list[str]:
        messages = request_data.get("messages", [])
        if not isinstance(messages, list):
            return []
        recent: list[str] = []
        for message in messages[-6:]:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                text = content.strip()
                if text:
                    recent.append(text)
        return recent

    def _map_evaluation(self, eval_result: EvaluationResult, session_id: str) -> _MappedResult:
        """Map an EvaluationResult to an enforcement decision.

        This is the sole enforcement gateway: engines report pure evaluation
        outcomes (ALLOW or VIOLATION), and this method converts them to the
        actual runtime action based on fail_action policy.
        """
        if eval_result.status == EvaluationStatus.ALLOW:
            return _MappedResult(
                decision="allow",
                message=None,
                metadata=eval_result.metadata,
            )

        # VIOLATION → map to enforcement action
        message = "\n".join(v.reason for v in eval_result.violations) if eval_result.violations else None
        meta = dict(eval_result.metadata)
        meta["violations"] = [
            {
                "message": v.reason,
                "severity": v.severity,
                "scope": v.scope,
                "engine": v.engine,
                **({"confidence": v.confidence} if v.confidence is not None else {}),
                **v.extra,
            }
            for v in eval_result.violations
        ]

        if self._fail_action == "shadow":
            logger.info("Shadow mode: routing violation to shadow pipeline")
            return _MappedResult(decision="shadow", message=None, metadata=meta)

        if self._fail_action == "block":
            return _MappedResult(decision="block", message=message, metadata=meta)

        # fail_action == "intervene" (default)
        return _MappedResult(decision="intervene", message=message, metadata=meta)

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
                            result=EvaluationResult(
                                status=EvaluationStatus.ALLOW,
                                metadata={"cancelled": True},
                            ),
                        )
                    )
                except Exception as e:
                    logger.error(f"Async evaluator task failed: {e}")
                    results.append(
                        _PendingResult(
                            evaluator_name="async_evaluator_error",
                            result=EvaluationResult(
                                status=EvaluationStatus.ALLOW,
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
        async_span_factory: Any | None = None,
    ) -> None:
        """Start an async evaluator task in the background."""

        async def run_evaluator() -> _PendingResult:
            eval_context = dict(context or {})
            trace_context = eval_context.get("_async_trace_context")
            _exec_ctx = (
                async_span_factory(evaluator.name, trace_context)
                if async_span_factory is not None
                else nullcontext()
            )
            try:
                with _exec_ctx as _exec_span:
                    if _exec_span is not None:
                        eval_context["_parent_span"] = _exec_span
                    if response_data is None:
                        result = await evaluator.evaluate_request(
                            session_id=session_id,
                            request_data=request_data,
                            context=eval_context,
                        )
                    else:
                        result = await evaluator.evaluate_response(
                            session_id=session_id,
                            response_data=response_data,
                            request_data=request_data,
                            context=eval_context,
                        )
                return _PendingResult(evaluator_name=evaluator.name, result=result)
            except Exception as e:
                logger.error(f"Async evaluator '{evaluator.name}' failed: {e}")
                return _PendingResult(
                    evaluator_name=evaluator.name,
                    result=EvaluationResult(
                        status=EvaluationStatus.ALLOW,
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

    @staticmethod
    def _build_async_trace_context(
        session_id: str,
        request_id: str,
        evaluator_name: str,
        dispatch_span: Any | None,
    ) -> dict[str, str]:
        """Build a serializable async trace payload from dispatch-time context."""
        payload = _AsyncTraceContext(
            session_id=session_id,
            request_id=request_id,
            evaluator_name=evaluator_name,
        )
        if dispatch_span is not None and hasattr(dispatch_span, "get_span_context"):
            try:
                span_context = dispatch_span.get_span_context()
                if span_context is not None:
                    payload.origin_trace_id = format(span_context.trace_id, "032x")
                    payload.origin_span_id = format(span_context.span_id, "016x")
            except Exception:
                logger.debug("Failed to serialize async trace context", exc_info=True)
        return payload.as_dict()

    def _apply_intervention(
        self,
        request_data: dict[str, Any],
        message: str,
        strategy: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Apply an intervention message to request data using the configured strategy.

        Args:
            request_data: The LLM request data to modify.
            message: The intervention guidance text.
            strategy: Strategy name override; falls back to self._default_strategy.

        Returns:
            New request data dict with the intervention applied, or None if
            the strategy cannot be applied at request time (e.g. response_modification).
        """
        result = dict(request_data)
        messages = result.get("messages", [])
        effective_strategy = strategy or self._default_strategy

        if effective_strategy == "user_message_inject":
            result["messages"] = UserMessageInjectStrategy.merge(messages, message)
            return result
        elif effective_strategy == "system_prompt_append":
            result["messages"] = SystemPromptAppendStrategy.merge(messages, message)
            return result
        elif effective_strategy == "response_modification":
            logger.warning(
                "response_modification is a response-time strategy and cannot be applied "
                "during request modification (PRE_CALL); intervention skipped"
            )

        return None

    def _on_session_evict(self, session_id: str, tasks: list[asyncio.Task[_PendingResult]]) -> None:
        """Cancel all async tasks when a session is evicted."""
        pending_count = 0
        for task in tasks:
            if not task.done():
                pending_count += 1
                task.cancel()
        if pending_count:
            logger.warning(
                "openbias.async.phase=dropped dropping %d pending async evaluator tasks for evicted session %s",
                pending_count,
                session_id,
            )

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
