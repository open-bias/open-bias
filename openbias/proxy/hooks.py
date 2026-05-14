"""
Open Bias LiteLLM hooks for workflow monitoring and intervention.

This module implements the core hook system that intercepts LLM calls:

1. async_pre_call_hook: Runs BEFORE LLM call
   - Applies pending async engine results from previous call
   - Runs sync PRE_CALL engines
   - Modifies request if INTERVENE with modified_data

2. async_post_call_success_hook: Runs AFTER LLM call succeeds
   - Runs sync POST_CALL engines (can trigger a replay request on INTERVENE)
   - Starts async engines in background
   - Completes tracing

All hooks are wrapped with fail-open semantics via `safe_hook()`:
- Timeout enforcement (configurable via hook_timeout_seconds)
- Exception catch-all returns fallback (pass-through unchanged)
- WorkflowViolationError (intentional blocks) always propagates
- Failure counter for monitoring

Based on LiteLLM's CustomLogger API:
https://docs.litellm.ai/docs/observability/custom_callback
"""

import asyncio
import contextvars
import json
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth


from openbias.config.settings import Settings
from openbias.core.intervention.strategies import (
    RESPONSE_CLEANUP_METADATA_KEY,
    WorkflowViolationError,
)
from openbias.core.utils import extract_response_content, extract_usage_info
from openbias.core.interceptor import Interceptor, SYNC_POST_REPLAY_KIND
from openbias.core.intervention.pipelines.cleanup import ResponseCleanupStage
from openbias.logging import session_id_var, request_id_var
from openbias.policy.protocols import PolicyEngine
from openbias.proxy.middleware import SessionExtractor

logger = logging.getLogger(__name__)

# ContextVar to carry the per-request OTEL span across hooks without polluting
# data["metadata"] (which gets deepcopied and cannot hold RLock objects).
request_span_var: contextvars.ContextVar[Any] = contextvars.ContextVar("request_span_var", default=None)

# ---------------------------------------------------------------------------
# Fail-open infrastructure
# ---------------------------------------------------------------------------

# Module-level counter tracking fail-open activations per hook.
# Useful for monitoring/alerting without requiring an external metrics library.
_fail_open_counter: dict[str, int] = {}

SYNC_POST_REPLAY_COUNT_KEY = "_openbias_sync_post_replay_count"
SYNC_POST_REPLAY_PARENT_REQUEST_ID_KEY = "_openbias_sync_post_replay_parent_request_id"


def _copy_replay_value(value: Any) -> Any:
    """Clone plain request containers without pickling runtime objects."""
    if isinstance(value, dict):
        return {key: _copy_replay_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_replay_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_replay_value(item) for item in value)
    if isinstance(value, set):
        return {_copy_replay_value(item) for item in value}
    return value


def get_fail_open_counts() -> dict[str, int]:
    """Return a snapshot of fail-open activation counts per hook."""
    return dict(_fail_open_counter)


async def safe_hook(
    hook_fn: Callable,
    *args: Any,
    timeout: float = 5.0,
    fallback: Any = None,
    hook_name: str = "unknown",
    fail_open: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Execute a hook with timeout and fail-open/fail-closed semantics.

    If the hook raises ``WorkflowViolationError`` it is intentional (a policy
    block) and is re-raised regardless of the ``fail_open`` setting.

    When ``fail_open=True`` (default), every other exception — including
    ``asyncio.TimeoutError`` — is caught, logged, counted, and the *fallback*
    value is returned so the agent's request passes through unchanged.

    When ``fail_open=False``, exceptions are re-raised after logging, causing
    the proxy to deny the request on hook failure.

    Args:
        hook_fn:   Async callable to execute.
        timeout:   Maximum seconds before the hook is cancelled.
        fallback:  Value to return on failure/timeout.
        hook_name: Human-readable name for logging and metrics.
        fail_open: If True, swallow exceptions and return fallback.
                   If False, re-raise exceptions after logging.
    """
    try:
        return await asyncio.wait_for(hook_fn(*args, **kwargs), timeout=timeout)
    except WorkflowViolationError:
        raise  # Intentional policy blocks must propagate
    except asyncio.TimeoutError:
        if fail_open:
            _fail_open_counter[hook_name] = _fail_open_counter.get(hook_name, 0) + 1
            logger.error(
                f"Open Bias hook '{hook_name}' timed out after {timeout}s "
                f"(fail-open, count={_fail_open_counter[hook_name]})"
            )
            return fallback
        logger.error(
            f"Open Bias hook '{hook_name}' timed out after {timeout}s (fail-closed)"
        )
        raise
    except Exception as e:
        if fail_open:
            _fail_open_counter[hook_name] = _fail_open_counter.get(hook_name, 0) + 1
            logger.error(
                f"Open Bias hook '{hook_name}' failed (fail-open, "
                f"count={_fail_open_counter[hook_name]}): {e}"
            )
            return fallback
        logger.error(
            f"Open Bias hook '{hook_name}' failed (fail-closed): {e}"
        )
        raise

# Type alias for call types
CallType = Literal[
    "completion",
    "text_completion",
    "embeddings",
    "image_generation",
    "moderation",
    "audio_transcription",
    "acompletion",
    "atext_completion",
    "aembeddings",
]




class EvaluatorSpanFactory:
    """Creates per-evaluator child spans under a phase span."""

    def __init__(self, tracer, phase_span, session_id):
        self._tracer = tracer
        self._phase_span = phase_span
        self._session_id = session_id

    @staticmethod
    def _span_name(evaluator_name: str, phase: str) -> str:
        if phase == "dispatch":
            return f"evaluator_dispatch:{evaluator_name}"
        if phase == "async_applied":
            return f"evaluator_applied:{evaluator_name}"
        return f"evaluator:{evaluator_name}"

    @contextmanager
    def __call__(self, evaluator_name: str, phase: str):
        with self._tracer.trace_block(
            self._span_name(evaluator_name, phase),
            self._session_id,
            attributes={
                "openbias.evaluator.name": evaluator_name,
                "openbias.evaluator.phase": phase,
            },
            parent_span=self._phase_span,
        ) as span:
            yield span



class AsyncEvaluatorExecutionSpanFactory:
    """Creates async execution spans linked to dispatch-time trace context."""

    def __init__(self, tracer, session_id):
        self._tracer = tracer
        self._session_id = session_id

    @contextmanager
    def __call__(self, evaluator_name: str, trace_context: dict[str, Any] | None = None):
        attrs = {
            "openbias.evaluator.name": evaluator_name,
            "openbias.evaluator.phase": "async_execute",
            "openbias.async.phase": "executing",
        }
        links = []
        if trace_context:
            request_id = trace_context.get("request_id")
            origin_trace_id = trace_context.get("origin_trace_id")
            origin_span_id = trace_context.get("origin_span_id")
            if request_id:
                attrs["openbias.request_id"] = request_id
            if origin_trace_id:
                attrs["openbias.origin.trace_id"] = origin_trace_id
            if origin_span_id:
                attrs["openbias.origin.span_id"] = origin_span_id
            link = self._tracer.build_span_link(
                trace_id_hex=origin_trace_id,
                span_id_hex=origin_span_id,
            )
            if link is not None:
                links.append(link)

        with self._tracer.trace_block(
            f"evaluator_execution:{evaluator_name}",
            self._session_id,
            attributes=attrs,
            links=links or None,
        ) as span:
            yield span


class Callback(CustomLogger):
    """
    Main Open Bias callback for LiteLLM.

    Implements policy enforcement through LiteLLM's hook system.
    This is registered as a callback when the proxy starts.

    Uses the Interceptor to orchestrate evaluators:
    - Sync PRE_CALL evaluators run before LLM call
    - Sync POST_CALL evaluators run after LLM call (can replay the request)
    - Async evaluators run in background, results applied next request

    The callback maintains:
    - Interceptor instance with configured evaluators
    - OpenTelemetry tracer for observability

    Thread-safety is ensured through asyncio locks for session state.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._response_cleanup = ResponseCleanupStage()
        self._router: Any | None = None

        # Interceptor (lazy initialized)
        self._interceptor: Interceptor | None = None
        self._interceptor_initialized = False

        # All evaluator engine instances (for shutdown/cleanup)
        self._evaluators: list[PolicyEngine] = []

        # Lock for lazy init to prevent concurrent initialization races
        self._init_lock = asyncio.Lock()

        # OpenTelemetry tracer for Open Bias events (lazy initialized)
        self._tracer = None

        # Effective hook timeout — starts at the configured value; may be
        # auto-adjusted upward after engine initialization to avoid racing
        # the engine's own LLM timeouts (see _compute_hook_timeout).
        self._effective_hook_timeout: float = self.settings.hook_timeout_seconds

        logger.info("Callback initialized")

        if self.settings.mode != "sync":
            logger.warning(
                "POST_CALL mode is '%s' — the first policy violation in each session "
                "will pass through to the user before intervention is applied on the "
                "next request. Set mode='sync' in config for immediate "
                "enforcement (adds latency).",
                self.settings.mode,
            )

    def attach_router(self, router: Any) -> None:
        """Attach the active LiteLLM router for sync replay requests."""
        self._router = router

    async def _get_interceptor(self) -> Interceptor | None:
        """Lazy-load interceptor with configured evaluators."""
        if self._interceptor_initialized:
            return self._interceptor

        async with self._init_lock:
            # Double-check after acquiring lock
            if self._interceptor_initialized:
                return self._interceptor

            return await self._initialize_evaluators()

    async def _initialize_evaluators(self) -> Interceptor | None:
        """Create evaluator engines and build the Interceptor (fail-open). Must be called under _init_lock."""
        try:
            self._interceptor = self._build_evaluators_and_interceptor(
                *(await self._create_evaluator_engines())
            )
        except Exception as e:
            logger.error(f"Failed to initialize evaluators: {e}", exc_info=True)
            # _create_evaluator_engines already shuts down partially-created
            # engines on failure, so just reset state here.
            self._evaluators = []
            self._interceptor = None

        self._interceptor_initialized = True
        self._effective_hook_timeout = self._compute_hook_timeout()
        return self._interceptor

    async def _create_evaluator_engines(
        self,
    ) -> tuple[list[PolicyEngine], list[PolicyEngine], list[PolicyEngine]]:
        """Instantiate evaluator engines from settings. Returns (all, pre_call, post_call).

        Engines are collected into local lists so that ``self._evaluators``
        is only assigned after all engines succeed (via ``_build_evaluators_and_interceptor``).

        If the Nth engine fails, all previously created engines are shut down
        before the exception propagates, preventing orphaned engine instances.
        """
        from openbias.policy.registry import PolicyEngineRegistry

        pre_call_evaluators: list[PolicyEngine] = []
        post_call_evaluators: list[PolicyEngine] = []
        all_evaluators: list[PolicyEngine] = []

        try:
            for ev_config in self.settings.evaluators:
                logger.info(
                    f"Initializing evaluator '{ev_config.name}' "
                    f"(type={ev_config.type}, phase={ev_config.phase})"
                )
                config = dict(ev_config.config)
                self.settings.inject_default_model(
                    ev_config.type, config, self.settings.proxy.default_model
                )
                engine = await PolicyEngineRegistry.create_and_initialize(
                    ev_config.type, config
                )
                # Wire up tracer for engines that support it (e.g. judge engine)
                if hasattr(engine, "set_tracer") and self.tracer:
                    engine.set_tracer(self.tracer)

                all_evaluators.append(engine)
                if ev_config.phase == "pre_call":
                    pre_call_evaluators.append(engine)
                else:
                    post_call_evaluators.append(engine)
        except Exception:
            # Shut down engines created before the failure
            for engine in all_evaluators:
                try:
                    await engine.shutdown()
                except Exception as shutdown_err:
                    logger.error(f"Error shutting down evaluator during cleanup: {shutdown_err}")
            raise

        return all_evaluators, pre_call_evaluators, post_call_evaluators

    def _build_evaluators_and_interceptor(
        self,
        all_evaluators: list[PolicyEngine],
        pre_call_evaluators: list[PolicyEngine],
        post_call_evaluators: list[PolicyEngine],
    ) -> Interceptor | None:
        """Assign evaluators and build the Interceptor. Returns None when no evaluators exist."""
        self._evaluators = all_evaluators

        if not all_evaluators:
            return None

        interceptor = Interceptor(
            pre_call_evaluators=pre_call_evaluators,
            post_call_evaluators=post_call_evaluators,
            mode=self.settings.mode,
            default_strategy=self.settings.strategy,
            fail_action=self.settings.fail_action,
            session_ttl=self.settings.session_ttl,
            max_sessions=self.settings.max_sessions,
        )
        logger.info("Interceptor initialized with %d evaluator(s)", len(all_evaluators))
        return interceptor

    def _compute_hook_timeout(self) -> float:
        """Compute a hook timeout that won't race any engine's own LLM timeouts.

        Takes the max timeout across all evaluator engines.  When any engine's
        per-model timeout plus a safety buffer exceeds the configured
        hook_timeout_seconds, we auto-adjust upward and log a warning so
        operators are aware.

        For engines without a timeout attribute (or when no engines are
        initialized) the configured value is returned unchanged.
        """
        configured = self.settings.hook_timeout_seconds

        if not self._evaluators:
            return configured

        max_needed = configured
        for engine in self._evaluators:
            engine_timeout = getattr(engine, "timeout", None)
            if not isinstance(engine_timeout, (int, float)):
                continue
            if engine_timeout is not None and engine_timeout > 0:
                needed = engine_timeout + 5.0
                if needed > max_needed:
                    max_needed = needed

        if max_needed > configured:
            # Find the engine with the largest timeout for the warning message
            max_engine_timeout = max(
                (getattr(e, "timeout", 0) for e in self._evaluators
                 if isinstance(getattr(e, "timeout", None), (int, float))),
                default=0,
            )
            logger.warning(
                "hook_timeout_seconds (%.1fs) is shorter than engine timeout "
                "(%.1fs). Auto-adjusting to %.1fs to prevent premature "
                "cancellation.",
                configured,
                max_engine_timeout,
                max_needed,
            )
            return max_needed

        return configured

    async def initialize(self) -> None:
        """
        Eagerly initialize evaluator engines and the interceptor.

        Called at startup by Proxy.initialize() so that configuration
        errors surface immediately rather than being silently swallowed on the
        first request.  Unlike the lazy path, this method does NOT catch
        exceptions — callers receive the error directly.
        """
        async with self._init_lock:
            if not self._interceptor_initialized:
                # Raises on failure — no try/except intentionally
                self._interceptor = self._build_evaluators_and_interceptor(
                    *(await self._create_evaluator_engines())
                )
                self._interceptor_initialized = True

            # Recompute the effective hook timeout now that engines are known.
            self._effective_hook_timeout = self._compute_hook_timeout()

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up all state for a session (interceptor tasks + engine sessions)."""
        if self._interceptor is not None:
            await self._interceptor.cleanup_session(session_id)
        for engine in self._evaluators:
            if hasattr(engine, "reset_session"):
                await engine.reset_session(session_id)
        logger.debug(f"Cleaned up session {session_id} from hooks")

    async def shutdown(self) -> None:
        """
        Shutdown the callback, cleaning up interceptor and policy engine.

        Cancels running async tasks, clears pending session state,
        and shuts down the policy engine.
        """
        logger.info("Callback shutting down...")

        if self._interceptor is not None:
            try:
                await self._interceptor.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down interceptor: {e}")

        for engine in self._evaluators:
            try:
                await engine.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down evaluator '{engine.name}': {e}")

        if self._tracer is not None:
            try:
                self._tracer.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down tracer: {e}")

        logger.info("Callback shutdown complete")

    @property
    def tracer(self) -> Any:
        """Lazy-load tracer for Open Bias event logging via OpenTelemetry."""
        if self._tracer is None:
            logger.debug(f"Tracer check: otel.enabled={self.settings.otel.enabled}")
            if self.settings.otel.enabled:
                from openbias.tracing.otel_tracer import Tracer

                self._tracer = Tracer(self.settings.otel)
                logger.info(f"Tracer initialized: {self._tracer}")
        return self._tracer

    @staticmethod
    def _sync_post_replay_count(data: dict[str, Any]) -> int:
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            return 0
        raw_count = metadata.get(SYNC_POST_REPLAY_COUNT_KEY, 0)
        try:
            return int(raw_count)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _trace_evaluator_summaries(
        results: list[dict[str, Any]] | None,
        *,
        phase: str,
        violations: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        shared_violations = [violation for violation in (violations or []) if isinstance(violation, dict)]
        for item in results or []:
            if not isinstance(item, dict):
                continue
            summaries.append(
                {
                    "name": item.get("evaluator", "unnamed"),
                    "engine_type": item.get("engine_type", "unknown"),
                    "phase": phase,
                    "decision": item.get("decision", "unknown"),
                    "violations": shared_violations if item.get("decision") in {"block", "intervene", "shadow"} else [],
                }
            )
        return summaries

    def _trace_interventions(
        self,
        metadata: dict[str, Any],
        *,
        applied_at: str,
    ) -> list[dict[str, Any]]:
        payload = metadata.get("intervention_payload", {})
        strategy = payload.get("strategy") if isinstance(payload, dict) else None
        interventions: list[dict[str, Any]] = []
        for item in metadata.get("interventions", []):
            if not isinstance(item, dict):
                continue
            interventions.append(
                {
                    "name": item.get("evaluator", "intervention"),
                    "strategy": strategy or self.settings.strategy,
                    "applied_at": applied_at,
                    "message": item.get("message"),
                }
            )
        return interventions

    def _record_jsonl_trace(
        self,
        *,
        session_id: str,
        request_id: str,
        data: dict[str, Any],
        response_data: Any,
        final_action: str,
        internal_metadata: dict[str, Any] | None = None,
        phase: str,
        source: dict[str, Any] | None = None,
    ) -> None:
        if not self.tracer or not getattr(self.tracer, "has_jsonl_sink", False):
            return

        metadata = dict(internal_metadata or {})
        trace_metadata = {
            "model": data.get("model"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": "proxy",
            "policy_hash": metadata.get("policy_hash"),
            "request_id": request_id,
            "evaluator_names": tuple(
                item.get("evaluator", "unnamed")
                for item in metadata.get("results", [])
                if isinstance(item, dict)
            ),
            "tags": (),
            "phase": phase,
        }
        self.tracer.record_trace_case(
            case_id=request_id or str(uuid.uuid4()),
            session_id=session_id,
            request_messages=data.get("messages", []),
            response_data=response_data,
            metadata=trace_metadata,
            final_action=final_action,
            evaluator_summaries=self._trace_evaluator_summaries(
                metadata.get("results"),
                phase=phase,
                violations=metadata.get("violations"),
            ),
            interventions=self._trace_interventions(
                metadata,
                applied_at="next_turn" if final_action == "intervene" else "post_call",
            ),
            source=source,
        )

    def _build_sync_post_replay_request(
        self,
        *,
        data: dict[str, Any],
        pending_intervention: dict[str, Any],
        session_id: str,
        request_id: str,
    ) -> dict[str, Any]:
        replay_request = _copy_replay_value(
            pending_intervention.get("request_data")
            if isinstance(pending_intervention.get("request_data"), dict)
            else data
        )
        replay_metadata = replay_request.setdefault("metadata", {})
        if not isinstance(replay_metadata, dict):
            replay_metadata = {}
            replay_request["metadata"] = replay_metadata

        session_metadata = data.get("metadata", {})
        if isinstance(session_metadata, dict) and session_metadata.get("session_id"):
            replay_metadata["session_id"] = session_metadata["session_id"]
        else:
            replay_metadata["session_id"] = session_id

        replay_metadata[SYNC_POST_REPLAY_COUNT_KEY] = self._sync_post_replay_count(data) + 1
        replay_metadata[SYNC_POST_REPLAY_PARENT_REQUEST_ID_KEY] = request_id
        replay_metadata.pop("_openbias_traced", None)
        replay_metadata.pop("_openbias_request_id", None)
        replay_metadata.pop("_openbias_llm_start_time", None)
        return replay_request

    def _apply_request_cleanup(self, data: dict[str, Any], response: Any) -> Any:
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            return response

        cleanup_rules = metadata.get(RESPONSE_CLEANUP_METADATA_KEY)
        if not isinstance(cleanup_rules, list) or not cleanup_rules:
            return response

        current_content = extract_response_content(response) or ""
        cleaned_content = self._response_cleanup.clean_text(current_content, cleanup_rules)
        if cleaned_content == current_content:
            return response

        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                choice.message.content = cleaned_content
                return response
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices and "message" in choices[0]:
                choices[0]["message"]["content"] = cleaned_content
        return response

    async def _maybe_execute_sync_post_replay(
        self,
        *,
        data: dict[str, Any],
        result: Any,
        session_id: str,
        request_id: str,
    ) -> Any | None:
        pending_intervention = result.pending_intervention
        if not isinstance(pending_intervention, dict):
            return None
        if pending_intervention.get("kind") != SYNC_POST_REPLAY_KIND:
            return None
        if self._router is None:
            logger.warning("Sync POST_CALL replay requested but no router is attached")
            return None
        if self._sync_post_replay_count(data) >= 1:
            logger.info(
                "Skipping sync POST_CALL replay for session %s because the single replay guard is already set",
                session_id,
            )
            return None

        replay_request = self._build_sync_post_replay_request(
            data=data,
            pending_intervention=pending_intervention,
            session_id=session_id,
            request_id=request_id,
        )
        logger.info(
            "Issuing sync POST_CALL replay for session %s via attached router",
            session_id,
        )
        return await self._router.acompletion(**replay_request)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, Any],
        call_type: CallType,
    ) -> Exception | str | dict | None:
        """
        Execute BEFORE LLM call.  Wrapped with fail-open + timeout.

        Returns modified data dict, or original data on failure.
        WorkflowViolationError (intentional blocks) still propagates.
        """
        return await safe_hook(
            self._pre_call_impl,
            user_api_key_dict, cache, data, call_type,
            timeout=self._effective_hook_timeout,
            fallback=data,
            hook_name="async_pre_call_hook",
            fail_open=self.settings.fail_open,
        )

    async def _pre_call_impl(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, Any],
        call_type: CallType,
    ) -> Exception | str | dict | None:
        """Inner implementation for async_pre_call_hook."""
        session_id = SessionExtractor.extract_session_id(data)
        request_span_var.set(None)

        # Persist session ID in metadata to ensure consistency across hooks
        # This prevents generating a new random UUID in post_call/failure hooks
        if "metadata" not in data:
            data["metadata"] = {}

        # Use existing if present, otherwise set the extracted/generated one
        if not data["metadata"].get("session_id"):
            data["metadata"]["session_id"] = session_id

        # Bind context vars so all downstream log records include these IDs
        session_id_var.set(session_id)

        logger.debug(f"pre_call_hook: session={session_id}, call_type={call_type}")

        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["_openbias_request_id"] = request_id

        # Create per-request grouping span for all requests.
        request_span = None
        if self.tracer:
            request_span = self.tracer.start_request_span(session_id, request_id)
        request_span_var.set(request_span)

        interceptor = await self._get_interceptor()
        if interceptor:

            # Wrap in a trace block under the request span
            if self.tracer:
                cm = self.tracer.trace_block(
                    "interceptor_pre_call",
                    session_id,
                    attributes={"hook": "pre_call"},
                    input_data=data.get("messages", []),
                    metadata={
                        "call_type": call_type,
                        "model": data.get("model", "unknown"),
                    },
                    parent_span=request_span,
                    request_id=request_id,
                    phase_order=1,
                )
            else:
                cm = nullcontext()

            with cm as span:
                span_factory = EvaluatorSpanFactory(self.tracer, span, session_id) if self.tracer and span else None

                result = await interceptor.run_pre_call(
                    session_id=session_id,
                    request_data=data,
                    user_request_id=request_id,
                    span_factory=span_factory,
                )

                # Set output on span
                if span is not None:
                    output_data = {
                        "allowed": result.allowed,
                        "has_modifications": result.modified_data is not None,
                    }
                    output_json = json.dumps(output_data, default=str)
                    span.set_attribute("output.value", output_json)
                    span.set_attribute("langfuse.span.output", output_json)
                    span.set_attribute("openbias.decision", "allow" if result.allowed else "block")
                    span.set_attribute("openbias.has_modifications", result.modified_data is not None)

                # Handle result INSIDE trace block so interventions nest correctly
                if not result.allowed:
                    message = result.user_message or "Request blocked by policy"
                    logger.warning(
                        f"Request blocked for session {session_id}: {message}"
                    )
                    self._record_jsonl_trace(
                        session_id=session_id,
                        request_id=request_id,
                        data=data,
                        response_data={"content": message},
                        final_action="block",
                        internal_metadata=result.internal_metadata,
                        phase="pre_call",
                        source={"hook": "async_pre_call_hook", "blocked": True},
                    )
                    if self.tracer and request_span is not None:
                        self.tracer.end_request_span(request_span)
                    request_span_var.set(None)
                    # LiteLLM convention: returning an Exception instance (not raising)
                    # signals the proxy to reject the request with this error message.
                    return Exception(message)

                # Apply modifications if any
                if result.modified_data:
                    data = result.modified_data

                    # Log intervention via OTEL (inside trace block for proper nesting)
                    if self.tracer and result.internal_metadata.get("results"):
                        self.tracer.log_intervention(
                            session_id=session_id,
                            intervention_name="pre_call_intervention",
                            context=result.internal_metadata,
                            parent_span=span,
                        )

                    self._record_jsonl_trace(
                        session_id=session_id,
                        request_id=request_id,
                        data=data,
                        response_data={"content": extract_response_content(data.get("messages", [])[-1]) if data.get("messages") else ""},
                        final_action="intervene",
                        internal_metadata=result.internal_metadata,
                        phase="pre_call",
                        source={"hook": "async_pre_call_hook", "modified_request": True},
                    )

        # Capture start time at the end of pre-call to accurately measure LLM latency in trace
        data["metadata"]["_openbias_llm_start_time"] = time.time()

        return data

    async def async_post_call_success_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """
        Execute AFTER successful LLM response.  Wrapped with fail-open + timeout.

        Returns response (potentially modified), or original response on failure.
        WorkflowViolationError (intentional blocks) still propagates.
        """
        return await safe_hook(
            self._post_call_success_impl,
            data, user_api_key_dict, response,
            timeout=self._effective_hook_timeout,
            fallback=response,
            hook_name="async_post_call_success_hook",
            fail_open=self.settings.fail_open,
        )

    async def _post_call_success_impl(
        self,
        data: dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """Inner implementation for async_post_call_success_hook."""
        session_id = SessionExtractor.extract_session_id(data)
        session_id_var.set(session_id)
        request_id = data.get("metadata", {}).get("_openbias_request_id", "")
        if request_id:
            request_id_var.set(request_id)
        llm_end_time = time.time()
        llm_start_time = data.get("metadata", {}).get("_openbias_llm_start_time")

        # Retrieve per-request grouping span created in pre_call.
        # Fall back to creating a new request span if the ContextVar was lost
        # (e.g. across async task boundaries), so phase spans still share a parent.
        request_span = request_span_var.get()
        if request_span is None and self.tracer and request_id:
            request_span = self.tracer.start_request_span(session_id, request_id)
            request_span_var.set(request_span)

        interceptor = await self._get_interceptor()

        try:
            # Log LLM call via OTEL BEFORE interceptor.
            # Guard against duplicate entries: _openbias_traced is set after logging so that
            # _log_success_impl (or any future caller) can detect that this request was already traced.
            already_traced = data.get("metadata", {}).get("_openbias_traced", False)
            if self.tracer and not already_traced:
                response_content = extract_response_content(response) or None

                usage_info = extract_usage_info(response)

                # Record LLM latency as metadata instead of backdating span
                # timestamps, so span start/end follow hook execution order
                # and Langfuse renders pre→llm→post correctly.
                llm_latency_ms = (
                    (llm_end_time - llm_start_time) * 1000
                    if llm_start_time is not None
                    else None
                )
                llm_metadata = {
                    "has_interceptor": interceptor is not None,
                    "hook": "post_call_success",
                }
                if llm_latency_ms is not None:
                    llm_metadata["llm_latency_ms"] = round(llm_latency_ms, 2)
                if llm_start_time is not None:
                    llm_metadata["llm_start_time"] = llm_start_time
                if llm_end_time is not None:
                    llm_metadata["llm_end_time"] = llm_end_time

                self.tracer.log_llm_call(
                    session_id=session_id,
                    model=data.get("model", "unknown"),
                    messages=data.get("messages", []),
                    response_content=response_content,
                    response_model=getattr(response, "model", None),
                    usage=usage_info,
                    metadata=llm_metadata,
                    parent_span=request_span,
                    request_id=request_id,
                    phase_order=2,
                )
                data.setdefault("metadata", {})["_openbias_traced"] = True

            if interceptor:
                # Extract response content for tracing
                response_content_for_trace = extract_response_content(response) or None

                if self.tracer:
                    cm = self.tracer.trace_block(
                        "interceptor_post_call",
                        session_id,
                        attributes={"hook": "post_call_success"},
                        input_data={
                            "response": response_content_for_trace,
                            "messages": data.get("messages", []),
                        },
                        metadata={
                            "model": data.get("model", "unknown"),
                        },
                        parent_span=request_span,
                        request_id=request_id,
                        phase_order=3,
                    )
                else:
                    cm = nullcontext()

                with cm as span:
                    request_id = (
                        data.get("metadata", {}).get("_openbias_request_id")
                        or str(uuid.uuid4())
                    )

                    span_factory = EvaluatorSpanFactory(self.tracer, span, session_id) if self.tracer and span else None

                    result = await interceptor.run_post_call(
                        session_id=session_id,
                        request_data=data,
                        response_data=response,
                        user_request_id=request_id,
                        parent_span=request_span,
                        span_factory=span_factory,
                        async_span_factory=(
                            AsyncEvaluatorExecutionSpanFactory(self.tracer, session_id)
                            if self.tracer
                            else None
                        ),
                    )

                    # Set output on span
                    if span is not None:
                        has_modifications = (
                            result.modified_data is not None
                            or result.pending_intervention is not None
                        )
                        output_data = {
                            "allowed": result.allowed,
                            "has_modifications": has_modifications,
                        }
                        output_json = json.dumps(output_data, default=str)
                        span.set_attribute("output.value", output_json)
                        span.set_attribute("langfuse.span.output", output_json)
                        span.set_attribute("openbias.decision", "allow" if result.allowed else "block")
                        span.set_attribute("openbias.has_modifications", has_modifications)

                    # Handle sync POST_CALL results INSIDE trace block for proper nesting
                    if not result.allowed:
                        message = result.user_message or "Response blocked by policy"
                        logger.warning(
                            f"Response blocked for session {session_id}: {message}"
                        )
                        self._record_jsonl_trace(
                            session_id=session_id,
                            request_id=request_id,
                            data=data,
                            response_data={"content": message},
                            final_action="block",
                            internal_metadata=result.internal_metadata,
                            phase="post_call",
                            source={"hook": "async_post_call_success_hook", "blocked": True},
                        )
                        raise WorkflowViolationError(
                            message, context=result.internal_metadata
                        )

                    replay_response = await self._maybe_execute_sync_post_replay(
                        data=data,
                        result=result,
                        session_id=session_id,
                        request_id=request_id,
                    )
                    if replay_response is not None and self.tracer and result.internal_metadata.get("results"):
                        self.tracer.log_intervention(
                            session_id=session_id,
                            intervention_name="post_call_intervention",
                            context=result.internal_metadata,
                            parent_span=span,
                        )
                    if replay_response is not None:
                        self._record_jsonl_trace(
                            session_id=session_id,
                            request_id=request_id,
                            data=data,
                            response_data=replay_response,
                            final_action="intervene",
                            internal_metadata=result.internal_metadata,
                            phase="post_call",
                            source={"hook": "async_post_call_success_hook", "replayed": True},
                        )
                    if replay_response is not None:
                        return replay_response

                    response = self._apply_request_cleanup(data, response)

                    if result.pending_intervention and self.tracer and result.internal_metadata.get("results"):
                        self.tracer.log_intervention(
                            session_id=session_id,
                            intervention_name="post_call_intervention",
                            context=result.internal_metadata,
                            parent_span=span,
                        )

                    final_action = "intervene" if result.pending_intervention else (
                        "shadow"
                        if any(
                            isinstance(item, dict) and item.get("decision") == "shadow"
                            for item in result.internal_metadata.get("results", [])
                        )
                        else "allow"
                    )
                    self._record_jsonl_trace(
                        session_id=session_id,
                        request_id=request_id,
                        data=data,
                        response_data=response,
                        final_action=final_action,
                        internal_metadata=result.internal_metadata,
                        phase="post_call",
                        source={"hook": "async_post_call_success_hook", "replayed": False},
                    )

            return response
        finally:
            # End request span after all post-call work is done
            if self.tracer and request_span is not None:
                self.tracer.end_request_span(request_span)
            request_span_var.set(None)

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        original_exception: Exception,
        **kwargs: Any,
    ) -> None:
        """Execute AFTER failed LLM call.  Wrapped fail-open."""
        return await safe_hook(
            self._post_call_failure_impl,
            request_data, user_api_key_dict, original_exception,
            timeout=self._effective_hook_timeout,
            fallback=None,
            hook_name="async_post_call_failure_hook",
            fail_open=self.settings.fail_open,
            **kwargs,
        )

    async def _post_call_failure_impl(
        self,
        request_data: dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        original_exception: Exception,
        **kwargs: Any,
    ) -> None:
        """Inner implementation for async_post_call_failure_hook."""
        session_id = SessionExtractor.extract_session_id(request_data)
        session_id_var.set(session_id)

        # Clean up request span on failure
        request_span = request_span_var.get()
        if self.tracer and request_span is not None:
            self.tracer.end_request_span(request_span)
        request_span_var.set(None)

        logger.warning(f"LLM call failed for session {session_id}: {original_exception}")

    # Synchronous hooks (for logging/metrics)

    def log_pre_api_call(self, model: str, messages: list[Any], kwargs: dict[str, Any]) -> None:
        """Log before API call (sync)."""
        logger.debug(f"API call starting: model={model}")

    def log_post_api_call(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Log after API call (sync)."""
        if end_time and start_time:
            duration = (end_time - start_time).total_seconds()
            logger.debug(f"API call completed: duration={duration:.2f}s")
        else:
            logger.debug("API call completed")

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Log successful completion (sync)."""
        pass  # Handled in async hook

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Log failed completion (sync)."""
        logger.error(f"LLM call failed: {response_obj}")

    # Async logging hooks (called by LiteLLM Router in library mode)

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Called AFTER successful LLM response (library/router mode). Wrapped fail-open."""
        return await safe_hook(
            self._log_success_impl,
            kwargs, response_obj, start_time, end_time,
            timeout=self._effective_hook_timeout,
            fallback=None,
            hook_name="async_log_success_event",
            fail_open=self.settings.fail_open,
        )

    async def _log_success_impl(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Inner implementation for async_log_success_event."""
        session_id = SessionExtractor.extract_session_id(kwargs)

        interceptor = await self._get_interceptor()
        logger.info(
            f"async_log_success_event: session={session_id}, "
            f"has_interceptor={interceptor is not None}, "
            f"has_tracer={self.tracer is not None}"
        )

        # NOTE: We skip interceptor evaluation here to avoid TimeoutErrors in the logging worker.
        # The logging worker has a short timeout and policy evaluation can take longer.
        # Evaluation is handled in `async_post_call_success_hook` which runs in the main flow.

        # LLM call tracing is handled in _post_call_success_impl to avoid duplicate trace entries.
        # The _openbias_traced flag is set there after logging; any tracing added here MUST
        # check that flag first to prevent duplicate OTEL spans.
        if kwargs.get("metadata", {}).get("_openbias_traced"):
            return

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Called AFTER failed LLM call (library/router mode). Wrapped fail-open."""
        return await safe_hook(
            self._log_failure_impl,
            kwargs, response_obj, start_time, end_time,
            timeout=self._effective_hook_timeout,
            fallback=None,
            hook_name="async_log_failure_event",
            fail_open=self.settings.fail_open,
        )

    async def _log_failure_impl(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Inner implementation for async_log_failure_event."""
        session_id = SessionExtractor.extract_session_id(kwargs)

        logger.warning(f"LLM call failed for session {session_id}: {response_obj}")
