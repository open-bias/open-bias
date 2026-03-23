"""
Open Bias LiteLLM hooks for workflow monitoring and intervention.

This module implements the core hook system that intercepts LLM calls:

1. async_pre_call_hook: Runs BEFORE LLM call
   - Applies pending async engine results from previous call
   - Runs sync PRE_CALL engines
   - Modifies request if INTERVENE with modified_data

2. async_post_call_success_hook: Runs AFTER LLM call succeeds
   - Runs sync POST_CALL engines (can modify response on INTERVENE)
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
import json
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Literal

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

from openbias.config.settings import Settings
from openbias.core.utils import extract_response_content, extract_usage_info
from openbias.core.interceptor import Interceptor
from openbias.core.intervention.strategies import WorkflowViolationError
from openbias.policy.protocols import PolicyEngine
from openbias.proxy.middleware import SessionExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fail-open infrastructure
# ---------------------------------------------------------------------------

# Module-level counter tracking fail-open activations per hook.
# Useful for monitoring/alerting without requiring an external metrics library.
_fail_open_counter: dict[str, int] = {}


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




class Callback(CustomLogger):
    """
    Main Open Bias callback for LiteLLM.

    Implements policy enforcement through LiteLLM's hook system.
    This is registered as a callback when the proxy starts.

    Uses the Interceptor to orchestrate checkers:
    - Sync PRE_CALL checkers run before LLM call
    - Sync POST_CALL checkers run after LLM call (can modify response)
    - Async checkers run in background, results applied next request

    The callback maintains:
    - Interceptor instance with configured checkers
    - OpenTelemetry tracer for observability

    Thread-safety is ensured through asyncio locks for session state.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

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
                engine = await PolicyEngineRegistry.create_and_initialize(
                    ev_config.type, ev_config.config
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
            max_intervention_attempts=self.settings.max_intervention_attempts,
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

        # Persist session ID in metadata to ensure consistency across hooks
        # This prevents generating a new random UUID in post_call/failure hooks
        if "metadata" not in data:
            data["metadata"] = {}

        # Use existing if present, otherwise set the extracted/generated one
        if not data["metadata"].get("session_id"):
            data["metadata"]["session_id"] = session_id


        logger.debug(f"pre_call_hook: session={session_id}, call_type={call_type}")

        interceptor = await self._get_interceptor()
        if interceptor:
            # Wrap in a trace block
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
                )
            else:
                cm = nullcontext()

            with cm as span:
                request_id = str(uuid.uuid4())
                if "metadata" not in data:
                    data["metadata"] = {}
                data["metadata"]["_openbias_request_id"] = request_id

                result = await interceptor.run_pre_call(
                    session_id=session_id,
                    request_data=data,
                    user_request_id=request_id,
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

            # Handle result
            if not result.allowed:
                message = result.message or "Request blocked by policy"
                logger.warning(
                    f"Request blocked for session {session_id}: {message}"
                )
                # LiteLLM convention: returning an Exception instance (not raising)
                # signals the proxy to reject the request with this error message.
                return Exception(message)

            # Apply modifications if any
            if result.modified_data:
                data = result.modified_data

                # Log intervention via OTEL
                if self.tracer:
                    self.tracer.log_intervention(
                        session_id=session_id,
                        intervention_name="pre_call_intervention",
                        context=result.metadata,
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
        llm_end_time = time.time()
        llm_start_time = data.get("metadata", {}).get("_openbias_llm_start_time")

        interceptor = await self._get_interceptor()

        # Log LLM call via OTEL BEFORE interceptor.
        # Guard against duplicate entries: _openbias_traced is set after logging so that
        # _log_success_impl (or any future caller) can detect that this request was already traced.
        already_traced = data.get("metadata", {}).get("_openbias_traced", False)
        if self.tracer and not already_traced:
            response_content = extract_response_content(response) or None

            usage_info = extract_usage_info(response)

            self.tracer.log_llm_call(
                session_id=session_id,
                model=data.get("model", "unknown"),
                messages=data.get("messages", []),
                response_content=response_content,
                response_model=getattr(response, "model", None),
                usage=usage_info,
                metadata={
                    "has_interceptor": interceptor is not None,
                    "hook": "post_call_success",
                },
                start_time=llm_start_time,
                end_time=llm_end_time,
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
                )
            else:
                cm = nullcontext()

            with cm as span:
                request_id = (
                    data.get("metadata", {}).get("_openbias_request_id")
                    or str(uuid.uuid4())
                )

                result = await interceptor.run_post_call(
                    session_id=session_id,
                    request_data=data,
                    response_data=response,
                    user_request_id=request_id,
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

            # Handle sync POST_CALL results
            if not result.allowed:
                message = result.message or "Response blocked by policy"
                logger.warning(
                    f"Response blocked for session {session_id}: {message}"
                )
                raise WorkflowViolationError(
                    message, context=result.metadata
                )

            if result.modified_data and "_interventions" in result.modified_data:
                from openbias.core.intervention.strategies import (
                    ResponseModificationStrategy,
                )

                for intervention in result.modified_data["_interventions"]:
                    response = ResponseModificationStrategy.apply_to_response(
                        response,
                        message=intervention.get("message"),
                        modified_messages=intervention.get("modified_messages"),
                    )
                    logger.info(
                        f"Applied POST_CALL intervention from "
                        f"'{intervention.get('checker')}' for session {session_id}"
                    )

                if self.tracer:
                    self.tracer.log_intervention(
                        session_id=session_id,
                        intervention_name="post_call_intervention",
                        context=result.metadata,
                    )

        return response

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
