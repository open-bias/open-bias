"""
Open Sentinel LiteLLM hooks for workflow monitoring and intervention.

This module implements the core hook system that intercepts LLM calls:

1. async_pre_call_hook: Runs BEFORE LLM call
   - Applies pending async checker results from previous call
   - Runs sync PRE_CALL checkers
   - Modifies request if INTERVENE with modified_data

2. async_post_call_success_hook: Runs AFTER LLM call succeeds
   - Runs sync POST_CALL checkers (can modify response on INTERVENE)
   - Starts async checkers in background
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
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Literal

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

from opensentinel.config.settings import SentinelSettings
from opensentinel.core.utils import extract_response_content, extract_usage_info
from opensentinel.core.interceptor import (
    CheckerMode,
    CheckPhase,
    Interceptor,
    PolicyEngineChecker,
)
from opensentinel.core.intervention.strategies import WorkflowViolationError
from opensentinel.policy.protocols import PolicyEngine
from opensentinel.proxy.middleware import SessionExtractor

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
                f"Open Sentinel hook '{hook_name}' timed out after {timeout}s "
                f"(fail-open, count={_fail_open_counter[hook_name]})"
            )
            return fallback
        logger.error(
            f"Open Sentinel hook '{hook_name}' timed out after {timeout}s (fail-closed)"
        )
        raise
    except Exception as e:
        if fail_open:
            _fail_open_counter[hook_name] = _fail_open_counter.get(hook_name, 0) + 1
            logger.error(
                f"Open Sentinel hook '{hook_name}' failed (fail-open, "
                f"count={_fail_open_counter[hook_name]}): {e}"
            )
            return fallback
        logger.error(
            f"Open Sentinel hook '{hook_name}' failed (fail-closed): {e}"
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




class SentinelCallback(CustomLogger):
    """
    Main Open Sentinel callback for LiteLLM.

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

    def __init__(self, settings: SentinelSettings | None = None):
        self.settings = settings or SentinelSettings()

        # Interceptor (lazy initialized)
        self._interceptor: Interceptor | None = None
        self._interceptor_initialized = False

        # Policy engine (lazy initialized) - kept for direct access if needed
        self._policy_engine: PolicyEngine | None = None
        self._policy_engine_initialized = False

        # Lock for lazy init to prevent concurrent initialization races
        self._init_lock = asyncio.Lock()

        # OpenTelemetry tracer for Open Sentinel events (lazy initialized)
        self._tracer = None

        logger.info("SentinelCallback initialized")

        if self.settings.policy.post_call_mode != "sync":
            logger.warning(
                "POST_CALL mode is '%s' — the first policy violation in each session "
                "will pass through to the user before intervention is applied on the "
                "next request. Set post_call_mode='sync' in config for immediate "
                "enforcement (adds latency).",
                self.settings.policy.post_call_mode,
            )

    async def _get_interceptor(self) -> Interceptor | None:
        """Lazy-load interceptor with configured checkers."""
        if self._interceptor_initialized:
            return self._interceptor

        async with self._init_lock:
            # Double-check after acquiring lock
            if self._interceptor_initialized:
                return self._interceptor

            return await self._initialize_interceptor()

    async def _initialize_interceptor(self) -> Interceptor | None:
        """Actually initialize the interceptor. Must be called under _init_lock."""
        try:
            # Call _initialize_policy_engine directly instead of _get_policy_engine
            # to avoid re-acquiring _init_lock (asyncio.Lock is not reentrant).
            if not self._policy_engine_initialized:
                await self._initialize_policy_engine()
            policy_engine = self._policy_engine
            if not policy_engine:
                self._interceptor_initialized = True
                return None

            # Create checkers from policy engine
            checkers: list[PolicyEngineChecker] = []

            # Sync PRE_CALL checker for request evaluation
            checkers.append(
                PolicyEngineChecker(
                    engine=policy_engine,
                    phase=CheckPhase.PRE_CALL,
                    mode=CheckerMode.SYNC,
                )
            )

            # POST_CALL checker — mode is configurable:
            #   async (default): results deferred to next PRE_CALL, zero latency
            #   sync: blocks response, enables real-time BLOCK/INTERVENE
            post_call_mode = (
                CheckerMode.SYNC
                if self.settings.policy.post_call_mode == "sync"
                else CheckerMode.ASYNC
            )
            checkers.append(
                PolicyEngineChecker(
                    engine=policy_engine,
                    phase=CheckPhase.POST_CALL,
                    mode=post_call_mode,
                )
            )
            logger.info(f"POST_CALL checker mode: {post_call_mode.value}")

            self._interceptor = Interceptor(
                checkers,
                default_strategy=self.settings.policy.default_strategy,
                fail_action=self.settings.policy.fail_action,
            )
            self._interceptor_initialized = True
            logger.info(f"Interceptor initialized with {len(checkers)} checkers")

        except Exception as e:
            logger.error(f"Failed to initialize interceptor: {e}")
            self._interceptor_initialized = True
            self._interceptor = None

        return self._interceptor

    async def _get_policy_engine(self) -> PolicyEngine | None:
        """Lazy-load policy engine based on configuration."""
        if self._policy_engine_initialized:
            return self._policy_engine

        async with self._init_lock:
            # Double-check after acquiring lock
            if self._policy_engine_initialized:
                return self._policy_engine

            return await self._initialize_policy_engine()

    async def _initialize_policy_engine(self) -> PolicyEngine | None:
        """Actually initialize the policy engine. Must be called under _init_lock."""
        try:
            from opensentinel.policy.registry import PolicyEngineRegistry

            policy_config = self.settings.get_policy_config()
            engine_type = policy_config.get("type", "judge")
            engine_config = policy_config.get("config", {})

            # Only initialize if we have configuration
            if engine_type == "fsm" and not engine_config.get("config_path") and not engine_config.get("workflow"):
                logger.debug("No config_path configured, skipping policy engine")
                self._policy_engine_initialized = True
                return None

            if engine_type == "nemo" and not engine_config.get("config_path"):
                logger.debug("No NeMo config_path configured, skipping policy engine")
                self._policy_engine_initialized = True
                return None

            if engine_type == "llm" and not engine_config.get("config_path") and not engine_config.get("workflow"):
                logger.debug("No config_path or workflow for LLM engine, skipping")
                self._policy_engine_initialized = True
                return None

            if engine_type == "judge" and not engine_config.get("models"):
                logger.debug("No judge models configured, skipping policy engine")
                self._policy_engine_initialized = True
                return None

            logger.info(f"Initializing policy engine: {engine_type}")
            self._policy_engine = await PolicyEngineRegistry.create_and_initialize(
                engine_type, engine_config
            )
            # Wire up tracer for engines that support it (e.g. judge engine)
            if hasattr(self._policy_engine, "set_tracer") and self.tracer:
                self._policy_engine.set_tracer(self.tracer)
            self._policy_engine_initialized = True
            logger.info(f"Policy engine initialized: {self._policy_engine.name}")

        except Exception as e:
            logger.error(f"Failed to initialize policy engine: {e}", exc_info=True)
            self._policy_engine_initialized = True
            self._policy_engine = None

        return self._policy_engine

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up all state for a session (interceptor tasks + engine session)."""
        if self._interceptor is not None:
            await self._interceptor.cleanup_session(session_id)
        if self._policy_engine is not None and hasattr(self._policy_engine, "reset_session"):
            await self._policy_engine.reset_session(session_id)
        logger.debug(f"Cleaned up session {session_id} from hooks")

    async def shutdown(self) -> None:
        """
        Shutdown the callback, cleaning up interceptor and policy engine.

        Cancels running async tasks, clears pending session state,
        and shuts down the policy engine.
        """
        logger.info("SentinelCallback shutting down...")

        if self._interceptor is not None:
            try:
                await self._interceptor.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down interceptor: {e}")

        if self._policy_engine is not None:
            try:
                await self._policy_engine.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down policy engine: {e}")

        if self._tracer is not None:
            try:
                self._tracer.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down tracer: {e}")

        logger.info("SentinelCallback shutdown complete")

    @property
    def tracer(self) -> Any:
        """Lazy-load tracer for Open Sentinel event logging via OpenTelemetry."""
        if self._tracer is None:
            logger.debug(f"Tracer check: otel.enabled={self.settings.otel.enabled}")
            if self.settings.otel.enabled:
                from opensentinel.tracing.otel_tracer import SentinelTracer

                self._tracer = SentinelTracer(self.settings.otel)
                logger.info(f"SentinelTracer initialized: {self._tracer}")
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
            timeout=self.settings.policy.hook_timeout_seconds,
            fallback=data,
            hook_name="async_pre_call_hook",
            fail_open=self.settings.policy.fail_open,
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
                result = await interceptor.run_pre_call(
                    session_id=session_id,
                    request_data=data,
                    user_request_id=str(id(data)),
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
        data["metadata"]["_opensentinel_llm_start_time"] = time.time()

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
            timeout=self.settings.policy.hook_timeout_seconds,
            fallback=response,
            hook_name="async_post_call_success_hook",
            fail_open=self.settings.policy.fail_open,
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
        llm_start_time = data.get("metadata", {}).get("_opensentinel_llm_start_time")

        interceptor = await self._get_interceptor()

        # Log LLM call via OTEL BEFORE interceptor
        if self.tracer:
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
                result = await interceptor.run_post_call(
                    session_id=session_id,
                    request_data=data,
                    response_data=response,
                    user_request_id=str(id(data)),
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
                from opensentinel.core.intervention.strategies import (
                    ResponseModificationStrategy,
                )

                # IMPORTANT: LiteLLM ignores the return value of
                # async_post_call_success_hook. POST_CALL interventions work
                # only because apply_to_response mutates the ModelResponse
                # object in-place. We still reassign `response` for clarity,
                # but the in-place mutation is what actually reaches the client.
                for intervention in result.modified_data["_interventions"]:
                    content_before = extract_response_content(response)
                    response = ResponseModificationStrategy.apply_to_response(
                        response,
                        message=intervention.get("message"),
                        modified_messages=intervention.get("modified_messages"),
                    )
                    content_after = extract_response_content(response)
                    if content_before == content_after:
                        logger.warning(
                            "POST_CALL intervention from "
                            f"'{intervention.get('checker')}' did not mutate "
                            f"response for session {session_id}"
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
            timeout=self.settings.policy.hook_timeout_seconds,
            fallback=None,
            hook_name="async_post_call_failure_hook",
            fail_open=self.settings.policy.fail_open,
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
            timeout=self.settings.policy.hook_timeout_seconds,
            fallback=None,
            hook_name="async_log_success_event",
            fail_open=self.settings.policy.fail_open,
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
            timeout=self.settings.policy.hook_timeout_seconds,
            fallback=None,
            hook_name="async_log_failure_event",
            fail_open=self.settings.policy.fail_open,
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
