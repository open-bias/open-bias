"""
Open Sentinel LiteLLM hooks for workflow monitoring and intervention.

This module implements the core hook system that intercepts LLM calls:

1. async_pre_call_hook: Runs BEFORE LLM call
   - Applies pending async checker results from previous call
   - Runs sync PRE_CALL checkers
   - Modifies request if WARN with modified_data

2. async_moderation_hook: Runs IN PARALLEL with LLM call (non-blocking)
   - Legacy hook for backward compatibility
   - Records pending interventions for next call

3. async_post_call_success_hook: Runs AFTER LLM call succeeds
   - Runs sync POST_CALL checkers (can modify response on WARN)
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
import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

from opensentinel.config.settings import SentinelSettings
from opensentinel.core.interceptor import (
    CheckerMode,
    CheckPhase,
    Interceptor,
    PolicyEngineChecker,
)
from opensentinel.core.intervention.strategies import WorkflowViolationError
from opensentinel.policy.protocols import PolicyEngine

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
    **kwargs: Any,
) -> Any:
    """
    Execute a hook with timeout and fail-open semantics.

    If the hook raises ``WorkflowViolationError`` it is intentional (a policy
    block) and is re-raised.  Every other exception — including
    ``asyncio.TimeoutError`` — is caught, logged, counted, and the *fallback*
    value is returned so the agent's request passes through unchanged.

    Args:
        hook_fn:   Async callable to execute.
        timeout:   Maximum seconds before the hook is cancelled.
        fallback:  Value to return on failure/timeout.
        hook_name: Human-readable name for logging and metrics.
    """
    try:
        return await asyncio.wait_for(hook_fn(*args, **kwargs), timeout=timeout)
    except WorkflowViolationError:
        raise  # Intentional policy blocks must propagate
    except asyncio.TimeoutError:
        _fail_open_counter[hook_name] = _fail_open_counter.get(hook_name, 0) + 1
        logger.error(
            f"Open Sentinel hook '{hook_name}' timed out after {timeout}s "
            f"(fail-open, count={_fail_open_counter[hook_name]})"
        )
        return fallback
    except Exception as e:
        _fail_open_counter[hook_name] = _fail_open_counter.get(hook_name, 0) + 1
        logger.error(
            f"Open Sentinel hook '{hook_name}' failed (fail-open, "
            f"count={_fail_open_counter[hook_name]}): {e}"
        )
        return fallback

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

        # OpenTelemetry tracer for Open Sentinel events (lazy initialized)
        self._tracer = None

        logger.info("SentinelCallback initialized")

    async def _get_interceptor(self) -> Interceptor | None:
        """Lazy-load interceptor with configured checkers."""
        if self._interceptor_initialized:
            return self._interceptor

        try:
            policy_engine = await self._get_policy_engine()
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

            # Async POST_CALL checker for response evaluation
            # Results deferred to next PRE_CALL via interceptor
            checkers.append(
                PolicyEngineChecker(
                    engine=policy_engine,
                    phase=CheckPhase.POST_CALL,
                    mode=CheckerMode.ASYNC,
                )
            )

            self._interceptor = Interceptor(checkers)
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

        try:
            from opensentinel.policy.registry import PolicyEngineRegistry

            policy_config = self.settings.get_policy_config()
            engine_type = policy_config.get("type", "nemo")
            engine_config = policy_config.get("config", {})

            # Only initialize if we have configuration
            if engine_type == "fsm" and not engine_config.get("config_path"):
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

            if engine_type == "composite" and not engine_config.get("engines"):
                logger.debug("No engines configured for composite, skipping")
                self._policy_engine_initialized = True
                return None

            logger.info(f"Initializing policy engine: {engine_type}")
            self._policy_engine = await PolicyEngineRegistry.create_and_initialize(
                engine_type, engine_config
            )
            self._policy_engine_initialized = True
            logger.info(f"Policy engine initialized: {self._policy_engine.name}")

        except Exception as e:
            logger.error(f"Failed to initialize policy engine: {e}", exc_info=True)
            self._policy_engine_initialized = True
            self._policy_engine = None

        return self._policy_engine

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

    def _extract_session_id(self, data: dict) -> str:
        """
        Extract session ID from request data.

        HTTP headers are automatically resolved from the LiteLLM data dict
        (``data["proxy_server_request"]["headers"]`` or
        ``data["metadata"]["headers"]``), so callers like OpenClaw that
        send ``x-sentinel-session-id`` as an HTTP header are supported
        without extra wiring.

        Priority:
        1. HTTP header: x-sentinel-session-id / x-session-id
        2. metadata.session_id / metadata.sentinel_session_id
        3. metadata.run_id (LangChain)
        4. user field (OpenAI pattern)
        5. thread_id (OpenAI Assistants)
        6. Random UUID (last resort, logged as warning)
        """
        from opensentinel.proxy.middleware import SessionExtractor

        return SessionExtractor.extract_session_id(data)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
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
        )

    async def _pre_call_impl(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallType,
    ) -> Exception | str | dict | None:
        """Inner implementation for async_pre_call_hook."""
        session_id = self._extract_session_id(data)

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
                from contextlib import nullcontext

                cm = nullcontext()

            with cm as span:
                result = await interceptor.run_pre_call(
                    session_id=session_id,
                    request_data=data,
                    user_request_id=str(id(data)),
                )

                # Set output on span
                if span is not None:
                    import json

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
                raise WorkflowViolationError(
                    message,
                    context={
                        "session_id": session_id,
                    },
                )

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
        if "metadata" not in data:
             data["metadata"] = {}
        data["metadata"]["_opensentinel_llm_start_time"] = time.time()

        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallType,
    ) -> None:
        """Execute IN PARALLEL with LLM call. Currently unused. Wrapped fail-open."""
        return await safe_hook(
            self._moderation_impl,
            data, user_api_key_dict, call_type,
            timeout=self.settings.policy.hook_timeout_seconds,
            fallback=None,
            hook_name="async_moderation_hook",
        )

    async def _moderation_impl(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallType,
    ) -> None:
        """Inner implementation for async_moderation_hook."""
        pass

    async def async_post_call_success_hook(
        self,
        data: dict,
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
        )

    async def _post_call_success_impl(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """Inner implementation for async_post_call_success_hook."""
        session_id = self._extract_session_id(data)
        llm_end_time = time.time()
        llm_start_time = data.get("metadata", {}).get("_opensentinel_llm_start_time")

        interceptor = await self._get_interceptor()
        policy_engine = await self._get_policy_engine()

        # Log LLM call via OTEL BEFORE interceptor
        if self.tracer:
            response_content = None
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "message") and first_choice.message:
                    response_content = first_choice.message.content
                elif hasattr(first_choice, "text"):
                    response_content = first_choice.text

            usage_info = None
            if hasattr(response, "usage") and response.usage:
                usage_info = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }

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
            response_content_for_trace = None
            # ... use the one we just extracted
            response_content_for_trace = response_content

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
                from contextlib import nullcontext

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
                    import json

                    output_data = {
                        "allowed": result.allowed,
                        "has_modifications": result.modified_data is not None,
                    }
                    output_json = json.dumps(output_data, default=str)
                    span.set_attribute("output.value", output_json)
                    span.set_attribute("langfuse.span.output", output_json)

            # With async POST_CALL, results are deferred to next PRE_CALL.
            # run_post_call only starts async checkers; no sync gates here.

        return response

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
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
            **kwargs,
        )

    async def _post_call_failure_impl(
        self,
        request_data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        original_exception: Exception,
        **kwargs: Any,
    ) -> None:
        """Inner implementation for async_post_call_failure_hook."""
        session_id = self._extract_session_id(request_data)

        logger.warning(f"LLM call failed for session {session_id}: {original_exception}")

    # Synchronous hooks (for logging/metrics)

    def log_pre_api_call(self, model: str, messages: list, kwargs: dict) -> None:
        """Log before API call (sync)."""
        logger.debug(f"API call starting: model={model}")

    def log_post_api_call(
        self,
        kwargs: dict,
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
        kwargs: dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Log successful completion (sync)."""
        pass  # Handled in async hook

    def log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Log failed completion (sync)."""
        logger.error(f"LLM call failed: {response_obj}")

    # Async logging hooks (called by LiteLLM Router in library mode)

    async def async_log_success_event(
        self,
        kwargs: dict,
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
        )

    async def _log_success_impl(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Inner implementation for async_log_success_event."""
        session_id = self._extract_session_id(kwargs)

        interceptor = await self._get_interceptor()
        logger.info(
            f"async_log_success_event: session={session_id}, "
            f"has_interceptor={interceptor is not None}, "
            f"has_tracer={self.tracer is not None}"
        )

        # NOTE: We skip interceptor evaluation here to avoid TimeoutErrors in the logging worker.
        # The logging worker has a short timeout and policy evaluation can take longer.
        # Evaluation is handled in `async_post_call_success_hook` which runs in the main flow.

        # Log LLM call via OTEL
        if self.tracer:
            response_content = None
            if hasattr(response_obj, "choices") and response_obj.choices:
                first_choice = response_obj.choices[0]
                if hasattr(first_choice, "message") and first_choice.message:
                    response_content = first_choice.message.content
                elif hasattr(first_choice, "text"):
                    response_content = first_choice.text

            usage_info = None
            if hasattr(response_obj, "usage") and response_obj.usage:
                usage_info = {
                    "prompt_tokens": getattr(response_obj.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(
                        response_obj.usage, "completion_tokens", 0
                    ),
                    "total_tokens": getattr(response_obj.usage, "total_tokens", 0),
                }

            self.tracer.log_llm_call(
                session_id=session_id,
                model=kwargs.get("model", "unknown"),
                messages=kwargs.get("messages", []),
                response_content=response_content,
                response_model=getattr(response_obj, "model", None),
                usage=usage_info,
                metadata={
                    "has_interceptor": interceptor is not None,
                    "hook": "async_log_success",
                },
                start_time=start_time,
                end_time=end_time,
            )

    async def async_log_failure_event(
        self,
        kwargs: dict,
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
        )

    async def _log_failure_impl(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Inner implementation for async_log_failure_event."""
        session_id = self._extract_session_id(kwargs)

        logger.warning(f"LLM call failed for session {session_id}: {response_obj}")
