"""
OpenTelemetry-based tracer for Open Bias events.

Uses the OpenTelemetry SDK for vendor-agnostic distributed tracing.
Traces can be exported to any OTLP-compatible backend including:
- Jaeger, Zipkin, or other OTLP backends
- Langfuse (via their OTLP endpoint)
"""

import base64
import json
import logging
from typing import Any
from contextlib import contextmanager

from openbias import __version__
from openbias.core.session import SessionStore
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPSpanExporterHTTP
from opentelemetry.trace import Link, SpanContext, Status, StatusCode, TraceFlags, TraceState

from openbias.config.settings import OTelConfig

logger = logging.getLogger(__name__)

class SpanEventManager(logging.Handler):
    """
    Logging handler that attaches log records as events to the current OTEL span.
    """
    def emit(self, record):
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                attributes = {
                    "log.level": record.levelname,
                    "log.logger": record.name,
                    "code.filepath": record.pathname,
                    "code.lineno": record.lineno,
                }
                session_id = getattr(record, "session_id", "")
                if session_id:
                    attributes["session.id"] = session_id
                request_id = getattr(record, "request_id", "")
                if request_id:
                    attributes["request.id"] = request_id
                current_span.add_event(record.getMessage(), attributes=attributes)
                
                # Debug print for verification
                # print(f"DEBUG: Captured log as event: {record.getMessage()}")  
        except Exception:
            self.handleError(record)

class Tracer:
    """
    OpenTelemetry-based tracer for Open Bias workflow events.

    This tracer uses the OpenTelemetry SDK for distributed tracing,
    allowing traces to be exported to any OTLP-compatible backend
    including Langfuse.
    """

    # Defaults for session memory management
    DEFAULT_SESSION_TTL = 3600      # 1 hour
    DEFAULT_MAX_SESSIONS = 10_000   # hard cap

    def __init__(
        self,
        config: OTelConfig,
        session_ttl_seconds: int | None = None,
        max_sessions: int | None = None,
    ):
        self.config = config
        # Session memory management
        self._sessions: SessionStore[trace.Span] = SessionStore(
            ttl=session_ttl_seconds if session_ttl_seconds is not None else self.DEFAULT_SESSION_TTL,
            max_sessions=max_sessions if max_sessions is not None else self.DEFAULT_MAX_SESSIONS,
            on_evict=self._on_session_evict,
        )
        self._enabled = config.enabled

        if not self._enabled:
            self._tracer = None
            logger.info("Tracer disabled")
            return

        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: "openbias"})

        # Create and set tracer provider
        provider = TracerProvider(resource=resource)

        # Configure exporter based on type
        resolved_type = config.resolved_exporter_type
        if resolved_type == "console":
            exporter = ConsoleSpanExporter()
            logger.info("Tracer using console exporter")
        elif resolved_type == "langfuse":
            # Langfuse OTLP endpoint with HTTP and Basic Auth
            if not config.langfuse_public_key or not config.langfuse_secret_key:
                logger.warning("Tracer disabled: missing Langfuse credentials")
                self._enabled = False
                self._tracer = None
                return
            
            # Build Langfuse OTLP endpoint
            langfuse_host = config.langfuse_host.rstrip("/")
            langfuse_endpoint = f"{langfuse_host}/api/public/otel/v1/traces"
            
            # Create Basic Auth header
            auth_str = f"{config.langfuse_public_key}:{config.langfuse_secret_key}"
            auth_bytes = base64.b64encode(auth_str.encode()).decode()
            headers = {"Authorization": f"Basic {auth_bytes}"}
            
            exporter = OTLPSpanExporterHTTP(
                endpoint=langfuse_endpoint,
                headers=headers,
            )
            logger.info(f"Tracer using Langfuse OTLP exporter (host={langfuse_host})")
        else:  # otlp (gRPC)
            exporter = OTLPSpanExporter(
                endpoint=config.endpoint,
                insecure=config.insecure,
            )
            logger.info(f"Tracer using OTLP gRPC exporter (endpoint={config.endpoint})")

        # Add batch processor for efficient export
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set as global provider
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer("openbias", "0.1.0")
        self._provider = provider

        # Attach span event manager to capture NeMo Guardrails logs
        span_handler = SpanEventManager()
        # Ensure we don't duplicate handlers
        nemo_logger = logging.getLogger("nemoguardrails")
        if not any(isinstance(h, SpanEventManager) for h in nemo_logger.handlers):
            nemo_logger.addHandler(span_handler)

        logger.info(f"Tracer initialized (exporter={config.exporter_type})")

    @staticmethod
    def _on_session_evict(session_id: str, span: trace.Span) -> None:
        """End the span when a session is evicted."""
        try:
            span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception:
            logger.debug("Failed to end span for session %s", session_id, exc_info=True)

    @staticmethod
    def _end_span(span: trace.Span, session_id: str) -> None:
        """Gracefully end a session span."""
        try:
            span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception:
            logger.debug("Failed to end span for session %s", session_id, exc_info=True)

    def _get_or_create_session_span(self, session_id: str) -> trace.Span:
        """Get existing session span or create new one.

        Triggers lazy eviction of stale / overflow sessions before returning.
        """
        self._sessions.evict_stale()

        span = self._sessions.get(session_id)
        if span is not None:
            self._sessions.touch(session_id)
            return span

        if self._tracer:
            span = self._tracer.start_span(
                "openbias-session",
                attributes={
                    "openbias.session_id": session_id,
                    "openbias.version": __version__,
                    "langfuse.trace.name": "openbias-session",
                    "langfuse.session.id": session_id,
                },
            )
            self._sessions.put(session_id, span)
            logger.debug(f"Created session span for {session_id}")

        return self._sessions.get(session_id)

    def _resolve_parent_context(self, session_id: str) -> trace.Context | None:
        """Resolve the parent context for a new span.

        If there is an active recording span in the current context (e.g. from
        a ``trace_block``), use it as the parent so that sub-spans nest
        correctly.  Otherwise, fall back to the session-level root span.
        """
        current = trace.get_current_span()
        if current and current.is_recording():
            return None  # OTEL will automatically use the current span as parent

        parent_span = self._get_or_create_session_span(session_id)
        return trace.set_span_in_context(parent_span) if parent_span else None

    def start_child_span(
        self,
        name: str,
        parent_span: trace.Span,
        attributes: dict[str, Any] | None = None,
    ) -> trace.Span | None:
        """Start a span as a child of the given parent span.

        Returns the span (not set as current) or ``None`` if tracing is
        disabled.  This is the public API for creating non-context-managed
        spans that need explicit ``end()`` calls.
        """
        if not self._enabled or not self._tracer:
            return None

        parent_ctx = trace.set_span_in_context(parent_span)
        return self._tracer.start_span(
            name,
            context=parent_ctx,
            attributes=attributes or {},
        )

    def start_request_span(self, session_id: str, request_id: str) -> trace.Span | None:
        """Start a request-level span as a child of the session span.

        Returns the span (not set as current) or ``None`` if tracing is disabled.
        """
        if not self._enabled or not self._tracer:
            return None

        session_span = self._get_or_create_session_span(session_id)
        parent_ctx = trace.set_span_in_context(session_span)

        span = self._tracer.start_span(
            "openbias-request",
            context=parent_ctx,
            attributes={
                "openbias.session_id": session_id,
                "openbias.request_id": request_id,
                "langfuse.trace.name": "openbias-session",
                "langfuse.session.id": session_id,
            },
        )
        return span

    def end_request_span(self, span: trace.Span | None) -> None:
        """End a request span previously started with :meth:`start_request_span`."""
        if span is None:
            return
        try:
            span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception:
            logger.debug("Failed to end request span", exc_info=True)

    def _safe_json(self, obj: Any) -> str:
        """Safely serialize object to JSON string for span attributes."""
        try:
            return json.dumps(obj, default=str, ensure_ascii=False)
        except Exception:
            return str(obj)

    @contextmanager
    def trace_block(
        self,
        name: str,
        session_id: str,
        attributes: dict[str, Any] | None = None,
        input_data: Any | None = None,
        metadata: dict[str, Any] | None = None,
        parent_span: Any | None = None,
        links: list[Link] | None = None,
        request_id: str | None = None,
        phase_order: int | None = None,
    ):
        """
        Context manager to trace a block of code.
        Useful for capturing logs emitted during execution as span events.

        Args:
            name: Span name
            session_id: Session identifier
            attributes: Additional span attributes
            input_data: Input data to record (will be JSON serialized)
            metadata: Additional metadata for the span
            parent_span: Optional explicit parent span to nest under
            links: Optional OTEL links to correlate async work
            request_id: Optional request ID to attach for cross-span correlation
            phase_order: Optional integer for deterministic phase ordering in UIs
        """
        if not self._enabled or not self._tracer:
            yield None
            return

        if parent_span is not None:
            parent_ctx = trace.set_span_in_context(parent_span)
        else:
            parent_ctx = self._resolve_parent_context(session_id)

        span_attrs = {
            "openbias.session_id": session_id,
            **(attributes or {}),
        }
        if request_id is not None:
            span_attrs["openbias.request_id"] = request_id
        if phase_order is not None:
            span_attrs["openbias.phase.order"] = phase_order

        with self._tracer.start_as_current_span(
            name,
            context=parent_ctx,
            attributes=span_attrs,
            links=links,
        ) as span:
            # Set input data using Langfuse-compatible attributes
            if input_data is not None:
                input_json = self._safe_json(input_data)
                span.set_attribute("input.value", input_json)
                span.set_attribute("langfuse.span.input", input_json)
            
            # Set metadata if provided
            if metadata:
                span.set_attribute("langfuse.span.metadata", self._safe_json(metadata))
                for key, value in metadata.items():
                    span.set_attribute(f"openbias.metadata.{key}", str(value))
            
            yield span

    def build_span_link(self, trace_id_hex: str | None, span_id_hex: str | None) -> Link | None:
        """Build an OTEL Link from serialized trace/span IDs."""
        if not trace_id_hex or not span_id_hex:
            return None
        try:
            span_context = SpanContext(
                trace_id=int(trace_id_hex, 16),
                span_id=int(span_id_hex, 16),
                is_remote=True,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
                trace_state=TraceState(),
            )
            if not span_context.is_valid:
                return None
            return Link(span_context)
        except Exception:
            logger.debug("Failed to build OTEL link from serialized context", exc_info=True)
            return None

    def log_event(
        self,
        session_id: str,
        name: str,
        metadata: dict[str, Any] | None = None,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        parent_span: Any | None = None,
    ) -> None:
        """
        Log an Open Bias event as an OTEL span.

        Use for workflow deviations, interventions, state transitions.
        """
        if not self._enabled or not self._tracer:
            return

        if parent_span is not None:
            parent_ctx = trace.set_span_in_context(parent_span)
        else:
            parent_ctx = self._resolve_parent_context(session_id)

        with self._tracer.start_as_current_span(
            name,
            context=parent_ctx,
            attributes={
                "openbias.session_id": session_id,
                "openbias.event_type": name,
            },
        ) as span:
            # Add input data as attributes
            if input_data:
                for key, value in input_data.items():
                    span.set_attribute(f"openbias.input.{key}", str(value))

            # Add output data as attributes
            if output_data:
                for key, value in output_data.items():
                    span.set_attribute(f"openbias.output.{key}", str(value))

            # Add metadata as attributes
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"openbias.metadata.{key}", str(value))

            logger.info(f"Logged event '{name}' for session {session_id}")

    def log_intervention(
        self,
        session_id: str,
        intervention_name: str,
        context: dict[str, Any] | None = None,
        parent_span: Any | None = None,
    ) -> None:
        """Log an intervention being applied."""
        self.log_event(
            session_id=session_id,
            name="intervention_applied",
            input_data={"intervention": intervention_name},
            metadata={
                "intervention_name": intervention_name,
                **(context or {}),
            },
            parent_span=parent_span,
        )

    def log_llm_call(
        self,
        session_id: str,
        model: str,
        messages: list,
        response_content: str | None = None,
        response_model: str | None = None,
        usage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent_span: Any | None = None,
        request_id: str | None = None,
        phase_order: int | None = None,
    ) -> None:
        """
        Log an LLM call as an OTEL span with GenAI semantic conventions.

        Span start/end use wall-clock time (not backdated) so that
        Langfuse renders pre → llm → post in execution order.
        LLM latency is recorded in metadata by the caller.

        Uses OpenTelemetry GenAI semantic conventions for Langfuse compatibility:
        - gen_ai.* attributes for model/usage info
        - input.value / output.value for content
        - langfuse.span.* for explicit Langfuse mapping
        """
        if not self._enabled or not self._tracer:
            return

        # Use provided parent, active context span, or session span
        if parent_span is not None:
            parent_ctx = trace.set_span_in_context(parent_span)
        else:
            parent_ctx = self._resolve_parent_context(session_id)

        # Build span attributes with GenAI semantic conventions
        span_attrs = {
            "openbias.session_id": session_id,
            # GenAI semantic conventions
            "gen_ai.system": "openai",  # or derive from model
            "gen_ai.request.model": model,
            "gen_ai.response.model": response_model or model,
        }
        if request_id is not None:
            span_attrs["openbias.request_id"] = request_id
        if phase_order is not None:
            span_attrs["openbias.phase.order"] = phase_order

        span = self._tracer.start_span(
            "llm-call",
            context=parent_ctx,
            attributes=span_attrs,
        )

        with trace.use_span(span, end_on_exit=False) as current_span:
            # Set input (messages)
            if messages:
                messages_json = self._safe_json(messages)
                span.set_attribute("gen_ai.content.prompt", messages_json)
                span.set_attribute("input.value", messages_json)
                span.set_attribute("langfuse.span.input", messages_json)

            # Set output (response)
            if response_content:
                span.set_attribute("gen_ai.content.completion", response_content)
                span.set_attribute("output.value", response_content)
                span.set_attribute("langfuse.span.output", response_content)

            # Add usage info with GenAI semantic conventions
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                # GenAI semantic conventions
                span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
                span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)

            # Add metadata
            if metadata:
                span.set_attribute("langfuse.span.metadata", self._safe_json(metadata))
                for key, value in metadata.items():
                    span.set_attribute(f"openbias.metadata.{key}", str(value))

            span.end()
            logger.info(f"Logged LLM call for session {session_id} (model={model})")

    def _annotate_judge_details(
        self,
        span: Any,
        rules_source: str,
        scope: str,
        action: str,
        participating_judges: list[str] | None = None,
        failed_rules: list[str] | None = None,
        rule_results: list[dict[str, Any]] | None = None,
        latency_ms: float | None = None,
        token_usage: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach judge verdict details to an existing span."""
        span.set_attribute("openbias.judge.rules_source", rules_source)
        span.set_attribute("openbias.judge.scope", scope)
        span.set_attribute("openbias.judge.action", action)
        span.set_attribute(
            "openbias.judge.failed_rules_count",
            len(failed_rules or []),
        )
        if participating_judges:
            span.set_attribute(
                "openbias.judge.participating_judges",
                list(participating_judges),
            )
        if failed_rules:
            span.set_attribute("openbias.judge.failed_rules", list(failed_rules))

        if latency_ms is not None:
            span.set_attribute("openbias.judge.latency_ms", latency_ms)
        if token_usage is not None:
            span.set_attribute("openbias.judge.token_usage", token_usage)

        if rule_results:
            span.set_attribute("openbias.judge.rules_count", len(rule_results))
            for rule_result in rule_results:
                if not isinstance(rule_result, dict):
                    continue
                judge_results = rule_result.get("judge_results", [])
                participating_rule_judges: list[str] = []
                failing_judges: list[str] = []
                judge_outcomes: list[dict[str, Any]] = []
                if isinstance(judge_results, list):
                    for judge_result in judge_results:
                        if not isinstance(judge_result, dict):
                            continue
                        judge_name = str(judge_result.get("judge_name", ""))
                        if judge_name:
                            participating_rule_judges.append(judge_name)
                            if not judge_result.get("passed", False):
                                failing_judges.append(judge_name)
                        judge_outcomes.append(
                            {
                                "judge_name": judge_name,
                                "judge_model": judge_result.get("judge_model"),
                                "passed": bool(judge_result.get("passed", False)),
                            }
                        )
                span.add_event(
                    f"judge.rule:{rule_result.get('rule', 'unknown')}",
                    attributes={
                        "passed": bool(rule_result.get("passed", False)),
                        "action": str(rule_result.get("action", "")),
                        "summary": str(rule_result.get("summary", "")),
                        "judge_count": len(judge_results) if isinstance(judge_results, list) else 0,
                        "participating_judges": ", ".join(participating_rule_judges),
                        "failing_judges": ", ".join(failing_judges),
                        "aggregation_mode": str(rule_result.get("aggregation_mode", "")),
                        "judge_outcomes": self._safe_json(judge_outcomes),
                    },
                )

        # Structured output for Langfuse
        output = {
            "action": action,
            "rules_source": rules_source,
            "scope": scope,
            "participating_judges": participating_judges or [],
            "failed_rules": failed_rules or [],
        }
        if rule_results:
            output["rule_results"] = rule_results
        span.set_attribute("output.value", self._safe_json(output))
        span.set_attribute("langfuse.span.output", self._safe_json(output))

        if metadata:
            span.set_attribute("langfuse.span.metadata", self._safe_json(metadata))
            for key, value in metadata.items():
                span.set_attribute(f"openbias.metadata.{key}", str(value))

    def log_judge_evaluation(
        self,
        session_id: str,
        rules_source: str,
        scope: str,
        action: str,
        participating_judges: list[str] | None = None,
        failed_rules: list[str] | None = None,
        rule_results: list[dict[str, Any]] | None = None,
        latency_ms: float | None = None,
        token_usage: int | None = None,
        metadata: dict[str, Any] | None = None,
        parent_span: Any | None = None,
        evaluator_name: str | None = None,
    ) -> None:
        """
        Log judge evaluation details on the active evaluator span.

        Args:
            session_id: Session identifier.
            rules_source: Source label for the compiled rules used for evaluation.
            scope: Evaluation scope label (currently "turn").
            action: Verdict action (pass/intervene/block).
            participating_judges: Judge names that participated in this turn.
            failed_rules: Aggregated rules that failed for this turn.
            rule_results: Per-rule aggregated results including per-judge outcomes.
            latency_ms: Evaluation latency in milliseconds.
            token_usage: Total tokens consumed.
            metadata: Additional metadata.
            parent_span: Optional explicit span to annotate.
            evaluator_name: Evaluator name for metadata consistency.
        """
        if not self._enabled or not self._tracer:
            return

        target_span = None
        if parent_span is not None:
            target_span = parent_span
        else:
            current = trace.get_current_span()
            if current and current.is_recording():
                target_span = current

        if target_span is None:
            logger.warning(
                "Skipping judge trace details for session %s: no active/parent span",
                session_id,
            )
            return

        if evaluator_name:
            target_span.set_attribute("openbias.evaluator.name", evaluator_name)
        self._annotate_judge_details(
            target_span,
            rules_source=rules_source,
            scope=scope,
            action=action,
            participating_judges=participating_judges,
            failed_rules=failed_rules,
            rule_results=rule_results,
            latency_ms=latency_ms,
            token_usage=token_usage,
            metadata=metadata,
        )

        logger.debug(
            f"Logged judge evaluation for session {session_id} "
            f"(rules_source={rules_source}, action={action}, failed_rules={len(failed_rules or [])})"
        )

    def end_trace(self, session_id: str) -> None:
        """Mark a session trace as complete and free the session memory."""
        span = self._sessions.remove(session_id)
        if span is not None:
            self._end_span(span, session_id)
            logger.debug(f"Ended trace for session {session_id}")

    def flush(self) -> None:
        """Force flush any pending spans."""
        if hasattr(self, "_provider") and self._provider:
            self._provider.force_flush()

    def shutdown(self) -> None:
        """Clean up any remaining traces."""
        for session_id in list(self._sessions.keys()):
            self.end_trace(session_id)

        if hasattr(self, "_provider") and self._provider:
            self._provider.shutdown()

        logger.info("Tracer shut down")
