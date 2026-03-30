
import time
import pytest
from unittest.mock import MagicMock, patch

from opentelemetry.trace import StatusCode

from openbias.tracing.otel_tracer import Tracer
from openbias.config.settings import OTelConfig

@pytest.fixture
def mock_otel():
    with patch("openbias.tracing.otel_tracer.trace") as mock_trace, \
         patch("openbias.tracing.otel_tracer.TracerProvider") as mock_provider, \
         patch("openbias.tracing.otel_tracer.OTLPSpanExporter") as mock_exporter, \
         patch("openbias.tracing.otel_tracer.BatchSpanProcessor") as mock_processor, \
         patch("openbias.tracing.otel_tracer.Resource") as mock_resource:
        
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        
        yield {
            "trace": mock_trace,
            "provider": mock_provider,
            "exporter": mock_exporter,
            "tracer": mock_tracer,
        }

def test_tracer_initialization(mock_otel):
    config = OTelConfig(
        enabled=True,
        service_name="test-service",
        endpoint="localhost:4317",
        exporter_type="otlp"
    )
    
    tracer = Tracer(config)
    
    assert tracer._enabled is True
    mock_otel["trace"].set_tracer_provider.assert_called_once()
    mock_otel["exporter"].assert_called_with(endpoint="localhost:4317", insecure=True)

def test_tracer_disabled(mock_otel):
    config = OTelConfig(enabled=False)
    
    tracer = Tracer(config)
    
    assert tracer._enabled is False
    mock_otel["trace"].set_tracer_provider.assert_not_called()

def test_log_event(mock_otel):
    config = OTelConfig(enabled=True, exporter_type="otlp")
    tracer = Tracer(config)
    
    # Mock span context manager
    mock_span = MagicMock()
    mock_otel["tracer"].start_as_current_span.return_value.__enter__.return_value = mock_span
    
    tracer.log_event(
        session_id="session-1",
        name="test_event",
        input_data={"k": "v"},
        output_data={"res": "ok"}
    )
    
    mock_otel["tracer"].start_as_current_span.assert_called()
    # Check attributes were set
    # openbias.session_id, openbias.event_type, openbias.input.k, openbias.output.res
    calls = mock_span.set_attribute.call_args_list
    assert any(call.args[0] == "openbias.input.k" for call in calls)
    assert any(call.args[0] == "openbias.output.res" for call in calls)

def test_log_llm_call(mock_otel):
    config = OTelConfig(enabled=True, exporter_type="otlp")
    tracer = Tracer(config)

    mock_span = MagicMock()
    mock_otel["tracer"].start_span.return_value = mock_span

    tracer.log_llm_call(
        session_id="session-1",
        model="gpt-4",
        messages=[{"role": "user", "content": "hi"}],
        response_content="hello",
        usage={"total_tokens": 100}
    )

    # log_llm_call uses start_span (not start_as_current_span)
    # Find the "llm-call" start_span call (there may also be session span calls)
    llm_call = None
    for call in mock_otel["tracer"].start_span.call_args_list:
        if call[0] and call[0][0] == "llm-call":
            llm_call = call
            break
        elif call[1].get("name") == "llm-call":
            llm_call = call
            break
    assert llm_call is not None, "start_span('llm-call', ...) not found"

    # Check that required attributes are present (GenAI semantic conventions)
    attrs = llm_call[1]["attributes"]
    assert attrs["openbias.session_id"] == "session-1"
    assert attrs["gen_ai.request.model"] == "gpt-4"
    assert attrs["gen_ai.response.model"] == "gpt-4"

def test_shutdown(mock_otel):
    config = OTelConfig(enabled=True, exporter_type="otlp")
    tracer = Tracer(config)
    
    tracer.shutdown()
    
    # We can't easily check internal provider shutdown call if we mocked class instantiation returning a mock
    # But checking if method calls proceed without error is good start.
    # Actually mock_provider is the class, mock_provider() is the instance.
    mock_otel["provider"].return_value.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# Session memory management tests
# ---------------------------------------------------------------------------

class TestSessionEviction:
    """Tests for TTL-based and max-cap session eviction."""

    def test_stale_sessions_evicted_by_ttl(self, mock_otel):
        """Sessions older than session_ttl_seconds should be ended and removed."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config, session_ttl_seconds=2)

        # Create session spans
        mock_span_a = MagicMock()
        mock_span_b = MagicMock()
        mock_otel["tracer"].start_span.side_effect = [mock_span_a, mock_span_b]

        tracer._get_or_create_session_span("sess-a")
        tracer._get_or_create_session_span("sess-b")
        assert len(tracer._sessions) == 2

        # Simulate time passing beyond the TTL
        for sid in list(tracer._sessions._timestamps):
            tracer._sessions._timestamps[sid] -= 5  # push 5s into the past

        # Next access should trigger eviction of both stale sessions
        mock_span_c = MagicMock()
        mock_otel["tracer"].start_span.side_effect = [mock_span_c]
        tracer._get_or_create_session_span("sess-c")

        assert "sess-a" not in tracer._sessions
        assert "sess-b" not in tracer._sessions
        assert "sess-c" in tracer._sessions
        # The stale spans should have been ended
        mock_span_a.end.assert_called_once()
        mock_span_b.end.assert_called_once()

    def test_active_session_refreshed_on_access(self, mock_otel):
        """Accessing an existing session should refresh its timestamp so it isn't evicted."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config, session_ttl_seconds=10)

        mock_span = MagicMock()
        mock_otel["tracer"].start_span.return_value = mock_span

        tracer._get_or_create_session_span("sess-1")
        old_ts = tracer._sessions._timestamps["sess-1"]

        # Small sleep to ensure monotonic() advances
        time.sleep(0.01)

        span = tracer._get_or_create_session_span("sess-1")
        assert span is mock_span  # same span returned
        assert tracer._sessions._timestamps["sess-1"] > old_ts

    def test_max_sessions_cap(self, mock_otel):
        """When max_sessions is exceeded, oldest sessions should be evicted."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config, max_sessions=3, session_ttl_seconds=9999)

        spans = [MagicMock() for _ in range(5)]
        mock_otel["tracer"].start_span.side_effect = spans

        for i in range(5):
            tracer._get_or_create_session_span(f"sess-{i}")

        # Only the last 3 should remain
        assert len(tracer._sessions) == 3
        assert "sess-0" not in tracer._sessions
        assert "sess-1" not in tracer._sessions
        assert "sess-2" in tracer._sessions
        assert "sess-3" in tracer._sessions
        assert "sess-4" in tracer._sessions
        # The evicted spans should have been ended
        spans[0].end.assert_called_once()
        spans[1].end.assert_called_once()

    def test_end_trace_cleans_up_timestamps(self, mock_otel):
        """end_trace should remove the session from both tracking dicts."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config)

        mock_span = MagicMock()
        mock_otel["tracer"].start_span.return_value = mock_span

        tracer._get_or_create_session_span("sess-1")
        assert "sess-1" in tracer._sessions

        tracer.end_trace("sess-1")
        assert "sess-1" not in tracer._sessions
        mock_span.end.assert_called_once()

    def test_default_ttl_and_max_sessions(self, mock_otel):
        """Verify default values are applied when not explicitly provided."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config)

        assert tracer._sessions._ttl == Tracer.DEFAULT_SESSION_TTL
        assert tracer._sessions._max_sessions == Tracer.DEFAULT_MAX_SESSIONS

    def test_custom_ttl_zero_allowed(self, mock_otel):
        """A TTL of 0 should be allowed (immediate eviction of all prior sessions)."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config, session_ttl_seconds=0)

        assert tracer._sessions._ttl == 0

    def test_shutdown_cleans_all_sessions(self, mock_otel):
        """shutdown() should end all remaining sessions and clear tracking."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config)

        spans = [MagicMock() for _ in range(3)]
        mock_otel["tracer"].start_span.side_effect = spans

        for i in range(3):
            tracer._get_or_create_session_span(f"sess-{i}")

        tracer.shutdown()
        assert len(tracer._sessions) == 0
        for s in spans:
            s.end.assert_called_once()


# ---------------------------------------------------------------------------
# Content redaction tests
# ---------------------------------------------------------------------------

class TestContentRedaction:
    """Tests for the redact_content flag on OTelConfig."""

    def test_log_llm_call_default_includes_content(self, mock_otel):
        """When redact_content is False (default), full content is in span attributes."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config)

        mock_span = MagicMock()
        mock_otel["tracer"].start_span.return_value = mock_span

        tracer.log_llm_call(
            session_id="s1",
            model="gpt-4",
            messages=[{"role": "user", "content": "secret data"}],
            response_content="secret response",
        )

        set_calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert "secret data" in set_calls.get("gen_ai.content.prompt", "")
        assert set_calls.get("gen_ai.content.completion") == "secret response"

    def test_log_llm_call_redacted(self, mock_otel):
        """When redact_content is True, content is replaced with [REDACTED]."""
        config = OTelConfig(enabled=True, exporter_type="otlp", redact_content=True)
        tracer = Tracer(config)

        mock_span = MagicMock()
        mock_otel["tracer"].start_span.return_value = mock_span

        tracer.log_llm_call(
            session_id="s1",
            model="gpt-4",
            messages=[{"role": "user", "content": "secret data"}],
            response_content="secret response",
        )

        set_calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert set_calls.get("gen_ai.content.prompt") == "[REDACTED]"
        assert set_calls.get("gen_ai.content.completion") == "[REDACTED]"

    def test_trace_block_default_includes_input(self, mock_otel):
        """When redact_content is False, trace_block includes full input_data."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config)

        mock_span = MagicMock()
        mock_otel["tracer"].start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_otel["tracer"].start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with tracer.trace_block("test", "s1", input_data={"msg": "hello"}):
            pass

        set_calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert "hello" in set_calls.get("input.value", "")
        assert "hello" in set_calls.get("langfuse.span.input", "")

    def test_trace_block_redacted(self, mock_otel):
        """When redact_content is True, trace_block replaces input with [REDACTED]."""
        config = OTelConfig(enabled=True, exporter_type="otlp", redact_content=True)
        tracer = Tracer(config)

        mock_span = MagicMock()
        mock_otel["tracer"].start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_otel["tracer"].start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with tracer.trace_block("test", "s1", input_data={"msg": "hello"}):
            pass

        set_calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert set_calls.get("input.value") == "[REDACTED]"
        assert set_calls.get("langfuse.span.input") == "[REDACTED]"


# ---------------------------------------------------------------------------
# Span hierarchy tests
# ---------------------------------------------------------------------------

class TestSpanHierarchy:
    """Tests for request-level span hierarchy and explicit parent_span support."""

    def test_start_request_span_creates_child_of_session(self, mock_otel):
        """start_request_span should create an 'openbias-request' span as a child of the session span."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config)

        # Set up a session span
        mock_session_span = MagicMock()
        mock_otel["tracer"].start_span.return_value = mock_session_span
        tracer._get_or_create_session_span("sess-1")

        # Now set up the request span return
        mock_request_span = MagicMock()
        mock_otel["tracer"].start_span.return_value = mock_request_span

        mock_ctx = MagicMock()
        mock_otel["trace"].set_span_in_context.return_value = mock_ctx

        result = tracer.start_request_span("sess-1", "req-1")

        # Verify set_span_in_context was called with the session span
        mock_otel["trace"].set_span_in_context.assert_called_with(mock_session_span)

        # Find the "openbias-request" start_span call
        request_call = None
        for call in mock_otel["tracer"].start_span.call_args_list:
            args = call[0] if call[0] else ()
            kwargs = call[1] if call[1] else {}
            name = args[0] if args else kwargs.get("name")
            if name == "openbias-request":
                request_call = call
                break
        assert request_call is not None, "start_span('openbias-request', ...) not found"

        # Verify attributes include session_id and request_id
        call_kwargs = request_call[1] if request_call[1] else {}
        attrs = call_kwargs.get("attributes", {})
        assert attrs.get("openbias.session_id") == "sess-1"
        assert attrs.get("openbias.request_id") == "req-1"

        # Verify the context passed uses the session span context
        assert call_kwargs.get("context") is mock_ctx

        assert result is mock_request_span

    def test_start_request_span_disabled(self, mock_otel):
        """start_request_span should return None when the tracer is disabled."""
        config = OTelConfig(enabled=False)
        tracer = Tracer(config)

        result = tracer.start_request_span("sess-1", "req-1")
        assert result is None

    def test_end_request_span(self, mock_otel):
        """end_request_span should set OK status and end the span."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config)

        mock_span = MagicMock()
        tracer.end_request_span(mock_span)

        # Verify set_status was called with OK status
        mock_span.set_status.assert_called_once()
        status_arg = mock_span.set_status.call_args[0][0]
        assert status_arg.status_code == StatusCode.OK

        # Verify end() was called
        mock_span.end.assert_called_once()

    def test_end_request_span_none(self, mock_otel):
        """end_request_span(None) should not raise."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config)

        # Should not raise any exception
        tracer.end_request_span(None)

    def test_trace_block_with_explicit_parent_span(self, mock_otel):
        """trace_block with parent_span should use it instead of _resolve_parent_context."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config)

        mock_parent = MagicMock()
        mock_span = MagicMock()
        mock_ctx = MagicMock()

        mock_otel["trace"].set_span_in_context.return_value = mock_ctx
        mock_otel["tracer"].start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_otel["tracer"].start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with tracer.trace_block("test-span", "sess-1", parent_span=mock_parent):
            pass

        # Verify set_span_in_context was called with the explicit parent span
        mock_otel["trace"].set_span_in_context.assert_called_with(mock_parent)

        # Verify start_as_current_span used that context
        mock_otel["tracer"].start_as_current_span.assert_called_once()
        call_kwargs = mock_otel["tracer"].start_as_current_span.call_args[1]
        assert call_kwargs.get("context") is mock_ctx

    def test_log_intervention_with_parent_span(self, mock_otel):
        """log_intervention with parent_span should forward it to log_event."""
        config = OTelConfig(enabled=True, exporter_type="otlp")
        tracer = Tracer(config)

        mock_parent = MagicMock()
        mock_span = MagicMock()
        mock_ctx = MagicMock()

        mock_otel["trace"].set_span_in_context.return_value = mock_ctx
        mock_otel["tracer"].start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_otel["tracer"].start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=False
        )

        tracer.log_intervention("sess-1", "test-intervention", parent_span=mock_parent)

        # Verify set_span_in_context was called with the explicit parent span
        mock_otel["trace"].set_span_in_context.assert_called_with(mock_parent)
