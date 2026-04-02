"""
Integration test: verify judge details are traced on evaluator spans
using a real TracerProvider + InMemorySpanExporter.

Unlike the mock-based tests, this proves the OTEL SDK stores judge
attributes/events directly on the evaluator span without extra judge-only spans.
"""

import json
import threading

import pytest
from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

from openbias.config.settings import OTelConfig
from openbias.tracing.otel_tracer import Tracer


class InMemorySpanExporter(SpanExporter):
    """Minimal in-memory exporter for testing."""

    def __init__(self):
        self._spans = []
        self._lock = threading.Lock()

    def export(self, spans):
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self):
        with self._lock:
            return list(self._spans)

    def shutdown(self):
        pass


@pytest.fixture()
def memory_exporter():
    """Create a real TracerProvider backed by an InMemorySpanExporter."""
    exporter = InMemorySpanExporter()
    resource = Resource.create({SERVICE_NAME: "test-openbias"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter, provider
    provider.shutdown()


@pytest.fixture()
def real_tracer(memory_exporter):
    """Build an openbias Tracer wired to the in-memory provider."""
    exporter, provider = memory_exporter

    config = OTelConfig(enabled=True, exporter_type="console")

    # Patch out the real provider/exporter setup so Tracer.__init__
    # doesn't try to connect to a real endpoint.
    with patch("openbias.tracing.otel_tracer.TracerProvider", return_value=provider):
        with patch("openbias.tracing.otel_tracer.BatchSpanProcessor"):
            with patch("openbias.tracing.otel_tracer.ConsoleSpanExporter"):
                tracer = Tracer(config)

    # Override internals so spans go through the real provider.
    tracer._tracer = provider.get_tracer("openbias-test")
    tracer._provider = provider

    return tracer, exporter


def _spans_by_name(exporter):
    """Return a dict mapping span name → span from exported spans."""
    return {s.name: s for s in exporter.get_finished_spans()}


class TestAsyncEvalSpanNesting:
    """Verify judge tracing stays on evaluator spans."""

    def test_judge_eval_nests_under_evaluator(self, real_tracer):
        """
        phase_span → evaluator_span
        judge details are attached to evaluator_span
        """
        tracer, exporter = real_tracer
        session_id = "test-session"

        with tracer.trace_block(
            "interceptor_pre_call", session_id
        ) as phase_span:
            with tracer.trace_block(
                "evaluator:test_judge", session_id, parent_span=phase_span
            ) as evaluator_span:
                tracer.log_judge_evaluation(
                    session_id=session_id,
                    rules_source="rules.md",
                    scope="turn",
                    composite_score=0.85,
                    action="pass",
                    judge_model="test-model",
                    scores=[
                        {
                            "criterion": "harmlessness",
                            "score": 4,
                            "max_score": 5,
                            "normalized": 0.85,
                        }
                    ],
                    parent_span=evaluator_span,
                )

        spans = _spans_by_name(exporter)

        # Phase + evaluator spans must be present
        assert "interceptor_pre_call" in spans
        assert "evaluator:test_judge" in spans

        phase = spans["interceptor_pre_call"]
        evaluator = spans["evaluator:test_judge"]

        # evaluator is child of phase
        assert evaluator.parent is not None
        assert evaluator.parent.span_id == phase.context.span_id

        # judge details are attached directly to evaluator span
        assert evaluator.attributes["openbias.judge.rules_source"] == "rules.md"
        assert evaluator.attributes["openbias.judge.action"] == "pass"
        assert evaluator.attributes["openbias.judge.scope"] == "turn"

    def test_judge_eval_has_expected_attributes(self, real_tracer):
        """Judge evaluation span carries rules-source, action, and score attributes."""
        tracer, exporter = real_tracer
        session_id = "test-session-attrs"

        with tracer.trace_block("phase", session_id) as phase_span:
            tracer.log_judge_evaluation(
                session_id=session_id,
                rules_source="compiled_rules",
                scope="turn",
                composite_score=0.72,
                action="intervene",
                judge_model="judge-v2",
                parent_span=phase_span,
            )

        spans = _spans_by_name(exporter)
        phase = spans["phase"]

        assert phase.attributes["openbias.judge.rules_source"] == "compiled_rules"
        assert phase.attributes["openbias.judge.action"] == "intervene"
        assert phase.attributes["openbias.judge.composite_score"] == 0.72
        assert phase.attributes["openbias.judge.model"] == "judge-v2"
        output_value = json.loads(phase.attributes["output.value"])
        assert output_value["rules_source"] == "compiled_rules"
        assert "rubric" not in phase.attributes["output.value"]

    def test_no_orphan_spans_outside_phase(self, real_tracer):
        """All spans share the same trace ID when properly nested."""
        tracer, exporter = real_tracer
        session_id = "test-session-trace"

        with tracer.trace_block("phase", session_id) as phase_span:
            with tracer.trace_block(
                "evaluator:e", session_id, parent_span=phase_span
            ) as eval_span:
                tracer.log_judge_evaluation(
                    session_id=session_id,
                    rules_source="rules.md",
                    scope="turn",
                    composite_score=1.0,
                    action="pass",
                    judge_model="m",
                    parent_span=eval_span,
                )

        finished = exporter.get_finished_spans()
        trace_ids = {s.context.trace_id for s in finished}
        assert len(trace_ids) == 1, f"Expected 1 trace, got {len(trace_ids)}"

    def test_log_judge_evaluation_without_parent_does_not_create_fallback_span(
        self, real_tracer
    ):
        """Legacy standalone judge_evaluation fallback span is not emitted."""
        tracer, exporter = real_tracer
        session_id = "test-no-fallback"

        tracer.log_judge_evaluation(
            session_id=session_id,
            rules_source="rules.md",
            scope="turn",
            composite_score=0.8,
            action="intervene",
            judge_model="judge-x",
            parent_span=None,
            evaluator_name="judge:safety",
        )

        assert exporter.get_finished_spans() == []


class TestPhaseSpanOrdering:
    """Verify phase spans carry request_id and phase_order attributes for stable UI ordering."""

    def test_phase_spans_have_sequential_order(self, real_tracer):
        """pre_call(1) → llm-call(2) → post_call(3) ordering via phase_order attribute."""
        tracer, exporter = real_tracer
        session_id = "test-ordering"
        request_id = "req-order-1"

        # Simulate the three sync phases with request_id and phase_order
        request_span = tracer.start_request_span(session_id, request_id)

        with tracer.trace_block(
            "interceptor_pre_call",
            session_id,
            parent_span=request_span,
            request_id=request_id,
            phase_order=1,
        ):
            pass

        tracer.log_llm_call(
            session_id=session_id,
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            parent_span=request_span,
            request_id=request_id,
            phase_order=2,
        )

        with tracer.trace_block(
            "interceptor_post_call",
            session_id,
            parent_span=request_span,
            request_id=request_id,
            phase_order=3,
        ):
            pass

        tracer.end_request_span(request_span)

        spans = {s.name: s for s in exporter.get_finished_spans()}

        # All three phase spans must exist
        assert "interceptor_pre_call" in spans
        assert "llm-call" in spans
        assert "interceptor_post_call" in spans

        # Verify phase_order attributes
        assert spans["interceptor_pre_call"].attributes["openbias.phase.order"] == 1
        assert spans["llm-call"].attributes["openbias.phase.order"] == 2
        assert spans["interceptor_post_call"].attributes["openbias.phase.order"] == 3

        # Verify request_id is consistent across all phase spans
        for name in ("interceptor_pre_call", "llm-call", "interceptor_post_call"):
            assert spans[name].attributes["openbias.request_id"] == request_id

    def test_all_phase_spans_share_request_parent(self, real_tracer):
        """All three phase spans must be children of the same request span."""
        tracer, exporter = real_tracer
        session_id = "test-parent"
        request_id = "req-parent-1"

        request_span = tracer.start_request_span(session_id, request_id)

        with tracer.trace_block(
            "interceptor_pre_call", session_id, parent_span=request_span,
            request_id=request_id, phase_order=1,
        ):
            pass

        tracer.log_llm_call(
            session_id=session_id,
            model="m",
            messages=[],
            parent_span=request_span,
            request_id=request_id,
            phase_order=2,
        )

        with tracer.trace_block(
            "interceptor_post_call", session_id, parent_span=request_span,
            request_id=request_id, phase_order=3,
        ):
            pass

        tracer.end_request_span(request_span)

        spans = {s.name: s for s in exporter.get_finished_spans()}
        request = spans["openbias-request"]
        request_span_id = request.context.span_id

        for phase_name in ("interceptor_pre_call", "llm-call", "interceptor_post_call"):
            span = spans[phase_name]
            assert span.parent is not None, f"{phase_name} has no parent"
            assert span.parent.span_id == request_span_id, (
                f"{phase_name} parent mismatch: expected {request_span_id}, "
                f"got {span.parent.span_id}"
            )

    def test_phase_order_absent_when_not_provided(self, real_tracer):
        """trace_block without phase_order should not set the attribute."""
        tracer, exporter = real_tracer
        session_id = "test-no-order"

        with tracer.trace_block("plain_span", session_id):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert "openbias.phase.order" not in spans[0].attributes
