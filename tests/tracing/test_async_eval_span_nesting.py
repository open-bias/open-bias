"""
Integration test: verify async eval spans nest correctly under the
evaluator span using a real TracerProvider + InMemorySpanExporter.

Unlike the mock-based tests, this proves that the OTEL SDK actually
records the parent-child chain: phase → evaluator → judge_evaluation_turn.
"""

import pytest
from unittest.mock import patch, MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory import InMemorySpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

from openbias.config.settings import OTelConfig
from openbias.tracing.otel_tracer import Tracer


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
    """Verify that judge_evaluation_turn nests under evaluator under phase."""

    def test_judge_eval_nests_under_evaluator(self, real_tracer):
        """
        phase_span → evaluator_span → judge_evaluation_turn

        The parent_span_id chain must hold in the real OTEL SDK.
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
                    rubric_name="safety",
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

        # All three spans must be present
        assert "interceptor_pre_call" in spans
        assert "evaluator:test_judge" in spans
        assert "judge_evaluation_turn" in spans

        phase = spans["interceptor_pre_call"]
        evaluator = spans["evaluator:test_judge"]
        judge = spans["judge_evaluation_turn"]

        # evaluator is child of phase
        assert evaluator.parent is not None
        assert evaluator.parent.span_id == phase.context.span_id

        # judge_evaluation_turn is child of evaluator
        assert judge.parent is not None
        assert judge.parent.span_id == evaluator.context.span_id

    def test_judge_eval_has_expected_attributes(self, real_tracer):
        """Judge evaluation span carries rubric, action, and score attributes."""
        tracer, exporter = real_tracer
        session_id = "test-session-attrs"

        with tracer.trace_block("phase", session_id) as phase_span:
            tracer.log_judge_evaluation(
                session_id=session_id,
                rubric_name="fairness",
                scope="turn",
                composite_score=0.72,
                action="intervene",
                judge_model="judge-v2",
                parent_span=phase_span,
            )

        spans = _spans_by_name(exporter)
        judge = spans["judge_evaluation_turn"]

        assert judge.attributes["openbias.judge.rubric"] == "fairness"
        assert judge.attributes["openbias.judge.action"] == "intervene"
        assert judge.attributes["openbias.judge.composite_score"] == 0.72
        assert judge.attributes["openbias.judge.model"] == "judge-v2"

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
                    rubric_name="r",
                    scope="turn",
                    composite_score=1.0,
                    action="pass",
                    judge_model="m",
                    parent_span=eval_span,
                )

        finished = exporter.get_finished_spans()
        trace_ids = {s.context.trace_id for s in finished}
        assert len(trace_ids) == 1, f"Expected 1 trace, got {len(trace_ids)}"
