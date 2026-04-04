from pathlib import Path
from unittest.mock import patch

from openbias.config.settings import OTelConfig
from openbias.tracing.otel_tracer import Tracer


def test_jsonl_trace_sink_appends_replayable_case(tmp_path: Path):
    tracer = Tracer(
        OTelConfig(exporter_type="jsonl", path=str(tmp_path / "traces" / "%Y-%m-%d.jsonl"))
    )

    output_path = tracer.record_trace_case(
        case_id="req-1",
        session_id="sess-1",
        request_messages=[{"role": "user", "content": "Refund me"}],
        response_data={"content": "Please verify your identity first."},
        metadata={
            "model": "gpt-4o-mini",
            "timestamp": "2026-04-05T00:00:00Z",
            "policy_hash": "abc123",
            "request_id": "req-1",
            "evaluator_names": ("workflow-guard",),
        },
        final_action="intervene",
        evaluator_summaries=[
            {
                "name": "workflow-guard",
                "engine_type": "fsm",
                "phase": "post_call",
                "decision": "intervene",
                "violations": [{"reason": "Verify identity before refund", "severity": "error"}],
            }
        ],
        interventions=[
            {
                "name": "merged",
                "strategy": "user_message_inject",
                "applied_at": "next_turn",
                "message": "Please verify identity first.",
            }
        ],
    )

    assert output_path is not None
    dataset_path = next((tmp_path / "traces").glob("*.jsonl"))
    contents = dataset_path.read_text(encoding="utf-8")
    assert '"id": "req-1"' in contents
    assert '"final_action": "intervene"' in contents


def test_jsonl_trace_sink_fail_open_on_write_error(tmp_path: Path):
    tracer = Tracer(
        OTelConfig(exporter_type="jsonl", path=str(tmp_path / "traces" / "%Y-%m-%d.jsonl"))
    )

    with patch.object(tracer._jsonl_sink, "append", side_effect=OSError("disk full")):
        output_path = tracer.record_trace_case(
            case_id="req-1",
            session_id="sess-1",
            request_messages=[{"role": "user", "content": "Hello"}],
            response_data={"content": "Hi"},
        )

    assert output_path is None
