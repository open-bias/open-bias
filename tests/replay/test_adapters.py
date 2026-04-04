from openbias.replay import trace_case_to_eval_case
from openbias.traces import TraceCase


def test_trace_case_to_eval_case_returns_none_without_labels():
    case = TraceCase(
        id="trace-1",
        session_id="sess-1",
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    )

    assert trace_case_to_eval_case(case) is None


def test_trace_case_to_eval_case_converts_eval_labels():
    case = TraceCase(
        id="trace-2",
        session_id="sess-2",
        messages=[
            {"role": "user", "content": "unsafe"},
            {"role": "assistant", "content": "unsafe reply"},
        ],
        labels={
            "violation": True,
            "detection_scope": "response",
            "detect_at_turn": 0,
            "repair_expected": None,
            "repair_verified_at_turn": None,
        },
    )

    eval_case = trace_case_to_eval_case(case)
    assert eval_case is not None
    assert eval_case.labels.violation is True
    assert eval_case.labels.detection_scope == "response"
