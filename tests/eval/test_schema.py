"""Schema validation tests for the rebuilt eval harness."""

from __future__ import annotations

import pytest

from openbias.eval import EvalCase, EvalLabels, EvalSuite, EvalValidationError


def test_eval_case_accepts_single_turn_response_case():
    case = EvalCase(
        id="single-turn",
        tags=["smoke"],
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        labels=EvalLabels(
            violation=False,
            detection_scope="either",
            detect_at_turn=None,
            repair_expected=None,
            repair_verified_at_turn=None,
        ),
    )

    suite = EvalSuite(name="schema-smoke", cases=[case])
    assert suite.cases[0].id == "single-turn"


def test_violation_case_requires_detect_turn():
    with pytest.raises(EvalValidationError, match="detect_at_turn"):
        EvalLabels(
            violation=True,
            detection_scope="response",
            detect_at_turn=None,
            repair_expected=None,
            repair_verified_at_turn=None,
        )


def test_repair_turn_must_point_to_assistant_turn():
    with pytest.raises(EvalValidationError, match="repair verification requires an assistant response turn"):
        EvalCase(
            id="bad-repair-turn",
            messages=[
                {"role": "user", "content": "unsafe prompt"},
                {"role": "assistant", "content": "unsafe answer"},
                {"role": "user", "content": "follow up"},
            ],
            labels=EvalLabels(
                violation=True,
                detection_scope="response",
                detect_at_turn=0,
                repair_expected=True,
                repair_verified_at_turn=1,
            ),
        )


def test_eval_suite_rejects_duplicate_case_ids():
    case = EvalCase(
        id="dup",
        messages=[{"role": "user", "content": "hello"}],
        labels=EvalLabels(
            violation=False,
            detection_scope="either",
            detect_at_turn=None,
            repair_expected=None,
            repair_verified_at_turn=None,
        ),
    )

    with pytest.raises(EvalValidationError, match="Duplicate eval case id"):
        EvalSuite(name="dups", cases=[case, case])
