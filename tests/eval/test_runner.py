"""Runner tests for the rebuilt eval harness."""

from __future__ import annotations

import pytest

from openbias.eval import (
    EvalCase,
    EvalLabels,
    EvalRunner,
    EvalRuntimeConfig,
    EvalSuite,
    load_native_suite,
)


def _case(
    case_id: str,
    *,
    messages: list[dict[str, str]],
    violation: bool,
    detection_scope: str,
    detect_at_turn: int | None,
    repair_expected: bool | None = None,
    repair_verified_at_turn: int | None = None,
) -> EvalCase:
    return EvalCase(
        id=case_id,
        messages=messages,
        tags=["runner"],
        labels=EvalLabels(
            violation=violation,
            detection_scope=detection_scope,
            detect_at_turn=detect_at_turn,
            repair_expected=repair_expected,
            repair_verified_at_turn=repair_verified_at_turn,
        ),
    )


async def test_runner_classifies_safe_case(keyword_engine):
    suite = EvalSuite(
        name="safe",
        cases=[
            _case(
                "safe-case",
                messages=[{"role": "user", "content": "just a safe request"}],
                violation=False,
                detection_scope="either",
                detect_at_turn=None,
            )
        ],
    )

    result = await EvalRunner().run(keyword_engine, suite)

    assert result.outcomes[0].outcome == "correct_non_violation"
    assert result.summary.true_negative == 1


async def test_runner_classifies_detected_and_missed_and_false_positive_cases(keyword_engine):
    suite = EvalSuite(
        name="detection-matrix",
        cases=[
            _case(
                "detected-violation",
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "unsafe answer"},
                ],
                violation=True,
                detection_scope="response",
                detect_at_turn=0,
            ),
            _case(
                "missed-violation",
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "subtle-risk"},
                ],
                violation=True,
                detection_scope="response",
                detect_at_turn=0,
            ),
            _case(
                "false-positive",
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "false-positive trigger"},
                ],
                violation=False,
                detection_scope="either",
                detect_at_turn=None,
            ),
        ],
    )

    result = await EvalRunner().run(keyword_engine, suite)
    outcomes = {outcome.case_id: outcome.outcome for outcome in result.outcomes}

    assert outcomes == {
        "detected-violation": "detected_violation",
        "missed-violation": "missed_violation",
        "false-positive": "false_positive",
    }
    assert result.summary.true_positive == 1
    assert result.summary.false_negative == 1
    assert result.summary.false_positive == 1


async def test_runner_verifies_detected_and_fixed_case_from_recovery_suite(keyword_engine):
    suite = load_native_suite("tests/eval/fixtures/recovery_suite.yaml")

    result = await EvalRunner().run(keyword_engine, suite)

    assert result.outcomes[0].outcome == "detected_and_fixed"
    assert result.outcomes[0].passed is True
    assert result.summary.fix_success_count == 1
    assert result.summary.fix_failure_count == 0


async def test_runner_classifies_detected_not_fixed_case(keyword_engine):
    suite = EvalSuite(
        name="repair-failure",
        cases=[
            _case(
                "detected-not-fixed",
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "unsafe answer"},
                    {"role": "user", "content": "please try again"},
                    {"role": "assistant", "content": "still unsafe"},
                ],
                violation=True,
                detection_scope="response",
                detect_at_turn=0,
                repair_expected=False,
                repair_verified_at_turn=1,
            )
        ],
    )

    result = await EvalRunner().run(keyword_engine, suite)

    assert result.outcomes[0].outcome == "detected_not_fixed"
    assert result.outcomes[0].passed is True
    assert result.summary.fix_success_count == 0
    assert result.summary.fix_failure_count == 1


async def test_runner_computes_binary_summary_metrics(keyword_engine):
    suite = EvalSuite(
        name="summary",
        cases=[
            _case(
                "safe",
                messages=[{"role": "user", "content": "safe"}],
                violation=False,
                detection_scope="either",
                detect_at_turn=None,
            ),
            _case(
                "detected",
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "unsafe answer"},
                ],
                violation=True,
                detection_scope="response",
                detect_at_turn=0,
            ),
            _case(
                "missed",
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "subtle-risk"},
                ],
                violation=True,
                detection_scope="response",
                detect_at_turn=0,
            ),
            _case(
                "fp",
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "false-positive trigger"},
                ],
                violation=False,
                detection_scope="either",
                detect_at_turn=None,
            ),
            _case(
                "fixed",
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "unsafe answer"},
                    {"role": "user", "content": "try again"},
                    {"role": "assistant", "content": "corrected answer"},
                ],
                violation=True,
                detection_scope="response",
                detect_at_turn=0,
                repair_expected=True,
                repair_verified_at_turn=1,
            ),
            _case(
                "not-fixed",
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "unsafe answer"},
                    {"role": "user", "content": "try again"},
                    {"role": "assistant", "content": "still unsafe"},
                ],
                violation=True,
                detection_scope="response",
                detect_at_turn=0,
                repair_expected=False,
                repair_verified_at_turn=1,
            ),
        ],
    )

    result = await EvalRunner().run(keyword_engine, suite)
    summary = result.summary

    assert summary.true_positive == 3
    assert summary.false_negative == 1
    assert summary.false_positive == 1
    assert summary.true_negative == 1
    assert summary.detection_recall == pytest.approx(0.75)
    assert summary.false_positive_rate == pytest.approx(0.5)
    assert summary.fix_success_count == 1
    assert summary.fix_failure_count == 1
    assert summary.fix_rate == pytest.approx(0.5)
    assert summary.exact_case_pass_rate == pytest.approx(4 / 6)


async def test_runner_respects_runtime_phase_for_request_cases(keyword_engine):
    suite = EvalSuite(
        name="request-phase",
        cases=[
            _case(
                "request-only",
                messages=[{"role": "user", "content": "this includes request-risk"}],
                violation=True,
                detection_scope="request",
                detect_at_turn=0,
            )
        ],
    )

    result = await EvalRunner(
        runtime=EvalRuntimeConfig(
            request_phase_enabled=False,
            response_phase_enabled=True,
            mode="async",
            fail_action="intervene",
        )
    ).run(keyword_engine, suite)

    assert result.outcomes[0].outcome == "missed_violation"
    assert result.outcomes[0].passed is False


async def test_runner_verifies_fixed_case_in_sync_intervene_mode(keyword_engine):
    suite = load_native_suite("tests/eval/fixtures/recovery_suite.yaml")

    result = await EvalRunner(
        runtime=EvalRuntimeConfig(
            request_phase_enabled=False,
            response_phase_enabled=True,
            mode="sync",
            fail_action="intervene",
        )
    ).run(keyword_engine, suite)

    assert result.outcomes[0].outcome == "detected_and_fixed"
    assert result.outcomes[0].passed is True
    assert "sync_intervention_queued_at_turn=0" in result.outcomes[0].notes


async def test_runner_marks_repair_case_not_fixed_in_sync_block_mode(keyword_engine):
    suite = load_native_suite("tests/eval/fixtures/recovery_suite.yaml")

    result = await EvalRunner(
        runtime=EvalRuntimeConfig(
            request_phase_enabled=False,
            response_phase_enabled=True,
            mode="sync",
            fail_action="block",
        )
    ).run(keyword_engine, suite)

    assert result.outcomes[0].outcome == "detected_not_fixed"
    assert result.outcomes[0].passed is False
    assert "response_blocked_at_turn=0" in result.outcomes[0].notes
