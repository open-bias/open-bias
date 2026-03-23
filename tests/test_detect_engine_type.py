"""Tests for _detect_engine_type heuristic in cli.py."""


from openbias.cli import _detect_engine_type


class TestDetectEngineType:
    """Test auto-detection of engine type from policy text."""

    def test_judge_keywords(self) -> None:
        assert _detect_engine_type("always be professional and never leak PII") == "judge"

    def test_fsm_keywords(self) -> None:
        assert _detect_engine_type("verify identity first, then proceed to refund") == "fsm"

    def test_strong_fsm_signal(self) -> None:
        assert _detect_engine_type(
            "workflow: step 1 transition to step 2, then proceed to next phase"
        ) == "fsm"

    def test_strong_judge_signal(self) -> None:
        assert _detect_engine_type(
            "ensure safe and appropriate tone, evaluate quality, never harmful"
        ) == "judge"

    def test_defaults_to_judge_when_ambiguous(self) -> None:
        assert _detect_engine_type("do something") == "judge"

    def test_defaults_to_judge_on_empty(self) -> None:
        assert _detect_engine_type("") == "judge"

    def test_case_insensitive(self) -> None:
        assert _detect_engine_type("ALWAYS be PROFESSIONAL") == "judge"

    def test_mixed_keywords_judge_wins(self) -> None:
        # "before" is FSM, but judge keywords dominate
        assert _detect_engine_type(
            "before responding, ensure professional tone, always safe and appropriate"
        ) == "judge"

    def test_mixed_keywords_fsm_wins(self) -> None:
        # "ensure" is judge, but FSM keywords dominate
        assert _detect_engine_type(
            "ensure step 1 before step 2, then transition to next stage in sequence"
        ) == "fsm"

    def test_tie_goes_to_judge(self) -> None:
        # Equal scores -> judge (the >= comparison in the function)
        assert _detect_engine_type("first ensure") == "judge"
