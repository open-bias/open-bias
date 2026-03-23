"""Tests that all session/context timestamps are UTC-aware and consistent."""

from datetime import datetime, timezone

import pytest

from openbias.policy.engines.judge.models import JudgeSessionContext
from openbias.policy.engines.fsm.workflow.state_machine import (
    SessionState,
    StateHistoryEntry,
)
from openbias.policy.engines.llm.models import ConfidenceTier, SessionContext


def _assert_utc_aware(dt: datetime) -> None:
    assert dt.tzinfo is not None, "datetime should be timezone-aware"
    assert dt.tzinfo == timezone.utc, "datetime should be UTC"


class TestJudgeSessionContextTimestamps:
    """Judge engine session context uses UTC-aware timestamps."""

    def test_created_at_and_last_updated_at_are_timezone_aware(self):
        ctx = JudgeSessionContext(session_id="test")
        _assert_utc_aware(ctx.created_at)
        _assert_utc_aware(ctx.last_updated_at)

    def test_serialization_roundtrip_preserves_utc(self):
        ctx = JudgeSessionContext(session_id="test")
        d = ctx.to_dict()
        created = datetime.fromisoformat(d["created_at"])
        last_updated = datetime.fromisoformat(d["last_updated_at"])
        _assert_utc_aware(created)
        _assert_utc_aware(last_updated)


class TestFSMSessionStateTimestamps:
    """FSM engine session state uses UTC-aware timestamps."""

    def test_created_at_and_last_updated_are_timezone_aware(self):
        session = SessionState(
            session_id="s",
            workflow_name="w",
            current_state="start",
        )
        _assert_utc_aware(session.created_at)
        _assert_utc_aware(session.last_updated)

    def test_state_history_entry_accepts_utc_aware_entered_at(self):
        now = datetime.now(timezone.utc)
        entry = StateHistoryEntry(state_name="start", entered_at=now)
        _assert_utc_aware(entry.entered_at)


class TestLLMSessionContextTimestamps:
    """LLM engine session context uses UTC-aware timestamps."""

    def test_created_at_and_last_updated_at_are_timezone_aware(self):
        ctx = SessionContext(
            session_id="s",
            workflow_name="w",
            current_state="start",
        )
        _assert_utc_aware(ctx.created_at)
        _assert_utc_aware(ctx.last_updated_at)

    def test_state_transition_timestamp_is_utc_aware(self):
        ctx = SessionContext(
            session_id="s",
            workflow_name="w",
            current_state="start",
        )
        ctx.record_transition(
            from_state="start",
            to_state="next",
            confidence=0.9,
            tier=ConfidenceTier.CONFIDENT,
            drift_score=0.0,
        )
        assert len(ctx.state_history) == 1
        _assert_utc_aware(ctx.state_history[0].timestamp)

    def test_serialization_roundtrip_preserves_utc(self):
        ctx = SessionContext(
            session_id="s",
            workflow_name="w",
            current_state="start",
        )
        d = ctx.to_dict()
        created = datetime.fromisoformat(d["created_at"])
        last_updated = datetime.fromisoformat(d["last_updated_at"])
        _assert_utc_aware(created)
        _assert_utc_aware(last_updated)
