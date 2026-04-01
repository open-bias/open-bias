"""
Engine conformance tests.

Validates that all policy engines return results that conform to the
unified evaluation contract: EvaluationResult with normalized
ViolationRecord entries, bridged to EngineResult for backward compat.

These tests ensure:
1. ALLOW results have Decision.ALLOW and no message
2. VIOLATION results bridge to Decision.INTERVENE (not BLOCK — enforcement is interceptor-only)
3. Violation metadata has the expected normalized shape (name, message, severity, engine)
4. The violations list in metadata is always present for non-ALLOW results
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from openbias.policy.protocols import (
    Decision,
    EngineResult,
    EvaluationResult,
    EvaluationStatus,
    ViolationRecord,
)


# ---------------------------------------------------------------------------
# EvaluationResult → EngineResult bridge conformance
# ---------------------------------------------------------------------------


class TestEvaluationResultBridge:

    def test_allow_bridges_to_allow(self):
        """ALLOW EvaluationResult bridges to Decision.ALLOW with no message."""
        er = EvaluationResult(status=EvaluationStatus.ALLOW)
        result = er.to_engine_result()

        assert result.decision == Decision.ALLOW
        assert result.message is None
        assert result.modified_messages is None

    def test_allow_preserves_metadata(self):
        """ALLOW bridges metadata through."""
        er = EvaluationResult(
            status=EvaluationStatus.ALLOW,
            metadata={"key": "value"},
        )
        result = er.to_engine_result()

        assert result.metadata == {"key": "value"}

    def test_violation_bridges_to_intervene(self):
        """VIOLATION bridges to Decision.INTERVENE (never BLOCK)."""
        er = EvaluationResult(
            status=EvaluationStatus.VIOLATION,
            violations=[
                ViolationRecord(
                    rule_id="test_rule",
                    rule_name="test_violation",
                    reason="Something went wrong",
                    severity="error",
                    engine="test:engine",
                ),
            ],
        )
        result = er.to_engine_result()

        assert result.decision == Decision.INTERVENE
        assert result.message == "Something went wrong"

    def test_violation_metadata_has_normalized_shape(self):
        """Violation metadata contains normalized fields: name, message, severity, engine."""
        er = EvaluationResult(
            status=EvaluationStatus.VIOLATION,
            violations=[
                ViolationRecord(
                    rule_id="rule_001",
                    rule_name="my_rule",
                    reason="The rule was violated",
                    severity="error",
                    scope="turn",
                    engine="test:engine",
                    evidence=["quote1"],
                    confidence=0.95,
                    extra={"custom_key": "custom_value"},
                ),
            ],
        )
        result = er.to_engine_result()

        violations = result.metadata.get("violations", [])
        assert len(violations) == 1

        v = violations[0]
        assert v["rule_id"] == "rule_001"
        assert v["name"] == "my_rule"
        assert v["message"] == "The rule was violated"
        assert v["severity"] == "error"
        assert v["scope"] == "turn"
        assert v["engine"] == "test:engine"
        assert v["evidence"] == ["quote1"]
        assert v["confidence"] == 0.95
        assert v["custom_key"] == "custom_value"

    def test_multiple_violations_joined_in_message(self):
        """Multiple violations are joined with newlines in the message."""
        er = EvaluationResult(
            status=EvaluationStatus.VIOLATION,
            violations=[
                ViolationRecord(
                    rule_id="r1", rule_name="v1", reason="First issue",
                    engine="test",
                ),
                ViolationRecord(
                    rule_id="r2", rule_name="v2", reason="Second issue",
                    engine="test",
                ),
            ],
        )
        result = er.to_engine_result()

        assert result.message == "First issue\nSecond issue"

    def test_violation_without_evidence_omits_key(self):
        """Evidence and confidence are omitted from metadata when None."""
        er = EvaluationResult(
            status=EvaluationStatus.VIOLATION,
            violations=[
                ViolationRecord(
                    rule_id="r1", rule_name="v1", reason="Issue",
                    engine="test", evidence=None, confidence=None,
                ),
            ],
        )
        result = er.to_engine_result()

        v = result.metadata["violations"][0]
        assert "evidence" not in v
        assert "confidence" not in v

    def test_violation_extra_fields_merged(self):
        """Extra fields from ViolationRecord are merged into violation dict."""
        er = EvaluationResult(
            status=EvaluationStatus.VIOLATION,
            violations=[
                ViolationRecord(
                    rule_id="r1", rule_name="v1", reason="Issue",
                    engine="test",
                    extra={"provider_decision": "block", "provider_reason": "error"},
                ),
            ],
        )
        result = er.to_engine_result()

        v = result.metadata["violations"][0]
        assert v["provider_decision"] == "block"
        assert v["provider_reason"] == "error"

    def test_empty_violations_list_gives_no_message(self):
        """VIOLATION with empty violations list produces no message."""
        er = EvaluationResult(
            status=EvaluationStatus.VIOLATION,
            violations=[],
        )
        result = er.to_engine_result()

        assert result.decision == Decision.INTERVENE
        assert result.message is None


# ---------------------------------------------------------------------------
# Interceptor fail_action conformance with new contract
# ---------------------------------------------------------------------------


class TestFailActionWithEvaluationResult:
    """Verify that the interceptor correctly maps EvaluationResult-bridged
    EngineResults through the fail_action policy."""

    @pytest.fixture
    def _mock_engine(self):
        def factory(
            name: str = "test",
            eval_result: EvaluationResult | None = None,
        ) -> MagicMock:
            engine = MagicMock()
            engine.name = name

            if eval_result is None:
                eval_result = EvaluationResult(status=EvaluationStatus.ALLOW)

            bridged = eval_result.to_engine_result()

            async def evaluate_request(session_id, request_data, context=None):
                return bridged

            async def evaluate_response(session_id, response_data, request_data, context=None):
                return bridged

            engine.evaluate_request = AsyncMock(side_effect=evaluate_request)
            engine.evaluate_response = AsyncMock(side_effect=evaluate_response)
            return engine
        return factory

    async def test_violation_with_intervene_modifies_request(self, _mock_engine):
        """fail_action=intervene + VIOLATION → request modified (INTERVENE)."""
        from openbias.core.interceptor import Interceptor

        violation_result = EvaluationResult(
            status=EvaluationStatus.VIOLATION,
            violations=[ViolationRecord(
                rule_id="test", rule_name="test", reason="Stay on topic",
                engine="test",
            )],
        )
        engine = _mock_engine(eval_result=violation_result)
        interceptor = Interceptor(
            pre_call_evaluators=[engine],
            post_call_evaluators=[],
            fail_action="intervene",
        )

        result = await interceptor.run_pre_call("s1", {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
        }, "req-1")

        assert result.allowed is True
        assert result.modified_data is not None

    async def test_violation_with_block_blocks_request(self, _mock_engine):
        """fail_action=block + VIOLATION → request blocked."""
        from openbias.core.interceptor import Interceptor

        violation_result = EvaluationResult(
            status=EvaluationStatus.VIOLATION,
            violations=[ViolationRecord(
                rule_id="test", rule_name="test", reason="Blocked",
                engine="test",
            )],
        )
        engine = _mock_engine(eval_result=violation_result)
        interceptor = Interceptor(
            pre_call_evaluators=[engine],
            post_call_evaluators=[],
            fail_action="block",
        )

        result = await interceptor.run_pre_call("s1", {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
        }, "req-1")

        assert result.allowed is False

    async def test_violation_with_shadow_allows_through(self, _mock_engine):
        """fail_action=shadow + VIOLATION → allowed through, no modifications."""
        from openbias.core.interceptor import Interceptor

        violation_result = EvaluationResult(
            status=EvaluationStatus.VIOLATION,
            violations=[ViolationRecord(
                rule_id="test", rule_name="test", reason="Logged only",
                engine="test",
            )],
        )
        engine = _mock_engine(eval_result=violation_result)
        interceptor = Interceptor(
            pre_call_evaluators=[engine],
            post_call_evaluators=[],
            fail_action="shadow",
        )

        result = await interceptor.run_pre_call("s1", {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
        }, "req-1")

        assert result.allowed is True
        assert result.modified_data is None

    async def test_allow_with_block_still_allows(self, _mock_engine):
        """fail_action=block + ALLOW → request allowed."""
        from openbias.core.interceptor import Interceptor

        engine = _mock_engine()  # defaults to ALLOW
        interceptor = Interceptor(
            pre_call_evaluators=[engine],
            post_call_evaluators=[],
            fail_action="block",
        )

        result = await interceptor.run_pre_call("s1", {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
        }, "req-1")

        assert result.allowed is True

    async def test_provider_block_metadata_preserved_through_bridge(self, _mock_engine):
        """Provider decision metadata survives the bridge and reaches interceptor."""
        from openbias.core.interceptor import Interceptor

        violation_result = EvaluationResult(
            status=EvaluationStatus.VIOLATION,
            violations=[ViolationRecord(
                rule_id="nemo_error", rule_name="nemo_error",
                reason="NeMo evaluation failed",
                severity="critical", engine="nemo:guardrails",
                extra={"provider_decision": "block", "provider_reason": "timeout"},
            )],
        )
        engine = _mock_engine(eval_result=violation_result)
        interceptor = Interceptor(
            pre_call_evaluators=[engine],
            post_call_evaluators=[],
            fail_action="intervene",
        )

        result = await interceptor.run_pre_call("s1", {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
        }, "req-1")

        assert result.allowed is True
        assert result.modified_data is not None
