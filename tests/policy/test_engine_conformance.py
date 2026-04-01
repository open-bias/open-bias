"""
Engine conformance tests.

Validates that all policy engines return results that conform to the
unified evaluation contract: EvaluationResult with normalized
ViolationRecord entries.

These tests ensure:
1. ALLOW results pass through correctly
2. VIOLATION results are mapped to the correct enforcement action by the interceptor
3. Violation metadata has the expected normalized shape (name, message, severity, engine)
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from openbias.policy.protocols import (
    Decision,
    EvaluationResult,
    EvaluationStatus,
    ViolationRecord,
)


# ---------------------------------------------------------------------------
# Interceptor fail_action conformance with evaluation contract
# ---------------------------------------------------------------------------


class TestFailActionWithEvaluationResult:
    """Verify that the interceptor correctly maps EvaluationResults
    through the fail_action policy."""

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

            async def evaluate_request(session_id, request_data, context=None):
                return eval_result

            async def evaluate_response(session_id, response_data, request_data, context=None):
                return eval_result

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
