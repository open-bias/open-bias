"""
Engine conformance tests.

Validates that all policy engines return results that conform to the
unified evaluation contract: EvaluationResult with normalized
ViolationRecord entries.

These tests ensure:
1. ALLOW results pass through correctly
2. VIOLATION results are mapped to the correct enforcement action by the interceptor
3. Violation metadata has the expected normalized shape (name, message, severity, engine)
4. Real engines populate ViolationRecord with required fields
"""

import pytest
import sys
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
                reason="Stay on topic",
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
                reason="Blocked",
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
                reason="Logged only",
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


# ---------------------------------------------------------------------------
# Real engine conformance: ViolationRecord required fields
# ---------------------------------------------------------------------------


def _assert_violation_record_shape(violation: ViolationRecord, engine_name: str) -> None:
    """Assert that a ViolationRecord has all required fields populated."""
    assert isinstance(violation.reason, str) and violation.reason, (
        f"{engine_name}: reason must be a non-empty string, got {violation.reason!r}"
    )
    assert isinstance(violation.engine, str) and violation.engine, (
        f"{engine_name}: engine must be a non-empty string, got {violation.engine!r}"
    )
    assert isinstance(violation.severity, str) and violation.severity, (
        f"{engine_name}: severity must be a non-empty string, got {violation.severity!r}"
    )
    assert isinstance(violation.scope, str) and violation.scope, (
        f"{engine_name}: scope must be a non-empty string, got {violation.scope!r}"
    )


class TestFSMEngineConformance:
    """FSM engine produces conformant ViolationRecord entries."""

    @pytest.fixture
    async def fsm_engine(self):
        from pathlib import Path
        from openbias.policy.registry import PolicyEngineRegistry

        workflow_path = (
            Path(__file__).resolve().parent.parent.parent
            / "examples"
            / "fsm_workflow"
            / "customer_support.yaml"
        )
        engine = await PolicyEngineRegistry.create_and_initialize(
            "fsm", {"config_path": str(workflow_path)}
        )
        yield engine
        await engine.shutdown()

    async def test_violation_record_shape(self, fsm_engine):
        """FSM violations have all required fields populated."""
        import json
        from pathlib import Path
        from openbias.eval.runner import EvalRunner

        evals_dir = Path(__file__).resolve().parent.parent.parent / "evals" / "fsm"
        messages = json.loads((evals_dir / "skip_verification.json").read_text())
        result = await EvalRunner().run(fsm_engine, messages)

        violations = [v for t in result.turns for v in t.response_eval.violations]
        assert len(violations) > 0, "Expected at least one FSM violation"

        for v in violations:
            _assert_violation_record_shape(v, "fsm")


class TestNeMoEngineConformance:
    """NeMo engine produces conformant ViolationRecord entries."""

    async def test_violation_record_shape(self):
        """NeMo violations have all required fields populated."""
        mock_nemo = MagicMock()
        sys.modules["nemoguardrails"] = mock_nemo

        from openbias.policy.engines.nemo.engine import NemoGuardrailsPolicyEngine

        engine = NemoGuardrailsPolicyEngine()

        mock_nemo.RailsConfig.from_path.return_value = MagicMock()
        mock_rails = mock_nemo.LLMRails.return_value = MagicMock()

        rail_result = MagicMock(spec=["log"])
        rail_result.log = MagicMock(spec=["activated_rails"])
        rail_result.log.activated_rails = [
            {"type": "input", "name": "block jailbreak"},
        ]
        mock_rails.generate_async = AsyncMock(return_value=rail_result)

        await engine.initialize({"config_path": "dummy"})

        result = await engine.evaluate_request(
            session_id="conformance",
            request_data={"messages": [{"role": "user", "content": "test"}]},
        )

        assert result.status == EvaluationStatus.VIOLATION
        assert len(result.violations) > 0

        for v in result.violations:
            _assert_violation_record_shape(v, "nemo")
            assert "provider_decision" in v.extra, "NeMo violations must include provider_decision"

    async def test_error_path_violation_record_shape(self):
        """NeMo fail_closed error violations have all required fields."""
        mock_nemo = MagicMock()
        sys.modules["nemoguardrails"] = mock_nemo

        from openbias.policy.engines.nemo.engine import NemoGuardrailsPolicyEngine

        engine = NemoGuardrailsPolicyEngine()
        mock_nemo.RailsConfig.from_path.return_value = MagicMock()
        mock_rails = mock_nemo.LLMRails.return_value = MagicMock()
        mock_rails.generate_async = AsyncMock(side_effect=RuntimeError("timeout"))

        await engine.initialize({"config_path": "dummy", "fail_closed": True})

        result = await engine.evaluate_request(
            session_id="conformance-err",
            request_data={"messages": [{"role": "user", "content": "test"}]},
        )

        assert result.status == EvaluationStatus.VIOLATION
        for v in result.violations:
            _assert_violation_record_shape(v, "nemo:error-path")
            assert v.extra.get("provider_decision") == "block"


class TestJudgeEngineConformance:
    """Judge engine produces conformant ViolationRecord entries."""

    async def test_violation_record_shape(self, tmp_path):
        """Judge violations have all required fields populated."""
        from openbias.policy.engines.judge.engine import JudgePolicyEngine

        engine = JudgePolicyEngine()
        rules_file = tmp_path / "rules.md"
        rules_file.write_text("- Be helpful\n- Be safe\n", encoding="utf-8")
        await engine.initialize({
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "rules_file": str(rules_file),
        })

        # Mock the judge call to return a failing verdict
        engine._client.call_judge = AsyncMock(return_value={
            "results": [
                {
                    "rule": "Be helpful",
                    "passed": False,
                    "reasoning": "Violated policy",
                    "evidence": ["bad content"],
                    "confidence": 0.9,
                },
                {
                    "rule": "Be safe",
                    "passed": True,
                    "reasoning": "OK",
                    "evidence": [],
                    "confidence": 0.9,
                },
            ],
            "summary": "Policy violation detected.",
        })

        result = await engine.evaluate_response(
            "conformance",
            {"choices": [{"message": {"content": "bad response"}}]},
            {"messages": [{"role": "user", "content": "test"}]},
        )

        assert result.status == EvaluationStatus.VIOLATION
        assert len(result.violations) > 0

        for v in result.violations:
            _assert_violation_record_shape(v, "judge")
            assert v.confidence is not None, "Judge violations must include confidence"


class TestLLMEngineConformance:
    """LLM engine produces conformant ViolationRecord entries."""

    async def test_violation_record_shape(self):
        """LLM engine violations have all required fields populated."""
        from openbias.policy.engines.llm import LLMPolicyEngine

        engine = LLMPolicyEngine()
        await engine.initialize({
            "workflow": {
                "name": "conformance-test",
                "states": [
                    {"name": "start", "is_initial": True, "description": "Start"},
                    {"name": "end", "is_terminal": True},
                ],
                "transitions": [{"from_state": "start", "to_state": "end"}],
                "constraints": [
                    {"name": "stay_on_topic", "type": "never", "target": "off_topic"},
                ],
            },
        })

        # Mock classifier to return a valid state
        engine._llm_client.complete_json = AsyncMock(
            return_value=[{"state_id": "start", "confidence": 0.9, "reasoning": "ok"}]
        )

        # Inject a constraint violation
        cv = MagicMock()
        cv.violated = True
        cv.constraint_id = "stay_on_topic"
        cv.severity = "warning"
        cv.evidence = "Went off topic"
        cv.confidence = 0.85
        engine._constraint_evaluator.evaluate = AsyncMock(return_value=[cv])

        result = await engine.evaluate_response(
            "conformance",
            {"choices": [{"message": {"content": "off topic response"}}]},
            {"messages": [{"role": "user", "content": "test"}]},
        )

        assert result.status == EvaluationStatus.VIOLATION
        assert len(result.violations) > 0

        for v in result.violations:
            _assert_violation_record_shape(v, "llm")
            assert v.confidence is not None, "LLM violations must include confidence"


# ---------------------------------------------------------------------------
# Multi-engine fail_action: interceptor treats all engines identically
# ---------------------------------------------------------------------------


# Violations from different engines, each with engine-specific metadata
_MULTI_ENGINE_VIOLATIONS = [
    pytest.param(
        ViolationRecord(
            reason="NeMo rail triggered",
            severity="error",
            engine="nemo:guardrails",
            extra={"provider_decision": "flagged", "rail_type": "input"},
        ),
        id="nemo-with-provider-metadata",
    ),
    pytest.param(
        ViolationRecord(
            reason="Constraint violated: precedence",
            severity="error",
            engine="fsm:customer_support",
        ),
        id="fsm-precedence-constraint",
    ),
    pytest.param(
        ViolationRecord(
            reason="Criterion not met: safety",
            severity="intervene",
            scope="turn",
            engine="judge:default",
            confidence=0.85,
            extra={"composite_score": 0.3, "judge_model": "gpt-4o-mini"},
        ),
        id="judge-with-composite-score",
    ),
    pytest.param(
        ViolationRecord(
            reason="Went off topic",
            severity="warning",
            engine="llm:test-workflow",
            confidence=0.85,
        ),
        id="llm-constraint-violation",
    ),
]


class TestMultiEngineFailAction:
    """Verify that the interceptor maps violations from all engines identically."""

    @pytest.fixture
    def _mock_engine_with_violation(self):
        def factory(violation: ViolationRecord) -> MagicMock:
            engine = MagicMock()
            engine.name = violation.engine
            eval_result = EvaluationResult(
                status=EvaluationStatus.VIOLATION,
                violations=[violation],
            )
            engine.evaluate_request = AsyncMock(return_value=eval_result)
            engine.evaluate_response = AsyncMock(return_value=eval_result)
            return engine
        return factory

    @pytest.mark.parametrize("violation", _MULTI_ENGINE_VIOLATIONS)
    async def test_block_blocks_regardless_of_engine(self, violation, _mock_engine_with_violation):
        """fail_action=block should block violations from any engine."""
        from openbias.core.interceptor import Interceptor

        engine = _mock_engine_with_violation(violation)
        interceptor = Interceptor(
            pre_call_evaluators=[engine],
            post_call_evaluators=[],
            fail_action="block",
        )

        result = await interceptor.run_pre_call("s1", {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
        }, "req-1")

        assert result.allowed is False, f"Expected block for engine {violation.engine}"

    @pytest.mark.parametrize("violation", _MULTI_ENGINE_VIOLATIONS)
    async def test_shadow_allows_regardless_of_engine(self, violation, _mock_engine_with_violation):
        """fail_action=shadow should allow violations from any engine."""
        from openbias.core.interceptor import Interceptor

        engine = _mock_engine_with_violation(violation)
        interceptor = Interceptor(
            pre_call_evaluators=[engine],
            post_call_evaluators=[],
            fail_action="shadow",
        )

        result = await interceptor.run_pre_call("s1", {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
        }, "req-1")

        assert result.allowed is True, f"Expected allow (shadow) for engine {violation.engine}"
        assert result.modified_data is None, "Shadow mode must not modify the request"

    @pytest.mark.parametrize("violation", _MULTI_ENGINE_VIOLATIONS)
    async def test_intervene_modifies_regardless_of_engine(self, violation, _mock_engine_with_violation):
        """fail_action=intervene should modify request for violations from any engine."""
        from openbias.core.interceptor import Interceptor

        engine = _mock_engine_with_violation(violation)
        interceptor = Interceptor(
            pre_call_evaluators=[engine],
            post_call_evaluators=[],
            fail_action="intervene",
        )

        result = await interceptor.run_pre_call("s1", {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
        }, "req-1")

        assert result.allowed is True, f"Expected allow (intervene) for engine {violation.engine}"
        assert result.modified_data is not None, f"Expected modified data for engine {violation.engine}"
