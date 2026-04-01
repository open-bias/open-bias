"""Evaluation tests for the FSM policy engine using conversation scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openbias.eval.runner import EvalRunner
from openbias.policy.protocols import EvaluationStatus
from openbias.policy.registry import PolicyEngineRegistry

EVALS_DIR = Path(__file__).resolve().parent.parent.parent / "evals" / "fsm"
WORKFLOW_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "fsm_workflow"
    / "customer_support.yaml"
)


@pytest.fixture
async def engine():
    eng = await PolicyEngineRegistry.create_and_initialize(
        "fsm",
        {"config_path": str(WORKFLOW_PATH)},
    )
    yield eng
    await eng.shutdown()


@pytest.fixture
def runner() -> EvalRunner:
    return EvalRunner()


async def test_happy_path_no_violations(engine, runner):
    """Happy path: greeting → identify → verify → account_action → resolution.

    All turns should be ALLOW with no violations since the agent follows
    proper procedure (verifies identity before account action).
    """
    messages = json.loads((EVALS_DIR / "happy_path.json").read_text())
    result = await runner.run(engine, messages)

    assert result.error is None
    assert len(result.turns) > 0

    for turn in result.turns:
        assert turn.response_eval.status == EvaluationStatus.ALLOW
        assert len(turn.response_eval.metadata.get("violations", [])) == 0


async def test_skip_verification_detected(engine, runner):
    """Violation: agent skips identity_verification and jumps to account_action.

    The change_subscription tool call should trigger the
    verify_identity_before_take_account_action constraint, producing an INTERVENE or BLOCK.
    """
    messages = json.loads((EVALS_DIR / "skip_verification.json").read_text())
    result = await runner.run(engine, messages)

    assert result.error is None
    assert len(result.turns) > 0

    # At least one turn must have caught the violation
    violation_turns = [
        t
        for t in result.turns
        if t.response_eval.status == EvaluationStatus.VIOLATION
    ]
    assert len(violation_turns) > 0, "Expected at least one turn with a violation decision"

    # The specific constraint name should appear in violations
    all_violation_names = [
        v.rule_name
        for t in result.turns
        for v in t.response_eval.violations
    ]
    assert "verify_identity_before_take_account_action" in all_violation_names, (
        f"Expected 'verify_identity_before_take_account_action' in violations, got: {all_violation_names}"
    )


async def test_escalation_path_no_violations(engine, runner):
    """Escalation path: greeting → identify → lookup → escalate → resolution.

    All turns should be ALLOW with no violations since the agent follows
    a valid workflow path without any account actions.
    """
    messages = json.loads((EVALS_DIR / "escalation_path.json").read_text())
    result = await runner.run(engine, messages)

    assert result.error is None
    assert len(result.turns) > 0

    for turn in result.turns:
        assert turn.response_eval.status == EvaluationStatus.ALLOW
        assert len(turn.response_eval.metadata.get("violations", [])) == 0


async def test_multi_issue_no_violations(engine, runner):
    """Multi-issue: resolution → identify_issue loop with lookups.

    All turns should be ALLOW with no violations since the agent handles
    multiple issues via knowledge base lookups without account modifications.
    """
    messages = json.loads((EVALS_DIR / "multi_issue.json").read_text())
    result = await runner.run(engine, messages)

    assert result.error is None
    assert len(result.turns) > 0

    for turn in result.turns:
        assert turn.response_eval.status == EvaluationStatus.ALLOW
        assert len(turn.response_eval.metadata.get("violations", [])) == 0


async def test_direct_account_action_detected(engine, runner):
    """Violation: agent jumps to account_action without identity verification.

    The update_billing tool call should trigger the
    verify_identity_before_take_account_action constraint, producing an INTERVENE or BLOCK.
    """
    messages = json.loads((EVALS_DIR / "direct_account_action.json").read_text())
    result = await runner.run(engine, messages)

    assert result.error is None
    assert len(result.turns) > 0

    # At least one turn must have caught the violation
    violation_turns = [
        t
        for t in result.turns
        if t.response_eval.status == EvaluationStatus.VIOLATION
    ]
    assert len(violation_turns) > 0, "Expected at least one turn with a violation decision"

    # The specific constraint name should appear in violations
    all_violation_names = [
        v.rule_name
        for t in result.turns
        for v in t.response_eval.violations
    ]
    assert "verify_identity_before_take_account_action" in all_violation_names, (
        f"Expected 'verify_identity_before_take_account_action' in violations, got: {all_violation_names}"
    )
