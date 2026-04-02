from __future__ import annotations

from pathlib import Path

import pytest

from openbias.policy.engines.judge.engine import JudgePolicyEngine
from openbias.policy.engines.judge.models import JudgeScore, JudgeVerdict, VerdictAction


class _TracerSpy:
    def __init__(self):
        self.calls = []

    def log_judge_evaluation(self, **kwargs):
        self.calls.append(kwargs)


@pytest.mark.asyncio
async def test_trace_verdict_uses_rules_file_rubric_name(tmp_path: Path):
    rules_file = tmp_path / "rules.md"
    rules_file.write_text("- Rule A\n", encoding="utf-8")
    engine = JudgePolicyEngine()
    await engine.initialize(
        {
            "models": [{"name": "primary", "model": "gpt-4o-mini"}],
            "rules_file": str(rules_file),
        }
    )

    tracer = _TracerSpy()
    engine.set_tracer(tracer)

    async def _mock_eval(*args, **kwargs):
        return JudgeVerdict(
            scores=[JudgeScore(criterion="Rule A", score=1, reasoning="ok")],
            composite_score=1.0,
            action=VerdictAction.PASS,
            summary="ok",
            judge_model="gpt-4o-mini",
        )

    engine._evaluator.evaluate_turn = _mock_eval

    await engine.evaluate_response(
        session_id="s1",
        response_data={"content": "safe"},
        request_data={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert tracer.calls
    assert tracer.calls[0]["rubric_name"] == "rules_file"
