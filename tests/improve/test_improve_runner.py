from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openbias.improve import render_improvement_markdown
from openbias.improve.runner import generate_variants, run_improvement
from openbias.improve.schema import (
    ImprovementAggregate,
    ImprovementResult,
    ImprovementVariantResult,
    PolicyVariant,
    VariantProvenance,
)
from openbias.policy.protocols import EvaluationResult, EvaluationStatus, ViolationRecord
from openbias.traces import TraceCase, TraceDataset, TraceMetadata, save_trace_dataset


@dataclass
class FakeReplayEngine:
    request_rule: str | None = None
    raise_on: str | None = None
    sessions: dict[str, list[str]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def engine_type(self) -> str:
        return "fake"

    async def initialize(self, config: dict[str, Any]) -> None:
        return None

    async def evaluate_request(
        self,
        session_id: str,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        del context
        self.sessions.setdefault(session_id, [])
        joined = " ".join(
            message.get("content", "")
            for message in request_data.get("messages", [])
            if isinstance(message, dict)
        )
        if self.raise_on and self.raise_on in joined:
            raise RuntimeError(f"boom: {self.raise_on}")
        if self.request_rule and self.request_rule in joined:
            return EvaluationResult(
                status=EvaluationStatus.VIOLATION,
                violations=[ViolationRecord(reason=self.request_rule)],
            )
        return EvaluationResult(status=EvaluationStatus.ALLOW)

    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        del session_id, response_data, request_data, context
        return EvaluationResult(status=EvaluationStatus.ALLOW)

    async def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        return {"session_id": session_id}

    async def reset_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    async def shutdown(self) -> None:
        self.sessions.clear()


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        replay=SimpleNamespace(boundary="request"),
        proxy=SimpleNamespace(default_model="gpt-4o-mini"),
    )


def _policy_variant(
    *,
    tmp_path: Path,
    variant_id: str,
    instruction: str = "tighten the policy",
) -> PolicyVariant:
    variants_dir = tmp_path / "reports" / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = tmp_path / "RULES.md"
    if not baseline_path.exists():
        baseline_path.write_text("- Baseline\n", encoding="utf-8")
    policy_path = variants_dir / f"{variant_id}.md"
    policy_path.write_text(f"# {variant_id}\n", encoding="utf-8")
    return PolicyVariant(
        variant_id=variant_id,
        policy_path=str(policy_path),
        provenance=VariantProvenance(
            baseline_policy_path=str(baseline_path),
            instruction=instruction,
            variant_id=variant_id,
            generated_policy_path=str(policy_path),
        ),
    )


def _dataset(name: str, cases: list[tuple[str, str, bool | None]]) -> TraceDataset:
    return TraceDataset(
        name=name,
        cases=[
            TraceCase(
                id=case_id,
                session_id=f"session-{case_id}",
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": "ok"},
                ],
                metadata=TraceMetadata(final_action="unknown"),
                labels=({"violation": violation} if violation is not None else None),
            )
            for case_id, user, violation in cases
        ],
    )


@pytest.mark.asyncio
async def test_generate_variants_includes_baseline_instruction_and_output_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = tmp_path / "openbias.yaml"
    config_path.write_text("model: gpt-4o-mini\n", encoding="utf-8")
    (tmp_path / "RULES.md").write_text("- Keep responses safe\n", encoding="utf-8")

    async def _fake_request_variant_documents(**kwargs) -> list[str]:
        assert kwargs["instruction"] == "tighten the policy"
        assert kwargs["variant_count"] == 2
        return ["- Candidate one\n", "- Candidate two\n"]

    monkeypatch.setattr(
        "openbias.improve.runner._request_variant_documents",
        _fake_request_variant_documents,
    )

    variants = await generate_variants(
        settings=_settings(),
        config_path=config_path,
        instruction="tighten the policy",
        variant_count=2,
        output_dir=tmp_path / ".openbias" / "reports" / "latest",
    )

    assert [variant.variant_id for variant in variants] == [
        "baseline",
        "candidate-1",
        "candidate-2",
    ]
    assert variants[0].provenance.instruction == "tighten the policy"
    assert Path(variants[0].policy_path).read_text(encoding="utf-8") == "- Keep responses safe\n"
    assert Path(variants[1].provenance.generated_policy_path).exists()
    assert Path(variants[2].provenance.generated_policy_path).exists()


@pytest.mark.asyncio
async def test_run_improvement_ranks_variants_across_multiple_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = tmp_path / "openbias.yaml"
    config_path.write_text("model: gpt-4o-mini\n", encoding="utf-8")
    trace_one = save_trace_dataset(
        _dataset(
            "trace-one",
            [
                ("case-1", "alpha risk", True),
                ("case-2", "hello", False),
            ],
        ),
        tmp_path / "trace-one.jsonl",
    )
    trace_two = save_trace_dataset(
        _dataset("trace-two", [("case-3", "bravo risk", True)]),
        tmp_path / "trace-two.jsonl",
    )

    variants = [
        _policy_variant(tmp_path=tmp_path, variant_id="baseline"),
        _policy_variant(tmp_path=tmp_path, variant_id="candidate-1"),
        _policy_variant(tmp_path=tmp_path, variant_id="candidate-2"),
    ]

    async def _fake_generate_variants(**kwargs) -> list[PolicyVariant]:
        del kwargs
        return variants

    async def _fake_build_engine_for_policy(*, settings, config_path, rules_path):
        del settings, config_path
        name = Path(rules_path).stem
        if name == "baseline":
            return FakeReplayEngine(request_rule="alpha")
        if name == "candidate-1":
            return FakeReplayEngine(request_rule="risk")
        return FakeReplayEngine(request_rule="hello")

    monkeypatch.setattr("openbias.improve.runner.generate_variants", _fake_generate_variants)
    monkeypatch.setattr(
        "openbias.improve.runner.build_engine_for_policy",
        _fake_build_engine_for_policy,
    )

    result = await run_improvement(
        settings=_settings(),
        config_path=config_path,
        trace_paths=(trace_one, trace_two),
        instruction="tighten the policy",
        variant_count=2,
        output_dir=tmp_path / ".openbias" / "reports" / "latest",
    )

    assert result.status == "pass"
    assert result.winner_variant_id == "candidate-1"
    assert result.ranked_variant_ids == ["candidate-1", "baseline", "candidate-2"]
    assert result.variants[0].aggregate.matched_rate == 1.0


@pytest.mark.asyncio
async def test_run_improvement_returns_review_when_labeled_coverage_is_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = tmp_path / "openbias.yaml"
    config_path.write_text("model: gpt-4o-mini\n", encoding="utf-8")
    trace_path = save_trace_dataset(
        _dataset("trace-unlabeled", [("case-1", "alpha risk", None)]),
        tmp_path / "trace.jsonl",
    )
    variants = [
        _policy_variant(tmp_path=tmp_path, variant_id="baseline"),
        _policy_variant(tmp_path=tmp_path, variant_id="candidate-1"),
    ]

    async def _fake_generate_variants(**kwargs) -> list[PolicyVariant]:
        del kwargs
        return variants

    async def _fake_build_engine_for_policy(*, settings, config_path, rules_path):
        del settings, config_path, rules_path
        return FakeReplayEngine(request_rule="risk")

    monkeypatch.setattr("openbias.improve.runner.generate_variants", _fake_generate_variants)
    monkeypatch.setattr(
        "openbias.improve.runner.build_engine_for_policy",
        _fake_build_engine_for_policy,
    )

    result = await run_improvement(
        settings=_settings(),
        config_path=config_path,
        trace_paths=(trace_path,),
        instruction="tighten the policy",
        variant_count=1,
        output_dir=tmp_path / ".openbias" / "reports" / "latest",
    )

    assert result.status == "review"
    assert result.winner_variant_id is None
    assert "No labeled replay coverage" in (result.review_reason or "")


@pytest.mark.asyncio
async def test_run_improvement_records_replay_failures_without_aborting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = tmp_path / "openbias.yaml"
    config_path.write_text("model: gpt-4o-mini\n", encoding="utf-8")
    trace_path = save_trace_dataset(
        _dataset(
            "trace-failures",
            [
                ("case-1", "hello", False),
                ("case-2", "explode", True),
            ],
        ),
        tmp_path / "trace.jsonl",
    )
    variants = [_policy_variant(tmp_path=tmp_path, variant_id="baseline")]

    async def _fake_generate_variants(**kwargs) -> list[PolicyVariant]:
        del kwargs
        return variants

    async def _fake_build_engine_for_policy(*, settings, config_path, rules_path):
        del settings, config_path, rules_path
        return FakeReplayEngine(request_rule="risk", raise_on="explode")

    monkeypatch.setattr("openbias.improve.runner.generate_variants", _fake_generate_variants)
    monkeypatch.setattr(
        "openbias.improve.runner.build_engine_for_policy",
        _fake_build_engine_for_policy,
    )

    result = await run_improvement(
        settings=_settings(),
        config_path=config_path,
        trace_paths=(trace_path,),
        instruction="tighten the policy",
        variant_count=0,
        output_dir=tmp_path / ".openbias" / "reports" / "latest",
    )

    assert result.status == "pass"
    assert result.variants[0].aggregate.failures == 1
    assert result.variants[0].traces[0].failures[0]["case_id"] == "case-2"


def test_render_improvement_markdown_includes_winner_and_ranking(tmp_path: Path):
    variant = _policy_variant(tmp_path=tmp_path, variant_id="candidate-1")
    markdown = render_improvement_markdown(
        ImprovementResult(
            status="pass",
            boundary="request",
            baseline_policy_path=str(tmp_path / "RULES.md"),
            instruction="tighten the policy",
            winner_variant_id="candidate-1",
            winner_policy_path=variant.policy_path,
            variants=[
                ImprovementVariantResult(
                    variant=variant,
                    aggregate=ImprovementAggregate(
                        labeled_cases=2,
                        matched_cases=2,
                        mismatched_cases=0,
                        matched_rate=1.0,
                        detection_rate=0.5,
                        failures=0,
                    ),
                )
            ],
            ranked_variant_ids=["candidate-1"],
        )
    )

    assert "candidate-1" in markdown
    assert "matched-rate `100.00%`" in markdown
