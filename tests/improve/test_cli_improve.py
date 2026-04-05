from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from openbias.cli_improve import run_improve
from openbias.improve.schema import (
    ImprovementAggregate,
    ImprovementResult,
    ImprovementVariantResult,
    PolicyVariant,
    VariantProvenance,
)


def test_run_improve_writes_artifacts_and_reports_winner(
    tmp_path: Path,
    monkeypatch,
):
    config_path = tmp_path / "openbias.yaml"
    config_path.write_text("model: gpt-4o-mini\n", encoding="utf-8")
    output_dir = tmp_path / ".openbias" / "reports" / "latest"
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        '{"id":"trace-1","session_id":"session-1","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"ok"}]}\n',
        encoding="utf-8",
    )
    variant_path = output_dir / "variants" / "candidate-1.md"
    variant_path.parent.mkdir(parents=True, exist_ok=True)
    variant_path.write_text("- Candidate\n", encoding="utf-8")

    monkeypatch.setattr(
        "openbias.cli_improve.Settings",
        lambda **kwargs: SimpleNamespace(validate=lambda: None),
    )

    async def _fake_run_improvement(**kwargs) -> ImprovementResult:
        del kwargs
        variant = PolicyVariant(
            variant_id="candidate-1",
            policy_path=str(variant_path),
            provenance=VariantProvenance(
                baseline_policy_path=str(tmp_path / "RULES.md"),
                instruction="tighten the policy",
                variant_id="candidate-1",
                generated_policy_path=str(variant_path),
            ),
        )
        return ImprovementResult(
            status="pass",
            boundary="request",
            baseline_policy_path=str(tmp_path / "RULES.md"),
            instruction="tighten the policy",
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
            winner_variant_id="candidate-1",
            winner_policy_path=str(variant_path),
            ranked_variant_ids=["candidate-1"],
        )

    monkeypatch.setattr("openbias.cli_improve.run_improvement", _fake_run_improvement)

    json_path, md_path = run_improve(
        config=config_path,
        trace_paths=(trace_path,),
        instruction="tighten the policy",
        variant_count=1,
        output_dir=output_dir,
    )

    assert json_path.exists()
    assert md_path.exists()
    assert '"winner_variant_id": "candidate-1"' in json_path.read_text(encoding="utf-8")
    assert "Winner: `candidate-1`" in md_path.read_text(encoding="utf-8")
