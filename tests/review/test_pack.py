from pathlib import Path

from openbias.review import render_review_pack, write_review_pack


def test_render_review_pack_includes_status_checklist_and_repro():
    markdown = render_review_pack(
        {
            "status": "review",
            "baseline_policy_path": "rules.md",
            "candidate_policy_path": "rules.candidate.md",
            "candidate_details": {"provider": "file", "metadata": {"source_path": "rules.candidate.md"}},
            "gates": [{"status": "review", "reason": "Needs a human look."}],
            "suites": [
                {
                    "name": "safe",
                    "delta_exact_case_pass_rate": 0.1,
                    "delta_false_positive_rate": -0.05,
                }
            ],
            "traces": [
                {
                    "name": "prod",
                    "delta_matched_detection_rate": -0.1,
                    "delta_detection_rate": 0.05,
                }
            ],
        }
    )

    assert "# Policy Review Pack" in markdown
    assert "Reviewer Checklist" in markdown
    assert "openbias compare --candidate rules.candidate.md" in markdown
    assert "Candidate provider: `file`" in markdown
    assert "`prod` matched-detection-rate dropped by -10.00%." in markdown
    assert "`prod` detection-rate improved by +5.00%." in markdown


def test_write_review_pack_persists_markdown(tmp_path: Path):
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        '{"status":"pass","baseline_policy_path":"rules.md","candidate_policy_path":"rules.candidate.md","gates":[],"suites":[],"traces":[]}',
        encoding="utf-8",
    )

    output_path = write_review_pack(
        comparison_path=comparison_path,
        output_path=tmp_path / "review-pack.md",
    )

    assert output_path.exists()
    assert "Policy Review Pack" in output_path.read_text(encoding="utf-8")
