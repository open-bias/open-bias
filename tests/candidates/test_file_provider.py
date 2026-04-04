from pathlib import Path

from openbias.candidates import FileCandidateProvider


def test_file_candidate_provider_builds_bundle(tmp_path: Path):
    baseline = tmp_path / "rules.md"
    baseline.write_text("- Baseline\n", encoding="utf-8")
    candidate = tmp_path / "rules.candidate.md"
    candidate.write_text("- Candidate\n", encoding="utf-8")

    bundle = FileCandidateProvider().generate(
        baseline_policy_path=baseline,
        candidate_policy_path=candidate,
    )

    assert bundle.provider == "file"
    assert bundle.policy_path == str(candidate)
    assert bundle.metadata["baseline_policy_path"] == str(baseline)


def test_candidate_bundle_provenance_omits_policy_text(tmp_path: Path):
    baseline = tmp_path / "rules.md"
    baseline.write_text("- Baseline\n", encoding="utf-8")
    candidate = tmp_path / "rules.candidate.md"
    candidate.write_text("- Candidate\n", encoding="utf-8")

    bundle = FileCandidateProvider().generate(
        baseline_policy_path=baseline,
        candidate_policy_path=candidate,
    )

    assert bundle.provenance_dict() == {
        "name": "rules.candidate",
        "provider": "file",
        "metadata": {
            "baseline_policy_path": str(baseline),
            "source_path": str(candidate),
        },
    }
