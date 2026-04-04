"""File-backed candidate provider for OSS workflows."""

from __future__ import annotations

from pathlib import Path

from openbias.candidates.models import CandidatePolicyBundle


class FileCandidateProvider:
    """Load a candidate policy directly from a local file."""

    def generate(
        self,
        *,
        baseline_policy_path: Path,
        candidate_policy_path: Path | None = None,
    ) -> CandidatePolicyBundle:
        if candidate_policy_path is None:
            raise ValueError("FileCandidateProvider requires candidate_policy_path.")

        return CandidatePolicyBundle(
            name=candidate_policy_path.stem,
            policy_path=str(candidate_policy_path),
            policy_text=candidate_policy_path.read_text(encoding="utf-8"),
            provider="file",
            metadata={
                "baseline_policy_path": str(baseline_policy_path),
                "source_path": str(candidate_policy_path),
            },
        )
