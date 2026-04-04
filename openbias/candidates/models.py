"""Models and protocols for candidate policy loading/generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class CandidatePolicyBundle:
    """Candidate policy plus provenance metadata."""

    name: str
    policy_path: str
    policy_text: str
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def provenance_dict(self) -> dict[str, Any]:
        """Return the persisted provenance payload for reports and review artifacts."""

        return {
            "name": self.name,
            "provider": self.provider,
            "metadata": dict(self.metadata),
        }


class PolicyCandidateProvider(Protocol):
    """Protocol for candidate-policy providers."""

    def generate(
        self,
        *,
        baseline_policy_path: Path,
        candidate_policy_path: Path | None = None,
    ) -> CandidatePolicyBundle:
        """Return a candidate policy bundle."""
