"""Generate human-review artifacts from policy comparison results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_comparison_payload(path: str | Path) -> dict[str, Any]:
    """Load a comparison JSON payload from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_review_pack(payload: dict[str, Any]) -> str:
    """Render a reviewer-friendly Markdown package from comparison JSON."""

    gates = payload.get("gates", [])
    suites = payload.get("suites", [])
    traces = payload.get("traces", [])
    status = payload.get("status", "review")
    candidate = payload.get("candidate_policy_path", "(unknown)")
    baseline = payload.get("baseline_policy_path", "(unknown)")
    candidate_details = payload.get("candidate_details", {})

    wins: list[str] = []
    regressions: list[str] = []
    for suite in suites:
        if suite.get("delta_exact_case_pass_rate", 0) > 0:
            wins.append(
                f"`{suite['name']}` exact-case pass rate improved by {suite['delta_exact_case_pass_rate']:+.2%}."
            )
        if suite.get("delta_false_positive_rate", 0) < 0:
            wins.append(
                f"`{suite['name']}` false-positive rate improved by {suite['delta_false_positive_rate']:+.2%}."
            )
        if suite.get("delta_exact_case_pass_rate", 0) < 0:
            regressions.append(
                f"`{suite['name']}` exact-case pass rate dropped by {suite['delta_exact_case_pass_rate']:+.2%}."
            )
        if suite.get("delta_false_positive_rate", 0) > 0:
            regressions.append(
                f"`{suite['name']}` false-positive rate increased by {suite['delta_false_positive_rate']:+.2%}."
            )

    for trace in traces:
        if trace.get("delta_matched_detection_rate", 0) > 0:
            wins.append(
                f"`{trace['name']}` matched-detection-rate improved by {trace['delta_matched_detection_rate']:+.2%}."
            )
        if trace.get("delta_detection_rate", 0) > 0:
            wins.append(
                f"`{trace['name']}` detection-rate improved by {trace['delta_detection_rate']:+.2%}."
            )
        if trace.get("delta_matched_detection_rate", 0) < 0:
            regressions.append(
                f"`{trace['name']}` matched-detection-rate dropped by {trace['delta_matched_detection_rate']:+.2%}."
            )
        if trace.get("delta_detection_rate", 0) < 0:
            regressions.append(
                f"`{trace['name']}` detection-rate dropped by {trace['delta_detection_rate']:+.2%}."
            )

    lines = [
        "# Policy Review Pack",
        "",
        f"- Status: `{status}`",
        f"- Baseline policy: `{baseline}`",
        f"- Candidate policy: `{candidate}`",
        f"- Candidate provider: `{candidate_details.get('provider', 'unknown')}`",
        "",
        "## Gate Summary",
    ]
    for gate in gates:
        lines.append(f"- `{gate.get('status', 'review')}` {gate.get('reason', '').strip()}")

    lines.extend(["", "## Wins"])
    for win in wins or ["- No metric improvements were recorded automatically."]:
        lines.append(win if win.startswith("- ") else f"- {win}")

    lines.extend(["", "## Regressions"])
    for regression in regressions or ["- No regressions were flagged automatically."]:
        lines.append(regression if regression.startswith("- ") else f"- {regression}")

    lines.extend(
        [
            "",
            "## Reproduction",
            f"- `openbias compare --candidate {candidate}`",
            "",
            "## Candidate Provenance",
            f"- Provider: `{candidate_details.get('provider', 'unknown')}`",
            f"- Source path: `{candidate_details.get('metadata', {}).get('source_path', candidate)}`",
            "",
            "## Reviewer Checklist",
            "- Confirm the candidate policy intent matches the business rule you want to change.",
            "- Read the failing or changed cases in `comparison.md` and spot-check whether the candidate behavior is desirable.",
            "- Verify guard/false-positive behavior did not regress on sensitive workflows.",
            "- Approve by merging the PR or copying the candidate policy into `rules.md` only after human review.",
            "",
        ]
    )
    return "\n".join(lines)


def write_review_pack(
    *,
    comparison_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Load comparison JSON and write a Markdown review package."""

    payload = load_comparison_payload(comparison_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_review_pack(payload), encoding="utf-8")
    return target
