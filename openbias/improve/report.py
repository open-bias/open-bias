"""Render compact review artifacts for replay-based policy improvement."""

from __future__ import annotations

from openbias.improve.schema import ImprovementResult


def render_improvement_markdown(result: ImprovementResult) -> str:
    """Render a minimal Markdown review artifact for one improvement run."""

    lines = [
        "# Policy Improvement",
        "",
        f"- Status: `{result.status}`",
        f"- Boundary: `{result.boundary}`",
        f"- Baseline: `{result.baseline_policy_path}`",
        f"- Instruction: {result.instruction}",
        (
            f"- Winner: `{result.winner_variant_id}` -> `{result.winner_policy_path}`"
            if result.winner_variant_id and result.winner_policy_path
            else "- Winner: none"
        ),
    ]
    if result.review_reason:
        lines.append(f"- Review reason: {result.review_reason}")

    lines.extend(["", "## Ranking"])
    for variant_result in result.variants:
        aggregate = variant_result.aggregate
        lines.append(
            "- "
            f"`{variant_result.variant.variant_id}` matched-rate `{aggregate.matched_rate:.2%}`, "
            f"mismatches `{aggregate.mismatched_cases}`, "
            f"detection-rate `{aggregate.detection_rate:.2%}`, "
            f"failures `{aggregate.failures}`"
        )

    lines.extend(
        [
            "",
            "## Reviewer Check",
            "- Review the winning variant in `variants/` before promoting any policy changes into `rules.md`.",
            "- Spot-check mismatches and replay failures in `improvement.json` when the winner is close or coverage is limited.",
            "",
        ]
    )
    return "\n".join(lines)
