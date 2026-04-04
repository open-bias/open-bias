"""Implementation of the ``openbias review-pack`` CLI command."""

from __future__ import annotations

from pathlib import Path

from openbias.cli_ui import config_panel, key_value
from openbias.review import write_review_pack


def run_review_pack(*, comparison_path: Path, output_path: Path) -> Path:
    """Write a Markdown review package from a comparison JSON report."""

    written_path = write_review_pack(
        comparison_path=comparison_path,
        output_path=output_path,
    )
    config_panel(
        "Review Pack",
        {
            "Comparison": str(comparison_path),
            "Output": str(written_path),
        },
    )
    key_value("Review Pack", str(written_path))
    return written_path
