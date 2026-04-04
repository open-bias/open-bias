"""Implementation of the ``openbias compare`` CLI command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from openbias.cli_ui import config_panel, console, key_value, spinner
from openbias.compare import compare_policy_runs, render_comparison_markdown
from openbias.config.settings import Settings


def run_compare(
    *,
    config: Path | None,
    candidate_policy_path: Path,
    trace_paths: tuple[Path, ...],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Run baseline-vs-candidate comparison and persist the reports."""

    with spinner("Loading configuration..."):
        settings = Settings(_config_path=str(config) if config else None)
        settings.validate()

    result = asyncio.run(
        compare_policy_runs(
            settings=settings,
            config_path=config,
            candidate_policy_path=candidate_policy_path,
            trace_paths=trace_paths,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    md_path = output_dir / "comparison.md"
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(render_comparison_markdown(result), encoding="utf-8")

    config_panel(
        "Policy Comparison",
        {
            "Status": result.status,
            "Suites": str(len(result.suites)),
            "Trace Datasets": str(len(result.traces)),
        },
    )
    key_value("JSON Output", str(json_path))
    key_value("Markdown Output", str(md_path))
    return json_path, md_path
