"""Implementation of the ``openbias improve`` CLI command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from openbias.cli_ui import config_panel, key_value, spinner
from openbias.config.settings import Settings
from openbias.improve import run_improvement


def run_improve(
    *,
    config: Path | None,
    trace_paths: tuple[Path, ...],
    instruction: str,
    variant_count: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Run replay-based policy improvement and persist the artifacts."""

    with spinner("Loading configuration..."):
        settings = Settings(_config_path=str(config) if config else None)
        settings.validate()

    result = asyncio.run(
        run_improvement(
            settings=settings,
            config_path=config,
            trace_paths=trace_paths,
            instruction=instruction,
            variant_count=variant_count,
            output_dir=output_dir,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "improvement.json"
    md_path = output_dir / "improvement.md"
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text("", encoding="utf-8")

    config_panel(
        "Policy Improvement",
        {
            "Status": result.status,
            "Trace Datasets": str(len(trace_paths)),
            "Variants": str(len(result.variants)),
        },
    )
    key_value("Winner", result.winner_variant_id or "(none)")
    key_value("JSON Output", str(json_path))
    key_value("Markdown Output", str(md_path))
    return json_path, md_path
