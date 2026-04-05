"""Replay-based policy improvement orchestration."""

from __future__ import annotations

from pathlib import Path

from openbias.config.settings import Settings
from openbias.improve.schema import ImprovementResult


async def run_improvement(
    *,
    settings: Settings,
    config_path: Path | None,
    trace_paths: tuple[Path, ...],
    instruction: str,
    variant_count: int,
    output_dir: Path,
) -> ImprovementResult:
    """Run the replay-based improvement flow.

    Phase 1 wires the public surface and shared orchestration contract.
    Variant generation, scoring, and artifact details are added in phase 2.
    """

    del settings, config_path, trace_paths, instruction, variant_count, output_dir
    raise NotImplementedError("Replay-based improvement is not fully implemented yet.")
