"""Implementation of the ``openbias replay`` CLI command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from openbias.cli_ui import config_panel, console, error, key_value, spinner
from openbias.config.settings import Settings
from openbias.policy.registry import PolicyEngineRegistry
from openbias.replay import ReplayRunner
from openbias.traces import load_trace_dataset


async def _run_replay_async(
    *,
    settings: Settings,
    trace_paths: tuple[Path, ...],
) -> list[dict[str, Any]]:
    policy_config = settings.get_policy_config()
    engine = await PolicyEngineRegistry.create_and_initialize(
        policy_config["type"],
        policy_config["config"],
    )
    runner = ReplayRunner(fail_action=settings.fail_action)
    results: list[dict[str, Any]] = []

    try:
        for path in trace_paths:
            dataset = load_trace_dataset(path)
            result = await runner.run(engine, dataset)
            results.append(result.to_dict())
    finally:
        await engine.shutdown()

    return results


def run_replay(
    *,
    config: Path | None,
    trace_paths: tuple[Path, ...],
    json_output: Path | None,
    verbose: bool,
    debug: bool,
) -> None:
    """Execute trace replay and print a summary."""

    del debug  # logging is configured by the CLI entrypoint

    with spinner("Loading configuration..."):
        settings = Settings(_config_path=str(config) if config else None)
        settings.validate()

    from openbias.cli import _compile_rules

    with spinner("Compiling runtime policy..."):
        _compile_rules(settings, config)

    with spinner("Replaying trace datasets..."):
        results = asyncio.run(
            _run_replay_async(settings=settings, trace_paths=trace_paths)
        )

    for result in results:
        summary = result["summary"]
        config_panel(
            f"Replay: {result['dataset_name']}",
            {
                "Cases": str(summary["total_cases"]),
                "Supported": str(summary["supported_cases"]),
                "Matched": str(summary["matched_cases"]),
                "Intervene Rate": f"{summary['intervention_rate']:.2%}",
                "Block Rate": f"{summary['block_rate']:.2%}",
                "Pass-through Rate": f"{summary['pass_through_rate']:.2%}",
            },
        )
        if verbose:
            for outcome in result["outcomes"]:
                key_value(
                    outcome["case_id"],
                    f"expected={outcome['expected_action']} observed={outcome['observed_action']}",
                )

    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps({"datasets": results}, indent=2), encoding="utf-8")
        console.print()
        key_value("JSON Output", str(json_output))

    failures = sum(len(result["failures"]) for result in results)
    if failures:
        error(f"Replay completed with {failures} failure(s).")
