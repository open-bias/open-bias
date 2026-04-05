"""Implementation of the ``openbias eval`` CLI command."""

from __future__ import annotations

import asyncio
import glob
import json
from pathlib import Path

from openbias.cli_ui import config_panel, error, key_value, spinner, success
from openbias.compare import build_engine_for_policy
from openbias.config.settings import Settings
from openbias.eval import EvalRunner, discover_native_suite_paths, load_native_suite, runtime_config_from_settings


def _resolve_eval_suite_paths(
    *,
    settings: Settings,
    config: Path | None,
    suite_paths: tuple[Path, ...],
) -> list[Path]:
    base_dir = (config.parent if config else Path.cwd()).resolve()

    if suite_paths:
        resolved: list[Path] = []
        for path in suite_paths:
            if path.is_dir():
                resolved.extend(discover_native_suite_paths(path))
            elif path.is_file():
                resolved.append(path)
        return list(dict.fromkeys(resolved))

    if settings.eval.suites:
        resolved = []
        for entry in settings.eval.suites:
            configured_path = Path(entry)
            if configured_path.exists():
                if configured_path.is_dir():
                    resolved.extend(discover_native_suite_paths(configured_path))
                elif configured_path.is_file():
                    resolved.append(configured_path)
                continue

            for match in sorted(glob.glob(entry, recursive=True)):
                match_path = Path(match)
                if match_path.is_dir():
                    resolved.extend(discover_native_suite_paths(match_path))
                elif match_path.is_file():
                    resolved.append(match_path)
        return list(dict.fromkeys(resolved))

    return discover_native_suite_paths(base_dir / "evals" / "suites")


async def _run_eval_async(
    *,
    settings: Settings,
    config: Path | None,
    suites: list[Path],
) -> list[dict[str, object]]:
    base_dir = (config.parent if config else Path.cwd()).resolve()
    rules_path = base_dir / "rules.md"
    if not rules_path.is_file():
        raise ValueError(f"Baseline rules.md not found at {rules_path}")

    engine = await build_engine_for_policy(
        settings=settings,
        config_path=config,
        rules_path=rules_path,
    )
    runner = EvalRunner(runtime=runtime_config_from_settings(settings))

    try:
        results = []
        for suite_path in suites:
            suite = load_native_suite(suite_path)
            results.append(runner.run(engine, suite))
        return [result.to_dict() for result in await asyncio.gather(*results)]
    finally:
        await engine.shutdown()


def run_eval(
    *,
    config: Path | None,
    suite_paths: tuple[Path, ...],
    json_output: Path | None,
    verbose: bool,
) -> list[dict[str, object]]:
    """Run repo-owned native eval suites against the configured policy engine."""

    with spinner("Loading configuration..."):
        settings = Settings(_config_path=str(config) if config else None)
        settings.validate()

    suites = _resolve_eval_suite_paths(
        settings=settings,
        config=config,
        suite_paths=suite_paths,
    )
    if not suites:
        error(
            "No native eval suites were found.",
            hint="Add suites under evals/suites, configure eval.suites, or pass --suite.",
        )
        raise SystemExit(1)

    with spinner("Running eval suites..."):
        results = asyncio.run(
            _run_eval_async(
                settings=settings,
                config=config,
                suites=suites,
            )
        )

    total_case_failures = 0
    total_execution_failures = 0
    for result in results:
        outcomes = result["outcomes"]
        failures = result["failures"]
        summary = result["summary"]
        case_failures = sum(1 for outcome in outcomes if not outcome["passed"])
        total_case_failures += case_failures
        total_execution_failures += len(failures)
        status = "pass" if case_failures == 0 and not failures else "fail"
        config_panel(
            f"Eval: {result['suite_name']}",
            {
                "Status": status,
                "Cases": str(len(outcomes)),
                "Case Failures": str(case_failures),
                "Execution Failures": str(len(failures)),
                "Pass Rate": f"{summary['exact_case_pass_rate']:.2%}",
                "Detection Recall": f"{summary['detection_recall']:.2%}",
                "False Positive Rate": f"{summary['false_positive_rate']:.2%}",
                "Fix Rate": f"{summary['fix_rate']:.2%}",
            },
        )
        if verbose:
            for outcome in outcomes:
                key_value(
                    outcome["case_id"],
                    (
                        f"outcome={outcome['outcome']} "
                        f"expected={outcome['expected_outcome']} "
                        f"passed={outcome['passed']}"
                    ),
                )
            for failure in failures:
                key_value(failure["case_id"], failure["error"])

    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps({"suites": results}, indent=2), encoding="utf-8")
        key_value("JSON Output", str(json_output))

    if total_case_failures or total_execution_failures:
        error(
            f"Eval completed with {total_case_failures} case failure(s) and "
            f"{total_execution_failures} execution failure(s)."
        )
        raise SystemExit(1)

    success(f"Eval passed across {len(results)} suite(s).")
    return results
