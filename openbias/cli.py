"""
Open Bias CLI entry point.

Commands:
- openbias init: Initialize a new Open Bias project
- openbias serve: Start the proxy server
- openbias eval: Run offline native eval suites against a policy engine
- openbias validate: Validate an Open Bias configuration file
- openbias info: Show workflow information
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.text import Text

from openbias import __version__
from openbias.cli_ui import (
    config_panel,
    console,
    dim,
    error,
    make_table,
    spinner,
)
from openbias.logging import configure_logging

if TYPE_CHECKING:
    from openbias.config.settings import Settings


def _require_config(config: Path | None) -> None:
    """Require openbias.yaml exists when no explicit --config given."""
    if config is not None:
        return
    _yaml_candidates = [Path("openbias.yaml"), Path("openbias.yml")]
    _env_path = os.environ.get("OBIAS_CONFIG")
    if _env_path:
        _yaml_candidates.insert(0, Path(_env_path))
    if not any(p.is_file() for p in _yaml_candidates):
        error(
            "No openbias.yaml found in the current directory.",
            hint="Run: openbias init",
        )
        raise SystemExit(1)



@click.group()
@click.version_option(version=__version__, prog_name="openbias")
def main() -> None:
    """Open Bias - Reliability layer for AI agents.

    Monitor workflow adherence and intervene when agents deviate.
    """
    pass


@main.command()
@click.option(
    "--port",
    "-p",
    type=int,
    default=4000,
    help="Proxy server port (default: 4000)",
)
@click.option(
    "--host",
    "-h",
    type=str,
    default="0.0.0.0",
    help="Proxy server host (default: 0.0.0.0)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to openbias.yaml config file",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging",
)
@click.pass_context
def serve(ctx: click.Context, port: int, host: str, config: Path, debug: bool) -> None:
    """Start the Open Bias proxy server.

    The proxy intercepts LLM calls and monitors workflow adherence.
    Point your LLM client's base_url to http://HOST:PORT/v1

    Configure runtime settings via openbias.yaml or CLI flags:
        openbias serve -c openbias.yaml
        openbias serve --port 4000
    """
    configure_logging(debug=debug)

    from openbias.config.settings import Settings
    from openbias.proxy.server import start_proxy

    # Gate: require openbias.yaml (or explicit --config) before doing anything.
    # This ensures users always run `openbias init` first.
    _require_config(config)

    try:
        with spinner("Loading configuration..."):
            settings = Settings(
                _config_path=str(config) if config else None,
                debug=debug,
            )
            if ctx.get_parameter_source("host") == click.core.ParameterSource.COMMANDLINE:
                settings.proxy.host = host
            if ctx.get_parameter_source("port") == click.core.ParameterSource.COMMANDLINE:
                settings.proxy.port = port
            settings.validate()

        with spinner("Compiling rules for runtime engines..."):
            _compile_rules(settings, config)

        # Show config summary
        engine_type = settings.evaluators[0].type if settings.evaluators else "judge"
        display_model = settings.proxy.default_model or "(none)"
        fail_open = settings.fail_open
        config_panel(
            "Open Bias Proxy",
            {
                "Engine": engine_type,
                "Model": display_model,
                "Host": f"{settings.proxy.host}:{settings.proxy.port}",
                "Fail Open": str(fail_open),
            },
        )
        console.print(
            Text.assemble(
                ("  Listening on ", ""),
                (f"http://{settings.proxy.host}:{settings.proxy.port}/v1", "bold cyan underline"),
            )
        )
        console.print()

    except SystemExit:
        raise
    except Exception as e:
        error(str(e), hint="Check your openbias.yaml or run: openbias init")
        if debug:
            import traceback

            traceback.print_exc()
        raise SystemExit(1)

    try:
        start_proxy(settings)
    except KeyboardInterrupt:
        dim("\nShutting down...")
    except Exception as e:
        error(str(e))
        raise SystemExit(1)
async def _compile_rules_async(settings: Settings, config_path: Path | None) -> None:
    """Compile canonical rules input into engine-native evaluator configs."""
    from openbias.policy.compiler.runtime import compile_runtime_config_for_evaluator

    base_dir = (config_path.parent if config_path else Path.cwd()).resolve()
    for evaluator in settings.evaluators:
        compiled = await compile_runtime_config_for_evaluator(
            evaluator_name=evaluator.name,
            evaluator_type=evaluator.type,
            evaluator_config=dict(evaluator.config),
            default_model=settings.proxy.default_model,
            base_dir=base_dir,
        )
        evaluator.config = compiled


def _compile_rules(settings: Settings, config_path: Path | None) -> None:
    """Compile evaluator configs from synchronous CLI commands."""
    import asyncio

    asyncio.run(_compile_rules_async(settings, config_path))


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to openbias.yaml config file",
)
@click.option(
    "--message",
    "-m",
    type=str,
    default=None,
    help="Custom message to send (default: built-in test message)",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging",
)
def trigger(config: Path, message: str, debug: bool) -> None:
    """Send a synthetic request through the policy pipeline.

    Initializes the proxy without starting an HTTP server and fires a
    single completion call so you can verify your policy configuration
    end-to-end.

    Examples:

        openbias trigger
        openbias trigger --message "Tell me something interesting"
        openbias trigger -c examples/judge/openbias.yaml
    """
    import asyncio

    configure_logging(debug=debug)

    from openbias.config.settings import Settings

    # Gate: require openbias.yaml (or explicit --config) before doing anything.
    _require_config(config)

    try:
        with spinner("Loading configuration..."):
            settings = Settings(
                _config_path=str(config) if config else None,
                debug=debug,
            )
            settings.validate()

        with spinner("Compiling rules for runtime engines..."):
            _compile_rules(settings, config)

    except SystemExit:
        raise
    except Exception as e:
        error(str(e), hint="Check your openbias.yaml or run: openbias init")
        if debug:
            import traceback
            traceback.print_exc()
        raise SystemExit(1)

    from openbias.cli_trigger import run_trigger

    asyncio.run(run_trigger(settings=settings, message=message, debug=debug))


@main.command()
@click.option(
    "--quick",
    "-q",
    is_flag=True,
    default=False,
    help="Quick setup with sensible defaults (skip interactive wizard)",
)
def init(quick: bool) -> None:
    """Initialize a new Open Bias project.

    Creates an openbias.yaml in the current directory.
    Without flags, runs an interactive wizard with arrow-key selection.

    Examples:

        # Interactive setup (default)
        openbias init

        # Quick setup with sensible defaults
        openbias init --quick
    """
    from openbias.cli_init import run_init

    run_init(quick=quick)


@main.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, path_type=Path),
)
def validate(config_path: Path) -> None:
    """Validate an Open Bias configuration file.

    Examples:
        openbias validate openbias.yaml
    """
    import yaml

    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        error(f"Failed to read {config_path}: {e}")
        raise SystemExit(1)

    _validate_openbias_config(config_path, raw)



def _validate_openbias_config(config_path: Path, raw: dict) -> None:
    """Validate an openbias.yaml configuration file."""
    import asyncio

    from openbias.config.settings import Settings

    try:
        with spinner("Loading configuration..."):
            settings = Settings(_config_path=str(config_path))
    except Exception as e:
        error(f"Configuration error: {e}")
        raise SystemExit(1)

    policy_config = settings.get_policy_config()
    engine_type = policy_config.get("type", "judge")
    judge_evaluators = [ev for ev in settings.evaluators if ev.type == "judge"]

    if judge_evaluators:
        from openbias.policy.compiler.runtime import compile_runtime_config_for_evaluator
        from openbias.policy.engines.judge.engine import JudgePolicyEngine

        # Validate each judge evaluator
        all_errors: list[str] = []
        compiled_judge_configs: list[dict] = []
        for ev in judge_evaluators:
            raw_engine_config = dict(ev.config)
            # Inject default model if not explicitly set
            if not raw_engine_config.get("models") and settings.proxy.default_model:
                raw_engine_config["models"] = [
                    {"name": "primary", "model": settings.proxy.default_model}
                ]
            try:
                engine_config = asyncio.run(
                    compile_runtime_config_for_evaluator(
                        evaluator_name=ev.name,
                        evaluator_type=ev.type,
                        evaluator_config=raw_engine_config,
                        default_model=settings.proxy.default_model,
                        base_dir=config_path.parent,
                    )
                )
                compiled_judge_configs.append(engine_config)
            except Exception as e:
                all_errors.append(str(e))
                continue
            ev_errors = JudgePolicyEngine.validate_config(engine_config)
            if ev_errors:
                all_errors.extend(ev_errors)

        if all_errors:
            error("Judge engine configuration errors:")
            for err in all_errors:
                console.print(f"    [dim]{err}[/]")
            raise SystemExit(1)

        # Build summary using first judge evaluator
        first_ev_config = compiled_judge_configs[0] if compiled_judge_configs else {}
        models = first_ev_config.get("models", [])
        model_name = models[0]["model"] if models else "(none)"
        fail_action = settings.fail_action

        # Count rules from runtime-compiled judge payload.
        resolved_rules = first_ev_config.get("_compiled_rules", [])

        config_panel(
            "\u2713 Valid Configuration",
            {
                "Engine": "judge",
                "Model": model_name,
                "Fail Action": fail_action,
                "Rules": str(len(resolved_rules)),
            },
        )
    else:
        # No evaluators configured — show basic summary
        config_panel(
            "\u2713 Valid Configuration",
            {
                "Engine": engine_type,
                "Model": str(settings.proxy.default_model or "(none)"),
                "Fail Action": settings.fail_action,
            },
        )


@main.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed information",
)
def info(config_path: Path, verbose: bool) -> None:
    """Show detailed workflow information.

    Displays states, transitions, constraints, and interventions.

    Example:
        openbias info workflow.yaml --verbose
    """
    from openbias.policy.engines.fsm.workflow.parser import WorkflowParser

    try:
        workflow = WorkflowParser.parse_file(config_path)

        console.print()
        console.print(
            Text.assemble(
                (workflow.name, "bold"),
            )
        )
        if workflow.description:
            dim(workflow.description)

        # States table
        state_rows: list[list[str]] = []
        for state in workflow.states:
            flags = []
            if state.is_initial:
                flags.append("[green]initial[/]")
            if state.is_terminal:
                flags.append("[blue]terminal[/]")
            if state.is_error:
                flags.append("[red]error[/]")
            flag_str = ", ".join(flags) if flags else "-"

            desc = ""
            if verbose and state.description:
                desc = state.description
            state_rows.append([state.name, flag_str, desc])

        columns = ["Name", "Type", "Description"] if verbose else ["Name", "Type"]
        rows = [r if verbose else r[:2] for r in state_rows]
        make_table("States", columns, rows)

        # Transitions table
        if workflow.transitions:
            t_rows = [[t.from_state, f"\u2192 {t.to_state}"] for t in workflow.transitions]
            make_table("Transitions", ["From", "To"], t_rows)

        # Constraints table
        if workflow.constraints:
            c_rows: list[list[str]] = []
            for c in workflow.constraints:
                row = [c.name, f"[cyan]{c.type.value}[/]"]
                if verbose:
                    row.append(c.message or c.description or "")
                c_rows.append(row)

            cols = ["Name", "Type"]
            if verbose:
                cols.append("Message")
            make_table("Constraints", cols, c_rows)

        console.print()

    except Exception as e:
        error(str(e))
        raise SystemExit(1)


@main.command()
def version() -> None:
    """Show version information."""
    console.print(
        Text.assemble(
            ("Open Bias", "bold"),
            (f" v{__version__}", "dim"),
        )
    )


@main.command(name="eval")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to openbias.yaml config file",
)
@click.option(
    "--suite",
    "suite_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    default=(),
    help="Native eval suite file or directory (repeatable); defaults to repo-owned suites",
)
@click.option(
    "--json-output",
    type=click.Path(path_type=Path),
    default=None,
    help="Export results to JSON file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show per-case outcomes",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging",
)
def eval_cmd(
    config: Path | None,
    suite_paths: tuple[Path, ...],
    json_output: Path | None,
    verbose: bool,
    debug: bool,
) -> None:
    """Run offline native eval suites against the configured policy engine."""
    configure_logging(debug=debug)
    _require_config(config)

    from openbias.cli_eval import run_eval

    run_eval(
        config=config,
        suite_paths=suite_paths,
        json_output=json_output,
        verbose=verbose,
    )


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to openbias.yaml config file",
)
@click.option(
    "--trace",
    "trace_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
    help="Path to a replayable trace JSONL dataset (repeatable)",
)
@click.option(
    "--json-output",
    type=click.Path(path_type=Path),
    default=None,
    help="Export replay results to JSON file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show per-case expected vs observed actions",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging",
)
def replay(
    config: Path | None,
    trace_paths: tuple[Path, ...],
    json_output: Path | None,
    verbose: bool,
    debug: bool,
) -> None:
    """Replay trace datasets against the configured policy engine."""
    configure_logging(debug=debug)
    _require_config(config)

    from openbias.cli_replay import run_replay

    run_replay(
        config=config,
        trace_paths=trace_paths,
        json_output=json_output,
        verbose=verbose,
        debug=debug,
    )


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to openbias.yaml config file",
)
@click.option(
    "--trace",
    "trace_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
    help="Replayable trace JSONL dataset (repeatable)",
)
@click.option(
    "--instruction",
    type=str,
    required=True,
    help="How the generated policy variants should differ from baseline rules.md",
)
@click.option(
    "--variant-count",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
    help="Number of generated candidate policy variants to evaluate alongside baseline",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path(".openbias/reports/latest"),
    help="Directory for improvement.json, improvement.md, and generated variants",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging",
)
def improve(
    config: Path | None,
    trace_paths: tuple[Path, ...],
    instruction: str,
    variant_count: int,
    output_dir: Path,
    debug: bool,
) -> None:
    """Generate and replay policy variants, then recommend one for review."""
    configure_logging(debug=debug)
    _require_config(config)

    from openbias.cli_improve import run_improve

    run_improve(
        config=config,
        trace_paths=trace_paths,
        instruction=instruction,
        variant_count=variant_count,
        output_dir=output_dir,
    )


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to openbias.yaml config file",
)
@click.option(
    "--candidate",
    "candidate_policy_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the candidate rules markdown file",
)
@click.option(
    "--trace",
    "trace_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    default=(),
    help="Optional replayable trace JSONL dataset (repeatable)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path(".openbias/reports/latest"),
    help="Directory for comparison.json and comparison.md",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging",
)
def compare(
    config: Path | None,
    candidate_policy_path: Path,
    trace_paths: tuple[Path, ...],
    output_dir: Path,
    debug: bool,
) -> None:
    """Compare baseline rules.md against a candidate policy file."""
    configure_logging(debug=debug)
    _require_config(config)

    from openbias.cli_compare import run_compare

    run_compare(
        config=config,
        candidate_policy_path=candidate_policy_path,
        trace_paths=trace_paths,
        output_dir=output_dir,
    )


@main.command(name="review-pack")
@click.option(
    "--comparison",
    "comparison_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to comparison.json",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=Path(".openbias/reports/latest/review-pack.md"),
    help="Output path for the reviewer-facing Markdown package",
)
def review_pack(comparison_path: Path, output_path: Path) -> None:
    """Generate a reviewer-friendly Markdown package from comparison.json."""
    from openbias.cli_review import run_review_pack

    run_review_pack(
        comparison_path=comparison_path,
        output_path=output_path,
    )
if __name__ == "__main__":
    main()
