"""
Open Sentinel CLI entry point.

Commands:
- osentinel init: Initialize a new Open Sentinel project
- osentinel serve: Start the proxy server
- osentinel compile: Compile natural language policy to engine config
- osentinel eval: Run evaluation scenarios against a policy engine
- osentinel validate: Validate a workflow file
- osentinel info: Show workflow information
"""

import logging
import sys
from pathlib import Path

import click

from opensentinel import __version__
from opensentinel.cli_ui import (
    config_panel,
    console,
    dim,
    error,
    key_value,
    make_table,
    next_steps,
    spinner,
    success,
    warning,
    yaml_preview,
)
from rich.text import Text


def setup_logging(debug: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@click.group()
@click.version_option(version=__version__, prog_name="osentinel")
def main() -> None:
    """Open Sentinel - Reliability layer for AI agents.

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
    help="Path to osentinel.yaml config file",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging",
)
def serve(port: int, host: str, config: Path, debug: bool) -> None:
    """Start the Open Sentinel proxy server.

    The proxy intercepts LLM calls and monitors workflow adherence.
    Point your LLM client's base_url to http://HOST:PORT/v1

    Configure via osentinel.yaml or environment variables:
        osentinel serve -c osentinel.yaml
        osentinel serve --port 4000
    """
    setup_logging(debug)

    from opensentinel.config.settings import SentinelSettings
    from opensentinel.proxy.server import start_proxy

    # Gate: require osentinel.yaml (or explicit --config) before doing anything.
    # This ensures users always run `osentinel init` first.
    if config is None:
        from pathlib import Path as _Path
        _yaml_candidates = [_Path("osentinel.yaml"), _Path("osentinel.yml")]
        _env_path = __import__("os").environ.get("OSNTL_CONFIG")
        if _env_path:
            _yaml_candidates.insert(0, _Path(_env_path))
        if not any(p.is_file() for p in _yaml_candidates):
            error(
                "No osentinel.yaml found in the current directory.",
                hint="Run: osentinel init",
            )
            raise SystemExit(1)

    try:
        with spinner("Loading configuration..."):
            settings = SentinelSettings(
                _config_path=str(config) if config else None,
                debug=debug,
            )
            settings.proxy.host = host
            settings.proxy.port = port
            settings.validate()

        # Show config summary
        engine_type = settings.policy.engine.type
        display_model = settings.proxy.default_model or "(none)"
        fail_open = settings.policy.fail_open
        config_panel(
            "Open Sentinel Proxy",
            {
                "Engine": engine_type,
                "Model": display_model,
                "Host": f"{host}:{port}",
                "Fail Open": str(fail_open),
            },
        )
        console.print(
            Text.assemble(
                ("  Listening on ", ""),
                (f"http://{host}:{port}/v1", "bold cyan underline"),
            )
        )
        console.print()

    except SystemExit:
        raise
    except Exception as e:
        error(str(e), hint="Check your osentinel.yaml or run: osentinel init")
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


@main.command()
@click.option(
    "--quick",
    "-q",
    is_flag=True,
    default=False,
    help="Quick setup with sensible defaults (skip interactive wizard)",
)
def init(quick: bool) -> None:
    """Initialize a new Open Sentinel project.

    Creates an osentinel.yaml in the current directory.
    Without flags, runs an interactive wizard with arrow-key selection.

    Examples:

        # Interactive setup (default)
        osentinel init

        # Quick setup with sensible defaults
        osentinel init --quick
    """
    from opensentinel.cli_init import run_init

    run_init(quick=quick)


@main.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, path_type=Path),
)
def validate(config_path: Path) -> None:
    """Validate a workflow or osentinel.yaml configuration file.

    For osentinel.yaml: validates engine config (model, rubrics, policy).
    For workflow YAML: validates workflow structure and references.

    Examples:
        osentinel validate osentinel.yaml
        osentinel validate workflow.yaml
    """
    import yaml

    # Detect whether this is an osentinel.yaml or a workflow file
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        error(f"Failed to read {config_path}: {e}")
        raise SystemExit(1)

    if isinstance(raw, dict) and "engine" in raw:
        _validate_osentinel_config(config_path, raw)
    else:
        _validate_workflow(config_path)


def _validate_workflow(config_path: Path) -> None:
    """Validate a workflow definition file."""
    from opensentinel.policy.engines.fsm.workflow.parser import WorkflowParser

    try:
        workflow = WorkflowParser.parse_file(config_path)

        config_panel(
            "\u2713 Valid Workflow",
            {
                "Name": workflow.name,
                "States": str(len(workflow.states)),
                "Transitions": str(len(workflow.transitions)),
                "Constraints": str(len(workflow.constraints)),
            },
        )

    except FileNotFoundError:
        error(f"File not found: {config_path}")
        raise SystemExit(1)
    except Exception as e:
        error(f"Validation error: {e}")
        raise SystemExit(1)


def _validate_osentinel_config(config_path: Path, raw: dict) -> None:
    """Validate an osentinel.yaml configuration file."""
    from opensentinel.config.settings import SentinelSettings

    engine_type = raw.get("engine", "judge")

    try:
        with spinner("Loading configuration..."):
            settings = SentinelSettings(_config_path=str(config_path))
            policy_config = settings.get_policy_config()
    except Exception as e:
        error(f"Configuration error: {e}")
        raise SystemExit(1)

    engine_config = policy_config.get("config", {})

    if engine_type == "judge":
        from opensentinel.policy.engines.judge.engine import JudgePolicyEngine

        errors = JudgePolicyEngine.validate_config(engine_config)
        if errors:
            error("Judge engine configuration errors:")
            for err in errors:
                console.print(f"    [dim]{err}[/]")
            raise SystemExit(1)

        # Build summary
        models = engine_config.get("models", [])
        model_name = models[0]["model"] if models else "(none)"
        fail_action = engine_config.get("fail_action", "block")

        # Count rubrics and criteria
        from opensentinel.policy.engines.judge.rubrics import RubricRegistry

        registry = RubricRegistry()
        inline_policy = engine_config.get("inline_policy")
        if inline_policy is not None:
            temp = JudgePolicyEngine()
            temp._registry = registry
            temp._load_inline_policy(inline_policy)
            registry = temp._registry

        rubric_names = registry.list_rubrics()
        total_criteria = 0
        for name in rubric_names:
            rubric = registry.get(name)
            if rubric is not None:
                total_criteria += len(rubric.criteria)

        config_panel(
            "\u2713 Valid Configuration",
            {
                "Engine": engine_type,
                "Model": model_name,
                "Fail Action": fail_action,
                "Rubrics": str(len(rubric_names)),
                "Criteria": str(total_criteria),
            },
        )
    else:
        # For non-judge engines, just verify settings loaded
        config_panel(
            "\u2713 Valid Configuration",
            {
                "Engine": engine_type,
                "Model": str(settings.proxy.default_model or "(none)"),
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
        osentinel info workflow.yaml --verbose
    """
    from opensentinel.policy.engines.fsm.workflow.parser import WorkflowParser

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
            ("Open Sentinel", "bold"),
            (f" v{__version__}", "dim"),
        )
    )


@main.command(name="eval")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to osentinel.yaml config file",
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
    help="Show per-turn details",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging",
)
def eval_cmd(config: Path | None, json_output: Path | None, verbose: bool, debug: bool) -> None:
    """Run evaluation scenarios against a policy engine.

    Loads engine configuration and scenario paths from the `eval` section
    of osentinel.yaml, replays conversations through the engine, and
    prints a summary report.

    Examples:

        osentinel eval
        osentinel eval -c examples/fsm_workflow/osentinel.yaml
        osentinel eval --json-output results.json --verbose
    """
    import asyncio
    import glob as globmod
    import json

    import yaml

    from opensentinel.eval import EvalRunner, export_json, print_report
    from opensentinel.eval.mocks import apply_mock_provider
    from opensentinel.policy.registry import PolicyEngineRegistry

    setup_logging(debug)

    # --- Discover config file ---
    config_path = config
    if config_path is None:
        for name in ("osentinel.yaml", "osentinel.yml"):
            candidate = Path(name)
            if candidate.is_file():
                config_path = candidate
                break
        env_path = __import__("os").environ.get("OSNTL_CONFIG")
        if config_path is None and env_path:
            config_path = Path(env_path)

    if config_path is None or not config_path.is_file():
        error("No osentinel.yaml found.", hint="Run: osentinel init")
        raise SystemExit(1)

    # --- Load YAML ---
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        error(f"Failed to load config: {e}")
        raise SystemExit(1)

    eval_section = raw.get("eval")
    if not eval_section or not isinstance(eval_section, dict):
        error(
            "No 'eval' section found in config.",
            hint="Add an eval section with scenario paths to your osentinel.yaml",
        )
        raise SystemExit(1)

    # --- Resolve scenario paths (relative to config file, with glob expansion) ---
    config_dir = config_path.parent
    raw_paths = eval_section.get("scenarios", [])
    scenario_paths: list[Path] = []
    for p in raw_paths:
        expanded = globmod.glob(str(config_dir / p))
        if not expanded:
            warning(f"No files matched: {p}")
        for ep in sorted(expanded):
            scenario_paths.append(Path(ep))

    if not scenario_paths:
        error("No scenario files found.", hint="Check the 'eval.scenarios' paths in your config.")
        raise SystemExit(1)

    # --- Determine engine type and config ---
    engine_type = raw.get("engine", "judge")

    async def run_eval() -> None:
        # Build engine config from the full YAML (reuse SentinelSettings logic)
        from opensentinel.config.settings import SentinelSettings

        try:
            with spinner("Loading engine..."):
                settings = SentinelSettings(_config_path=str(config_path), debug=debug)
                policy_config = settings.get_policy_config()

                engine = await PolicyEngineRegistry.create_and_initialize(
                    policy_config["type"],
                    policy_config.get("config", {}),
                )
        except Exception as e:
            error(f"Failed to initialize engine: {e}")
            if debug:
                import traceback

                traceback.print_exc()
            raise SystemExit(1)

        # Apply mock provider if configured
        mock_cfg = eval_section.get("mock_provider", {})
        if mock_cfg and isinstance(mock_cfg, dict):
            mock_responses = mock_cfg.get("responses")
            apply_mock_provider(engine, engine_type, mock_responses)

        # Run scenarios
        runner = EvalRunner()
        dim(f"Running {len(scenario_paths)} scenario(s)...")
        console.print()

        try:
            results = await runner.run_suite(engine, scenario_paths)
        except Exception as e:
            error(f"Evaluation failed: {e}")
            if debug:
                import traceback

                traceback.print_exc()
            raise SystemExit(1)
        finally:
            await engine.shutdown()

        # Report
        print_report(results, verbose=verbose)

        # JSON export
        if json_output:
            export_data = export_json(results)
            json_output.write_text(json.dumps(export_data, indent=2))
            success(f"Results exported to {json_output}")

    asyncio.run(run_eval())


def _detect_engine_type(policy_text: str) -> str:
    """Auto-detect the best engine type from the policy text.

    Uses keyword heuristics:
    - Temporal/workflow keywords -> fsm
    - Quality/behavioral keywords -> judge
    - Default to judge if ambiguous
    """
    text_lower = policy_text.lower()

    fsm_keywords = {
        "before",
        "after",
        "first",
        "then",
        "state",
        "step",
        "workflow",
        "sequence",
        "transition",
        "phase",
        "stage",
        "proceed",
        "next",
        "previous",
        "order",
    }
    judge_keywords = {
        "ensure",
        "always",
        "never",
        "professional",
        "safe",
        "appropriate",
        "tone",
        "quality",
        "helpful",
        "accurate",
        "polite",
        "respectful",
        "pii",
        "harmful",
        "block",
        "warn",
        "evaluate",
        "score",
        "rubric",
    }

    fsm_score = sum(1 for kw in fsm_keywords if kw in text_lower)
    judge_score = sum(1 for kw in judge_keywords if kw in text_lower)

    if fsm_score > judge_score:
        return "fsm"
    return "judge"





@main.command()
@click.argument("policy", type=str)
@click.option(
    "--engine",
    "-e",
    type=click.Choice(["fsm", "judge", "nemo", "auto"]),
    default="auto",
    help="Target engine type (default: auto)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (default: auto-selected based on engine)",
)

@click.option(
    "--domain",
    "-d",
    type=str,
    help="Application domain hint (e.g., 'customer support')",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate generated output (default: True)",
)
@click.option(
    "--base-url",
    "-b",
    type=str,
    help="Base URL for LLM API",
)
@click.option(
    "--api-key",
    "-k",
    type=str,
    help="API key for LLM provider",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging",
)
def compile(
    policy: str,
    engine: str,
    output: Path,
    domain: str,
    validate: bool,
    base_url: str,
    api_key: str,
    debug: bool,
) -> None:
    """Compile natural language policy to engine configuration.

    POLICY is the natural language policy description.

    Examples:

        # Auto-detect engine type
        osentinel compile "be professional, never leak PII"

        # Explicit judge engine
        osentinel compile "never share internal info" --engine judge

        # FSM workflow
        osentinel compile "verify identity before refunds" --engine fsm
    """
    import asyncio

    setup_logging(debug)

    # Handle 'auto' engine selection
    if engine == "auto":
        engine = _detect_engine_type(policy)
        dim(f"Auto-detected engine: {engine}")

    # Determine output path based on engine type
    if output is None:
        output = Path("policy.yaml") if engine == "judge" else Path("workflow.yaml")

    # Build context
    context = {}
    if domain:
        context["domain"] = domain

    async def run_compile() -> None:
        from opensentinel.policy.compiler import PolicyCompilerRegistry
        from opensentinel.policy.registry import PolicyEngineRegistry
        from opensentinel.cli_init import ensure_model_and_key
        from opensentinel.config.settings import SentinelSettings

        try:
            # --- Resolve model (yaml > interactive) and validate API key ---
            yaml_model = None
            try:
                _settings = SentinelSettings()
                yaml_model = _settings.proxy.default_model
            except Exception:
                pass

            resolved_model, resolved_api_key = ensure_model_and_key(
                model=yaml_model,
                explicit_api_key=api_key,
                auto_confirm=True,
            )

            # --- Get compiler via engine (preferred) or registry (fallback) ---
            compiler = None
            engine_cls = PolicyEngineRegistry.get(engine)
            if engine_cls is not None:
                eng_instance = engine_cls()
                compiler = eng_instance.get_compiler(
                    model=resolved_model,
                    api_key=resolved_api_key,
                    base_url=base_url,
                )

            if compiler is None:
                compiler_class = PolicyCompilerRegistry.get(engine)
                if not compiler_class:
                    available = PolicyCompilerRegistry.list_compilers()
                    raise click.ClickException(
                        f"No compiler registered for engine type: '{engine}'. "
                        f"Available compilers: {', '.join(available) or 'none'}"
                    )
                compiler = compiler_class()
                if hasattr(compiler, "model"):
                    compiler.model = resolved_model
                if resolved_api_key and hasattr(compiler, "_api_key"):
                    compiler._api_key = resolved_api_key
                if base_url and hasattr(compiler, "_base_url"):
                    compiler._base_url = base_url

            key_value("Engine", engine)
            key_value("Model", resolved_model)
            if base_url:
                key_value("Base URL", base_url)

            with spinner("Compiling policy..."):
                result = await compiler.compile(policy, context if context else None)

            if not result.success:
                error("Compilation failed")
                for err in result.errors:
                    console.print(f"    [dim]{err}[/]")
                raise SystemExit(1)

            for w in result.warnings:
                warning(w)

            if validate:
                validation_errors = compiler.validate_result(result)
                if validation_errors:
                    error("Validation errors:")
                    for err in validation_errors:
                        console.print(f"    [dim]{err}[/]")
                    raise SystemExit(1)

            compiler.export(result, output)
            success("Compiled successfully")

            # Show summary panel
            summary: dict[str, str] = {"Output": str(output)}
            if result.metadata:
                if "state_count" in result.metadata:
                    summary["States"] = str(result.metadata["state_count"])
                if "constraint_count" in result.metadata:
                    summary["Constraints"] = str(result.metadata["constraint_count"])
                if "rubric_count" in result.metadata:
                    summary["Rubrics"] = str(result.metadata["rubric_count"])
                if "criteria_count" in result.metadata:
                    summary["Criteria"] = str(result.metadata["criteria_count"])
            config_panel("Compilation Result", summary)

            # Show yaml preview
            try:
                generated = output.read_text()
                yaml_preview(generated, title=str(output))
            except OSError:
                pass

            # Next steps
            if engine == "judge":
                next_steps(
                    [
                        f"Review the generated rubric: {output}",
                        f"Update osentinel.yaml:  policy: {output}",
                        "Start the proxy: osentinel serve",
                    ]
                )
            else:
                next_steps(
                    [
                        f"Validate: osentinel validate {output}",
                        f"Update osentinel.yaml:  engine: fsm / policy: {output}",
                        "Start the proxy: osentinel serve",
                    ]
                )

        except click.ClickException:
            raise
        except ValueError as e:
            error(str(e))
            raise SystemExit(1)
        except ImportError as e:
            error(f"Missing dependency: {e}", hint="Install with: pip install openai")
            raise SystemExit(1)

    asyncio.run(run_compile())


if __name__ == "__main__":
    main()
