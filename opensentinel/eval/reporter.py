"""Rich console report and JSON export for eval results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opensentinel.eval.metrics import compute_metrics
from opensentinel.eval.runner import EvalResult
from opensentinel.policy.protocols import Decision


def print_report(results: list[EvalResult], verbose: bool = False) -> None:
    """Print a Rich console summary of eval results."""
    from opensentinel.cli_ui import config_panel, console, make_table, success, warning

    metrics = compute_metrics(results)

    # Summary panel
    config_panel(
        "Eval Summary",
        {
            "Scenarios": str(len(results)),
            "Total Turns": str(metrics.total_turns),
            "Violations": str(metrics.violation_count),
            "Interventions": str(metrics.intervention_count),
        },
    )

    # Decision breakdown
    if metrics.decisions:
        rows = [[k, str(v)] for k, v in sorted(metrics.decisions.items())]
        make_table("Decisions", ["Decision", "Count"], rows)

    # Per-scenario details
    scenario_rows: list[list[str]] = []
    for result in results:
        name = Path(result.scenario_path).name if result.scenario_path else "unknown"
        turns = str(len(result.turns))
        violations = sum(
            len(t.response_eval.metadata.get("violations", [])) for t in result.turns
        )
        status = (
            "[red]error[/]"
            if result.error
            else (f"[yellow]{violations} violations[/]" if violations else "[green]pass[/]")
        )
        scenario_rows.append([name, turns, str(violations), status])

    make_table("Scenarios", ["Scenario", "Turns", "Violations", "Status"], scenario_rows)

    # Verbose: per-turn details
    if verbose:
        for result in results:
            name = Path(result.scenario_path).name if result.scenario_path else "unknown"
            console.print(f"\n[bold]{name}[/]")
            for turn in result.turns:
                decision = turn.response_eval.decision.value
                violations_list = turn.response_eval.metadata.get("violations", [])
                v_count = len(violations_list)
                marker = "[green].[/]" if v_count == 0 else "[red]![/]"
                console.print(
                    f"  {marker} Turn {turn.turn_index}: decision={decision}, violations={v_count}"
                )
                for v in violations_list:
                    v_name = v.get("name") or v.get("constraint_id") or "unknown"
                    v_msg = v.get("message") or ""
                    console.print(f"      - {v_name}: {v_msg}")

    # Final status
    total_violations = metrics.violation_count
    errors = sum(1 for r in results if r.error)
    console.print()
    if errors:
        warning(f"{errors} scenario(s) had errors")
    if total_violations:
        warning(f"{total_violations} total violation(s) detected")
    else:
        success("All scenarios passed with no violations")


def export_json(results: list[EvalResult]) -> dict[str, Any]:
    """Export eval results as a JSON-serializable dictionary."""
    metrics = compute_metrics(results)

    scenarios: list[dict[str, Any]] = []
    for result in results:
        turns: list[dict[str, Any]] = []
        for turn in result.turns:
            turns.append(
                {
                    "turn_index": turn.turn_index,
                    "request_decision": turn.request_eval.decision.value,
                    "response_decision": turn.response_eval.decision.value,
                    "violations": turn.response_eval.metadata.get("violations", []),
                    "intervention_needed": turn.response_eval.decision in (Decision.INTERVENE, Decision.BLOCK),
                }
            )

        scenarios.append(
            {
                "scenario_path": result.scenario_path,
                "session_id": result.session_id,
                "engine_type": result.engine_type,
                "error": result.error,
                "turns": turns,
            }
        )

    return {
        "summary": {
            "total_scenarios": len(results),
            "total_turns": metrics.total_turns,
            "decisions": metrics.decisions,
            "violation_count": metrics.violation_count,
            "intervention_count": metrics.intervention_count,
        },
        "scenarios": scenarios,
    }
