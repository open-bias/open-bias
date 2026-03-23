"""
Open Bias trigger command — send a synthetic request through the policy pipeline.

Initializes the proxy without starting an HTTP server, fires a single completion
call, and prints a rich summary of the pipeline decision.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

DEFAULT_MESSAGE = "Do androids dream of electric sheep?"

# ---------------------------------------------------------------------------
# Log collector
# ---------------------------------------------------------------------------

class TriggerLogCollector(logging.Handler):
    """Capture log records from interceptor/hooks loggers during a trigger run.

    Attach to ``openbias.core.interceptor`` and ``openbias.proxy.hooks``
    loggers before calling ``proxy.completion()``, then call ``pipeline_trace()``
    afterwards to get a structured list of hook decisions.
    """

    def __init__(self) -> None:
        super().__init__()
        self._entries: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self._entries.append(
            {
                "logger": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
                "time": record.created,
            }
        )

    def pipeline_trace(self) -> list[dict[str, str]]:
        """Parse captured log entries into a pipeline trace.

        Returns a list of dicts with keys ``phase``, ``hook``, ``decision``.
        """
        trace: list[dict[str, str]] = []

        for entry in self._entries:
            msg = entry["message"]

            # PRE_CALL decisions
            # "Request blocked by sync engine 'judge': ..."
            m = re.search(r"Request blocked by sync engine '([^']+)'", msg)
            if m:
                trace.append(
                    {"phase": "PRE_CALL", "hook": m.group(1), "decision": "BLOCK"}
                )
                continue

            # "Applying sync message replacement from 'judge'"
            # "Applying sync intervention from 'judge'"
            m = re.search(r"Applying sync (?:message replacement|intervention) from '([^']+)'", msg)
            if m:
                trace.append(
                    {"phase": "PRE_CALL", "hook": m.group(1), "decision": "INTERVENE"}
                )
                continue

            # PRE_CALL blocked from async pending result
            # "Request blocked by async engine 'judge': ..."
            m = re.search(r"Request blocked by async engine '([^']+)'", msg)
            if m:
                trace.append(
                    {"phase": "PRE_CALL", "hook": m.group(1), "decision": "BLOCK"}
                )
                continue

            # POST_CALL decisions
            # "Sync POST_CALL engine 'judge' returned INTERVENE: ..."
            m = re.search(r"Sync POST_CALL engine '([^']+)' returned INTERVENE", msg)
            if m:
                trace.append(
                    {"phase": "POST_CALL", "hook": m.group(1), "decision": "INTERVENE"}
                )
                continue

            # "Response blocked by sync engine 'judge': ..."
            m = re.search(r"Response blocked by sync engine '([^']+)'", msg)
            if m:
                trace.append(
                    {"phase": "POST_CALL", "hook": m.group(1), "decision": "BLOCK"}
                )
                continue

            # Applied POST_CALL intervention (after the fact)
            # "Applied POST_CALL intervention from 'judge' for session ..."
            m = re.search(r"Applied POST_CALL intervention from '([^']+)'", msg)
            if m:
                trace.append(
                    {"phase": "POST_CALL", "hook": m.group(1), "decision": "INTERVENE"}
                )
                continue

        # If the pipeline ran without any decision entries logged, insert
        # ALLOW entries so the output shows something meaningful.
        if not trace:
            # Try to find engine name from initialization logs
            engine_name = None
            for entry in self._entries:
                m = re.search(r"Policy engine (?:ready|initialized): (.+)", entry["message"])
                if m:
                    engine_name = m.group(1).strip()
                    break
            if engine_name:
                trace.append({"phase": "PRE_CALL", "hook": engine_name, "decision": "ALLOW"})
                trace.append({"phase": "POST_CALL", "hook": engine_name, "decision": "ALLOW"})

        return trace


# ---------------------------------------------------------------------------
# Main trigger function
# ---------------------------------------------------------------------------

async def run_trigger(
    settings: Any,
    message: str | None,
    debug: bool,
) -> None:
    """Run a synthetic request through the proxy policy pipeline.

    Args:
        settings: Loaded and validated Settings instance.
        message:  User message to send (defaults to DEFAULT_MESSAGE).
        debug:    If True, print tracebacks on error.
    """
    from openbias.cli_ui import (
        config_panel,
        console,
        dim,
        warning,
    )
    from openbias.core.utils import extract_response_content
    from openbias.proxy.server import Proxy

    resolved_message = message or DEFAULT_MESSAGE

    # ------------------------------------------------------------------
    # Show config panel
    # ------------------------------------------------------------------
    engine_type = settings.policy.engine.type
    display_model = settings.proxy.default_model or "(none)"
    fail_action = settings.policy.fail_action

    config_panel(
        "Trigger",
        {
            "Engine": engine_type,
            "Model": display_model,
            "Fail Action": fail_action,
        },
    )

    # ------------------------------------------------------------------
    # Show request
    # ------------------------------------------------------------------
    console.print()
    console.print("  [bold]Request[/]")
    console.print(f"  [dim]User:[/] {resolved_message}")

    # ------------------------------------------------------------------
    # Attach log collector to intercept pipeline decisions
    # ------------------------------------------------------------------
    collector = TriggerLogCollector()
    collector.setLevel(logging.DEBUG)

    interceptor_logger = logging.getLogger("openbias.core.interceptor")
    hooks_logger = logging.getLogger("openbias.proxy.hooks")
    interceptor_logger.addHandler(collector)
    hooks_logger.addHandler(collector)

    # ------------------------------------------------------------------
    # Build proxy and fire completion
    # ------------------------------------------------------------------
    proxy = Proxy(settings)

    try:
        await proxy.initialize()
    except Exception as exc:
        interceptor_logger.removeHandler(collector)
        hooks_logger.removeHandler(collector)
        console.print()
        console.print("  [red]\u2717[/] [bold red]Error[/]")
        console.print()
        console.print(f"  {type(exc).__name__}: {exc}")
        if debug:
            import traceback
            traceback.print_exc()
        return

    # Check pass-through mode (no policy engine configured)
    if proxy._callback is not None and proxy._callback._policy_engine is None:
        console.print()
        warning("No policy engine configured — pass-through mode")
        # Still run the completion so we can show the response
        # fall through to completion below

    messages = [{"role": "user", "content": resolved_message}]
    model = settings.proxy.default_model  # may be None, let proxy handle it

    start_time = time.monotonic()

    response_content: str | None = None
    error_occurred: Exception | None = None

    try:
        response = await proxy.completion(model=model, messages=messages)
        response_content = extract_response_content(response)
    except Exception as exc:
        error_occurred = exc
    finally:
        interceptor_logger.removeHandler(collector)
        hooks_logger.removeHandler(collector)

    end_time = time.monotonic()
    duration = end_time - start_time

    # ------------------------------------------------------------------
    # Shutdown callback
    # ------------------------------------------------------------------
    if proxy._callback is not None:
        try:
            await proxy._callback.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Print result
    # ------------------------------------------------------------------
    console.print()

    if error_occurred is not None:
        console.print(f"  [red]\u2717[/] [bold red]Error[/] ({duration:.2f}s)")
        console.print()
        console.print(f"  {type(error_occurred).__name__}: {error_occurred}")
        if debug:
            import traceback
            traceback.print_exc()
        return

    # Success
    console.print(f"  [green]\u2713[/] [bold]ALLOW[/] ({duration:.2f}s)")
    console.print()
    console.print("  [bold]Response[/]")

    if response_content:
        # Wrap long lines at ~70 chars for readability
        for line in response_content.splitlines():
            console.print(f"  {line}")
    else:
        dim("(empty response)")

    # ------------------------------------------------------------------
    # Pipeline trace
    # ------------------------------------------------------------------
    trace = collector.pipeline_trace()
    if trace:
        console.print()
        console.print("  [bold]Pipeline[/]")
        for entry in trace:
            phase = entry["phase"]
            hook = entry["hook"]
            decision = entry["decision"]

            if decision == "ALLOW":
                decision_styled = f"[green]{decision}[/]"
            elif decision == "BLOCK":
                decision_styled = f"[red]{decision}[/]"
            else:
                decision_styled = f"[yellow]{decision}[/]"

            console.print(f"  [dim]{phase:<10}[/] {hook:<20} {decision_styled}")
    console.print()
