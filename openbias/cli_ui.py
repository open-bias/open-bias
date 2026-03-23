"""Shared CLI UI primitives for Open Bias.

Wraps Rich and questionary to provide a consistent visual identity.
All CLI files should import from here, never from rich/questionary directly.
"""

from __future__ import annotations

import sys
from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

import questionary
from questionary import Choice, Style as QStyle

# ---------------------------------------------------------------------------
# Theme & singletons
# ---------------------------------------------------------------------------

THEME = Theme(
    {
        "info": "dim",
        "warning": "yellow",
        "error": "bold red",
        "success": "green",
        "accent": "cyan",
        "heading": "bold",
        "url": "bold cyan underline",
        "key": "bold",
        "dim": "dim",
    }
)

console = Console(theme=THEME, highlight=False)

Q_STYLE = QStyle(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:cyan"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:cyan"),
        ("instruction", "fg:gray"),
    ]
)

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_MAX_WIDTH = 80

def _panel_width() -> int:
    return min(console.width, _MAX_WIDTH)

def banner(version: str) -> None:
    """Print the welcome banner — styled name + version, one-line description."""
    console.print()
    console.print(
        Text.assemble(
            ("Open Bias", "bold"),
            (f"  v{version}", "dim"),
        )
    )
    console.print("[dim]Reliability layer for AI agents[/]")
    console.print()

def heading(text: str, step: int | None = None, total: int = 5) -> None:
    """Print a section heading, optionally prefixed with step number."""
    console.print()
    if step is not None:
        console.print(
            Text.assemble(
                (f"  {step}/{total} ", "dim"),
                (text, "bold"),
            )
        )
    else:
        console.print(f"[bold]{text}[/]")

def success(msg: str) -> None:
    """Green checkmark + message."""
    console.print(f"  [green]\u2713[/] {msg}")

def error(msg: str, hint: str | None = None) -> None:
    """Red X + message, optional dim hint."""
    console.print(f"  [red]\u2717[/] {msg}", style="bold red")
    if hint:
        console.print(f"    [dim]{hint}[/]")

def warning(msg: str) -> None:
    """Yellow warning prefix + message."""
    console.print(f"  [yellow]![/] {msg}")

def dim(msg: str) -> None:
    """Print dim secondary text."""
    console.print(f"  [dim]{msg}[/]")

def key_value(key: str, value: str, indent: int = 2) -> None:
    """Print 'key: value' with bold key."""
    pad = " " * indent
    console.print(f"{pad}[bold]{key}:[/] {value}")

def next_steps(steps: list[str]) -> None:
    """Print a numbered 'Next steps' list."""
    console.print()
    console.print("[bold]Next steps:[/]")
    for i, step in enumerate(steps, 1):
        console.print(f"  {i}. {step}")
    console.print()

def yaml_preview(content: str, title: str = "") -> None:
    """Panel with syntax-highlighted YAML."""
    # Truncate to first 30 lines
    lines = content.splitlines()
    truncated = len(lines) > 30
    preview = "\n".join(lines[:30])
    if truncated:
        preview += "\n# ... truncated"

    syntax = Syntax(preview, "yaml", theme="ansi_dark", line_numbers=False)
    console.print()
    console.print(
        Panel(
            syntax,
            title=title or "YAML",
            title_align="left",
            border_style="dim",
            width=_panel_width(),
            padding=(0, 1),
        )
    )

def config_panel(title: str, items: dict[str, str]) -> None:
    """Panel showing key-value config summary."""
    text_parts: list[str] = []
    for k, v in items.items():
        text_parts.append(f"[bold]{k}:[/] {v}")
    body = "\n".join(text_parts)

    console.print()
    console.print(
        Panel(
            body,
            title=title,
            title_align="left",
            border_style="dim",
            width=_panel_width(),
            padding=(0, 1),
        )
    )

def spinner(message: str) -> Any:
    """Context manager for a loading spinner. Use only during I/O."""
    return console.status(f"  {message}", spinner="dots")

def make_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    """Build and print a Rich table."""
    table = Table(
        title=title,
        title_style="bold",
        show_header=True,
        header_style="bold dim",
        border_style="dim",
        width=_panel_width(),
        show_lines=False,
        padding=(0, 1),
    )
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*row)
    console.print()
    console.print(table)

# ---------------------------------------------------------------------------
# Input helpers (questionary wrappers)
# ---------------------------------------------------------------------------

def select(message: str, choices: list[dict[str, str]]) -> str:
    """Arrow-key select prompt. choices: [{"name": "display", "value": "val"}, ...]"""
    q_choices = [Choice(title=c["name"], value=c["value"]) for c in choices]
    result: str | None = questionary.select(
        message,
        choices=q_choices,
        style=Q_STYLE,
        instruction="(arrow keys to move, enter to select)",
    ).ask()
    if result is None:
        raise SystemExit(0)
    return result

def confirm(message: str, default: bool = True) -> bool:
    """Yes/no confirmation prompt."""
    result: bool | None = questionary.confirm(
        message,
        default=default,
        style=Q_STYLE,
    ).ask()
    if result is None:
        raise SystemExit(0)
    return result

def text(message: str, default: str = "") -> str:
    """Text input prompt."""
    result: str | None = questionary.text(
        message,
        default=default,
        style=Q_STYLE,
    ).ask()
    if result is None:
        raise SystemExit(0)
    return result

def password(message: str) -> str:
    """Hidden text input prompt."""
    result: str | None = questionary.password(
        message,
        style=Q_STYLE,
    ).ask()
    if result is None:
        raise SystemExit(0)
    return result

def is_interactive() -> bool:
    """Check if stdin is a TTY (supports interactive prompts)."""
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
