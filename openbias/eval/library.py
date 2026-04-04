"""Helpers for repo-owned native eval suite discovery."""

from __future__ import annotations

from pathlib import Path

from openbias.eval.adapters import load_native_suite
from openbias.eval.schema import EvalSuite

_NATIVE_SUFFIXES = {".yaml", ".yml", ".json"}


def discover_native_suite_paths(root: str | Path = "evals/suites") -> list[Path]:
    """Return sorted repo-owned native suite paths."""

    root_path = Path(root)
    if not root_path.exists():
        return []

    return sorted(
        path
        for path in root_path.iterdir()
        if path.is_file() and path.suffix.lower() in _NATIVE_SUFFIXES
    )


def load_native_suites(root: str | Path = "evals/suites") -> list[EvalSuite]:
    """Load every repo-owned native suite under the given root."""

    return [load_native_suite(path) for path in discover_native_suite_paths(root)]
