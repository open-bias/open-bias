"""Smoke tests for example scripts: ensure they are valid Python and contain expected patterns."""

import ast
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _parse_example(relative_path: str) -> ast.Module:
    """Parse an example script as an AST (no execution — avoids needing API keys)."""
    path = EXAMPLES_DIR / relative_path
    if not path.exists():
        pytest.skip(f"Example not found: {path}")
    source = path.read_text()
    return ast.parse(source, filename=str(path))


def _has_function(tree: ast.Module, name: str) -> bool:
    """Check if the AST contains a top-level function with the given name."""
    return any(
        isinstance(node, ast.FunctionDef) and node.name == name
        for node in ast.walk(tree)
    )


def _has_import(tree: ast.Module, module: str) -> bool:
    """Check if the AST imports the given module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    return True
        if isinstance(node, ast.ImportFrom) and node.module == module:
            return True
    return False


class TestQuickstart:
    def test_parses(self):
        tree = _parse_example("quickstart/quickstart.py")
        assert _has_function(tree, "detect_model")
        assert _has_import(tree, "openai")

    def test_config_exists(self):
        assert (EXAMPLES_DIR / "quickstart" / "openbias.yaml").exists()


class TestJudgeSalesAgent:
    def test_parses(self):
        tree = _parse_example("judge/sales_agent.py")
        assert _has_function(tree, "detect_model")
        assert _has_import(tree, "openai")

    def test_config_exists(self):
        assert (EXAMPLES_DIR / "judge" / "openbias.yaml").exists()


class TestContentSafety:
    def test_parses(self):
        tree = _parse_example("nemo_guardrails/content_safety.py")
        assert _has_function(tree, "detect_model")
        assert _has_import(tree, "openai")

    def test_config_exists(self):
        assert (EXAMPLES_DIR / "nemo_guardrails" / "openbias.yaml").exists()
