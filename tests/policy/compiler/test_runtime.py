"""Tests for compile_runtime_config_for_evaluator (serve-time compilation)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbias.policy.compiler.protocol import CompilationResult
from openbias.policy.compiler.runtime import compile_runtime_config_for_evaluator


# ---------------------------------------------------------------------------
# Judge engine runtime config normalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_compiles_from_project_rules_md_only(tmp_path: Path):
    """Judge receives internal compiled rules from project rules.md."""
    (tmp_path / "rules.md").write_text("- Be helpful\n- No secrets\n", encoding="utf-8")
    result = await compile_runtime_config_for_evaluator(
        evaluator_name="behavior",
        evaluator_type="judge",
        evaluator_config={"rules": ["ignored"], "rules_file": "./ignored.md"},
        default_model="gpt-4o-mini",
        base_dir=tmp_path,
    )
    assert result["_compiled_rules"] == ["Be helpful", "No secrets"]
    assert result["_rules_source"] == "rules.md"
    assert "rules" not in result
    assert "rules_file" not in result


@pytest.mark.asyncio
async def test_judge_uses_registered_compiler_and_merges_result(tmp_path: Path):
    """Judge follows the shared runtime compiler registry flow."""
    (tmp_path / "rules.md").write_text("- Guard secrets\n- Stay on task\n", encoding="utf-8")
    fake_result = CompilationResult(
        success=True,
        config={"_compiled_rules": ["Guard secrets", "Stay on task"], "_rules_source": "rules.md"},
    )

    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value=fake_result)

    with patch(
        "openbias.policy.compiler.runtime.PolicyCompilerRegistry.get",
        return_value=lambda: mock_compiler,
    ):
        result = await compile_runtime_config_for_evaluator(
            evaluator_name="behavior",
            evaluator_type="judge",
            evaluator_config={},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )

    assert result["_compiled_rules"] == ["Guard secrets", "Stay on task"]
    assert result["_rules_source"] == "rules.md"
    mock_compiler.compile.assert_awaited_once_with(
        "Guard secrets\nStay on task",
        context=None,
    )


@pytest.mark.asyncio
async def test_judge_without_project_rules_md_fails(tmp_path: Path):
    """Judge fails fast when project rules.md is missing."""
    with pytest.raises(ValueError, match="requires project rules.md"):
        await compile_runtime_config_for_evaluator(
            evaluator_name="safety",
            evaluator_type="judge",
            evaluator_config={"models": [{"name": "primary", "model": "gpt-4o-mini"}]},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# FSM engine compilation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fsm_compiles_with_workflow_context(tmp_path: Path):
    """FSM compilation passes simple_config context and stores workflow result."""
    (tmp_path / "rules.md").write_text("Step 1: greet user", encoding="utf-8")
    fake_workflow = {"name": "test", "states": []}
    fake_result = CompilationResult(success=True, config=fake_workflow)

    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value=fake_result)

    with patch(
        "openbias.policy.compiler.runtime.PolicyCompilerRegistry.get",
        return_value=lambda: mock_compiler,
    ):
        result = await compile_runtime_config_for_evaluator(
            evaluator_name="workflow",
            evaluator_type="fsm",
            evaluator_config={},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )

    assert result["workflow"] == fake_workflow
    assert "rules" not in result
    # Verify compilation context included simple_config
    call_kwargs = mock_compiler.compile.call_args
    assert call_kwargs[1]["context"]["simple_config"]["name"] == "workflow"


# ---------------------------------------------------------------------------
# NeMo engine compilation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nemo_compiles_and_exports(tmp_path: Path):
    """NeMo compilation exports to runtime dir and stores config_path."""
    (tmp_path / "rules.md").write_text("Always be safe", encoding="utf-8")
    fake_result = CompilationResult(success=True, config={"rails": {}})

    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value=fake_result)
    mock_compiler.export = MagicMock()

    with patch(
        "openbias.policy.compiler.runtime.PolicyCompilerRegistry.get",
        return_value=lambda: mock_compiler,
    ):
        result = await compile_runtime_config_for_evaluator(
            evaluator_name="nemo-rails",
            evaluator_type="nemo",
            evaluator_config={},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )

    assert "config_path" in result
    assert "nemo-rails" in result["config_path"]
    mock_compiler.export.assert_called_once()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compilation_failure_raises(tmp_path: Path):
    """A failed compilation raises ValueError with the evaluator name."""
    (tmp_path / "rules.md").write_text("Be safe", encoding="utf-8")
    with patch(
        "openbias.policy.compiler.runtime.PolicyCompilerRegistry.get",
        return_value=None,
    ), pytest.raises(ValueError, match="No compiler registered"):
        await compile_runtime_config_for_evaluator(
            evaluator_name="safety",
            evaluator_type="unsupported",
            evaluator_config={},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_no_registered_compiler_raises(tmp_path: Path):
    """Raises when no compiler is registered for the evaluator type."""
    (tmp_path / "rules.md").write_text("Some rule", encoding="utf-8")
    with patch(
        "openbias.policy.compiler.runtime.PolicyCompilerRegistry.get",
        return_value=None,
    ):
        with pytest.raises(ValueError, match="No compiler registered"):
            await compile_runtime_config_for_evaluator(
                evaluator_name="unknown",
                evaluator_type="unsupported",
                evaluator_config={},
                default_model="gpt-4o-mini",
                base_dir=tmp_path,
            )


@pytest.mark.asyncio
async def test_fallback_to_compiler_registry(tmp_path: Path):
    """Compiler registry fallback still works with project rules.md."""
    (tmp_path / "rules.md").write_text("Be safe", encoding="utf-8")
    fake_result = CompilationResult(success=True, config={"compiled": True})
    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value=fake_result)

    with patch(
        "openbias.policy.compiler.runtime.PolicyCompilerRegistry.get",
        return_value=lambda: mock_compiler,
    ):
        result = await compile_runtime_config_for_evaluator(
            evaluator_name="safety",
            evaluator_type="llm",
            evaluator_config={},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )

    assert result["workflow"] == {"compiled": True}
