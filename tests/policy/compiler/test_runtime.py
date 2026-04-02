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
    fake_result = CompilationResult(
        success=True,
        config={"_compiled_rules": ["Be helpful", "No secrets"], "_rules_source": "rules.md"},
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
            evaluator_config={"temperature": 0.2},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )

    assert result["_compiled_rules"] == ["Be helpful", "No secrets"]
    assert result["_rules_source"] == "rules.md"
    assert result["temperature"] == 0.2


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("evaluator_type", "compiled_config"),
    [
        (
            "judge",
            {"_compiled_rules": ["Rule one", "Rule two"], "_rules_source": "rules.md"},
        ),
        ("fsm", {"states": [{"name": "start"}]}),
        ("llm", {"steps": [{"name": "guard"}]}),
        ("nemo", {"rails": {"input": {}}}),
    ],
)
async def test_shared_runtime_flow_uses_project_rules_md_for_all_engines(
    tmp_path: Path,
    evaluator_type: str,
    compiled_config: dict,
):
    """Every registered engine compiles from project rules.md and strips legacy rule inputs."""
    (tmp_path / "rules.md").write_text("- Rule one\n- Rule two\n", encoding="utf-8")
    fake_result = CompilationResult(success=True, config=compiled_config)

    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value=fake_result)
    mock_compiler.export = MagicMock()

    with patch(
        "openbias.policy.compiler.runtime.PolicyCompilerRegistry.get",
        return_value=lambda: mock_compiler,
    ):
        result = await compile_runtime_config_for_evaluator(
            evaluator_name="shared-evaluator",
            evaluator_type=evaluator_type,
            evaluator_config={"temperature": 0.2},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )

    mock_compiler.compile.assert_awaited_once_with(
        "Rule one\nRule two",
        context=(
            {
                "simple_config": {
                    "name": "shared-evaluator",
                    "steps": ["Rule one", "Rule two"],
                    "rules": ["Rule one", "Rule two"],
                }
            }
            if evaluator_type == "fsm"
            else None
        ),
    )
    assert result["temperature"] == 0.2

    if evaluator_type == "judge":
        assert result["_compiled_rules"] == ["Rule one", "Rule two"]
        assert result["_rules_source"] == "rules.md"
    elif evaluator_type in {"fsm", "llm"}:
        assert result["workflow"] == compiled_config
    else:
        assert result["config_path"] == str(
            tmp_path / ".openbias_runtime" / "nemo" / "shared-evaluator"
        )
        mock_compiler.export.assert_called_once_with(
            fake_result,
            tmp_path / ".openbias_runtime" / "nemo" / "shared-evaluator",
        )


@pytest.mark.asyncio
async def test_runtime_preserves_unvalidated_evaluator_config_keys(tmp_path: Path):
    """Runtime compilation should not special-case legacy authored keys."""
    (tmp_path / "rules.md").write_text("Rule one", encoding="utf-8")
    fake_result = CompilationResult(
        success=True,
        config={"_compiled_rules": ["Rule one"], "_rules_source": "rules.md"},
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
            evaluator_config={"rules": ["legacy"], "rules_file": "./legacy.md"},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )

    assert result["rules"] == ["legacy"]
    assert result["rules_file"] == "./legacy.md"
    assert result["_compiled_rules"] == ["Rule one"]


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluator_type", ["judge", "fsm", "llm", "nemo"])
async def test_all_runtime_engines_require_project_rules_md(
    tmp_path: Path, evaluator_type: str
):
    """Missing rules.md should fail before any engine-specific compile step runs."""
    with pytest.raises(ValueError, match="requires project rules.md"):
        await compile_runtime_config_for_evaluator(
            evaluator_name="missing-rules",
            evaluator_type=evaluator_type,
            evaluator_config={},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )
