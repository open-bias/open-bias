"""Tests for compile_runtime_config_for_evaluator (serve-time compilation)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbias.policy.compiler.protocol import CompilationResult
from openbias.policy.compiler.runtime import compile_runtime_config_for_evaluator


# ---------------------------------------------------------------------------
# Judge engine compilation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_compiles_inline_rules(tmp_path: Path):
    """Inline rules are compiled and placed into inline_rules config key."""
    fake_result = CompilationResult(
        success=True,
        config={"rubrics": [{"name": "safety", "criteria": [{"name": "c1"}]}]},
    )
    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value=fake_result)

    mock_engine_cls = MagicMock()
    mock_engine_cls.return_value.get_compiler.return_value = mock_compiler

    with patch(
        "openbias.policy.compiler.runtime.PolicyEngineRegistry.get",
        return_value=mock_engine_cls,
    ):
        result = await compile_runtime_config_for_evaluator(
            evaluator_name="safety",
            evaluator_type="judge",
            evaluator_config={"rules": ["Be professional", "No PII"]},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )

    assert "inline_rules" in result
    assert result["inline_rules"] == [{"name": "safety", "criteria": [{"name": "c1"}]}]
    assert "rules" not in result


@pytest.mark.asyncio
async def test_judge_compiles_from_rules_file(tmp_path: Path):
    """Rules loaded from rules.md file are compiled for the judge engine."""
    rules_file = tmp_path / "rules.md"
    rules_file.write_text("- Be helpful\n- No secrets\n")

    fake_result = CompilationResult(
        success=True,
        config={"rubrics": [{"name": "helpfulness", "criteria": [{"name": "c1"}]}]},
    )
    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value=fake_result)

    mock_engine_cls = MagicMock()
    mock_engine_cls.return_value.get_compiler.return_value = mock_compiler

    with patch(
        "openbias.policy.compiler.runtime.PolicyEngineRegistry.get",
        return_value=mock_engine_cls,
    ):
        result = await compile_runtime_config_for_evaluator(
            evaluator_name="behavior",
            evaluator_type="judge",
            evaluator_config={"rules_file": str(rules_file)},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )

    assert "inline_rules" in result
    assert "rules_file" not in result
    mock_compiler.compile.assert_called_once()


@pytest.mark.asyncio
async def test_judge_auto_discovers_rules_md(tmp_path: Path):
    """When no explicit rules are set, rules.md is auto-discovered."""
    (tmp_path / "rules.md").write_text("Auto-discovered rule")

    fake_result = CompilationResult(
        success=True,
        config={"rubrics": [{"name": "auto", "criteria": [{"name": "c1"}]}]},
    )
    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value=fake_result)

    mock_engine_cls = MagicMock()
    mock_engine_cls.return_value.get_compiler.return_value = mock_compiler

    with patch(
        "openbias.policy.compiler.runtime.PolicyEngineRegistry.get",
        return_value=mock_engine_cls,
    ):
        result = await compile_runtime_config_for_evaluator(
            evaluator_name="safety",
            evaluator_type="judge",
            evaluator_config={},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )

    assert "inline_rules" in result
    mock_compiler.compile.assert_called_once()


@pytest.mark.asyncio
async def test_no_rules_returns_config_unchanged(tmp_path: Path):
    """When there are no rules to compile, the original config is returned."""
    original = {"models": [{"name": "primary", "model": "gpt-4o-mini"}]}

    result = await compile_runtime_config_for_evaluator(
        evaluator_name="safety",
        evaluator_type="judge",
        evaluator_config=dict(original),
        default_model="gpt-4o-mini",
        base_dir=tmp_path,
    )

    assert result == original


# ---------------------------------------------------------------------------
# FSM engine compilation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fsm_compiles_with_workflow_context(tmp_path: Path):
    """FSM compilation passes simple_config context and stores workflow result."""
    fake_workflow = {"name": "test", "states": []}
    fake_result = CompilationResult(success=True, config=fake_workflow)

    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value=fake_result)

    mock_engine_cls = MagicMock()
    mock_engine_cls.return_value.get_compiler.return_value = mock_compiler

    with patch(
        "openbias.policy.compiler.runtime.PolicyEngineRegistry.get",
        return_value=mock_engine_cls,
    ):
        result = await compile_runtime_config_for_evaluator(
            evaluator_name="workflow",
            evaluator_type="fsm",
            evaluator_config={"rules": ["Step 1: greet user", "Step 2: collect info"]},
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
    fake_result = CompilationResult(success=True, config={"rails": {}})

    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value=fake_result)
    mock_compiler.export = MagicMock()

    mock_engine_cls = MagicMock()
    mock_engine_cls.return_value.get_compiler.return_value = mock_compiler

    with patch(
        "openbias.policy.compiler.runtime.PolicyEngineRegistry.get",
        return_value=mock_engine_cls,
    ):
        result = await compile_runtime_config_for_evaluator(
            evaluator_name="nemo-rails",
            evaluator_type="nemo",
            evaluator_config={"rules": ["Always be safe"]},
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
    fake_result = CompilationResult(
        success=False,
        config=None,
        errors=["LLM returned invalid JSON"],
    )
    mock_compiler = AsyncMock()
    mock_compiler.compile = AsyncMock(return_value=fake_result)

    mock_engine_cls = MagicMock()
    mock_engine_cls.return_value.get_compiler.return_value = mock_compiler

    with patch(
        "openbias.policy.compiler.runtime.PolicyEngineRegistry.get",
        return_value=mock_engine_cls,
    ):
        with pytest.raises(ValueError, match="Failed to compile rules for evaluator 'safety'"):
            await compile_runtime_config_for_evaluator(
                evaluator_name="safety",
                evaluator_type="judge",
                evaluator_config={"rules": ["Be safe"]},
                default_model="gpt-4o-mini",
                base_dir=tmp_path,
            )


@pytest.mark.asyncio
async def test_no_registered_compiler_raises(tmp_path: Path):
    """Raises when no compiler is registered for the evaluator type."""
    with patch(
        "openbias.policy.compiler.runtime.PolicyEngineRegistry.get",
        return_value=None,
    ), patch(
        "openbias.policy.compiler.runtime.PolicyCompilerRegistry.get",
        return_value=None,
    ):
        with pytest.raises(ValueError, match="No compiler registered"):
            await compile_runtime_config_for_evaluator(
                evaluator_name="unknown",
                evaluator_type="unsupported",
                evaluator_config={"rules": ["Some rule"]},
                default_model="gpt-4o-mini",
                base_dir=tmp_path,
            )


@pytest.mark.asyncio
async def test_fallback_to_compiler_registry(tmp_path: Path):
    """When engine has no get_compiler, falls back to PolicyCompilerRegistry."""
    fake_result = CompilationResult(success=True, config={"rubrics": []})

    mock_compiler_instance = AsyncMock()
    mock_compiler_instance.compile = AsyncMock(return_value=fake_result)
    mock_compiler_instance.model = None

    mock_compiler_cls = MagicMock(return_value=mock_compiler_instance)

    with patch(
        "openbias.policy.compiler.runtime.PolicyEngineRegistry.get",
        return_value=None,
    ), patch(
        "openbias.policy.compiler.runtime.PolicyCompilerRegistry.get",
        return_value=mock_compiler_cls,
    ):
        result = await compile_runtime_config_for_evaluator(
            evaluator_name="safety",
            evaluator_type="judge",
            evaluator_config={"rules": ["Be safe"]},
            default_model="gpt-4o-mini",
            base_dir=tmp_path,
        )

    assert mock_compiler_instance.model == "gpt-4o-mini"
    mock_compiler_instance.compile.assert_called_once()
