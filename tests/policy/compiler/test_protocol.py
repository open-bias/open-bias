"""Tests for rules compiler protocol and registry."""

import pytest
from pathlib import Path
from typing import Any, Dict, Optional

from openbias.policy.compiler.protocol import PolicyCompiler, CompilationResult
from openbias.policy.compiler.registry import PolicyCompilerRegistry, register_compiler


class TestCompilationResult:
    """Tests for CompilationResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful compilation result."""
        result = CompilationResult(
            success=True,
            config={"name": "test"},
            warnings=["minor issue"],
            metadata={"tokens": 100},
        )

        assert result.success is True
        assert result.config == {"name": "test"}
        assert result.warnings == ["minor issue"]
        assert result.errors == []
        assert result.metadata == {"tokens": 100}

    def test_failed_result(self):
        """Test creating a failed compilation result."""
        result = CompilationResult(
            success=False,
            config=None,
            errors=["parse error"],
        )

        assert result.success is False
        assert result.config is None
        assert result.errors == ["parse error"]

    def test_failure_factory(self):
        """Test CompilationResult.failure factory method."""
        result = CompilationResult.failure(
            errors=["error 1", "error 2"],
            warnings=["warning 1"],
        )

        assert result.success is False
        assert result.config is None
        assert result.errors == ["error 1", "error 2"]
        assert result.warnings == ["warning 1"]


class MockCompiler(PolicyCompiler):
    """Mock compiler for testing."""

    @property
    def engine_type(self) -> str:
        return "mock"

    async def compile(
        self,
        rules_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> CompilationResult:
        return CompilationResult(
            success=True,
            config={"rules": rules_text},
        )

    def export(self, result: CompilationResult, output_path: Path) -> None:
        output_path.write_text(str(result.config))


class TestPolicyCompilerRegistry:
    """Tests for PolicyCompilerRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        PolicyCompilerRegistry.clear()

    def test_register_compiler(self):
        """Test registering a compiler."""
        PolicyCompilerRegistry.register("mock", MockCompiler)

        assert PolicyCompilerRegistry.is_registered("mock")
        assert PolicyCompilerRegistry.get("mock") == MockCompiler

    def test_get_unknown_compiler_returns_none(self):
        """Test getting unknown compiler returns None."""
        assert PolicyCompilerRegistry.get("unknown") is None

    def test_list_compilers(self):
        """Test listing registered compilers."""
        PolicyCompilerRegistry.register("mock1", MockCompiler)
        PolicyCompilerRegistry.register("mock2", MockCompiler)

        compilers = PolicyCompilerRegistry.list_compilers()

        assert "mock1" in compilers
        assert "mock2" in compilers

    def test_decorator_registration(self):
        """Test @register_compiler decorator."""

        @register_compiler("decorated")
        class DecoratedCompiler(MockCompiler):
            @property
            def engine_type(self) -> str:
                return "decorated"

        assert PolicyCompilerRegistry.is_registered("decorated")
        compiler_class = PolicyCompilerRegistry.get("decorated")
        assert compiler_class is not None
        compiler = compiler_class()
        assert compiler.engine_type == "decorated"


class TestPolicyCompiler:
    """Tests for PolicyCompiler base class."""

    @pytest.fixture
    def compiler(self):
        """Create a mock compiler."""
        return MockCompiler()

    async def test_compile(self, compiler):
        """Test basic compilation."""
        result = await compiler.compile("test rules")

        assert result.success is True
        assert result.config == {"rules": "test rules"}

    def test_export(self, compiler, tmp_path):
        """Test exporting compilation result."""
        result = CompilationResult(success=True, config={"test": "data"})
        output_path = tmp_path / "output.txt"

        compiler.export(result, output_path)

        assert output_path.exists()
        assert "test" in output_path.read_text()

    def test_validate_result_success(self, compiler):
        """Test validation of successful result."""
        result = CompilationResult(success=True, config={"test": "data"})

        errors = compiler.validate_result(result)

        assert errors == []

    def test_validate_result_failure(self, compiler):
        """Test validation of failed result."""
        result = CompilationResult(success=False, config=None)

        errors = compiler.validate_result(result)

        assert len(errors) > 0
        assert "not successful" in errors[0]

    def test_validate_result_no_config(self, compiler):
        """Test validation when config is None."""
        result = CompilationResult(success=True, config=None)

        errors = compiler.validate_result(result)

        assert len(errors) > 0
        assert "No config" in errors[0]
