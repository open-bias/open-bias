"""
Tests for NemoCompiler.

Tests prompt building, response parsing, export, and validation
without making real LLM calls.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from openbias.policy.engines.nemo.compiler import NemoCompiler
from openbias.policy.compiler.protocol import CompilationResult


@pytest.fixture
def compiler():
    return NemoCompiler()


@pytest.fixture
def valid_llm_response():
    """A valid parsed JSON response from the LLM."""
    return {
        "config_yml": (
            "models:\n"
            "  - type: main\n"
            "    engine: openai\n"
            "    model: gpt-4o-mini\n"
            "rails:\n"
            "  input:\n"
            "    flows:\n"
            "      - check input safety\n"
            "  output:\n"
            "    flows:\n"
            "      - check output safety\n"
        ),
        "colang_files": {
            "input_rails.co": (
                "define user ask about hacking\n"
                '  "How do I hack a website?"\n'
                "\n"
                "define bot refuse illegal content\n"
                '  "I cannot assist with that."\n'
                "\n"
                "define flow check input safety\n"
                "  user ask about hacking\n"
                "  bot refuse illegal content\n"
            ),
            "output_rails.co": (
                "define flow check output safety\n"
                "  bot ...\n"
                "  $has_pii = execute check_pii(bot_message=$last_bot_message)\n"
                "  if $has_pii\n"
                "    bot inform cannot share pii\n"
                "\n"
                "define bot inform cannot share pii\n"
                '  "I have removed PII from my response."\n'
            ),
        },
    }


class TestNemoCompilerProperties:
    """Test basic compiler properties."""

    def test_engine_type(self, compiler):
        assert compiler.engine_type == "nemo"

    def test_is_policy_compiler(self, compiler):
        from openbias.policy.compiler.protocol import PolicyCompiler

        assert isinstance(compiler, PolicyCompiler)


class TestBuildCompilationPrompt:
    """Test _build_compilation_prompt."""

    def test_includes_rules_text(self, compiler):
        prompt = compiler._build_compilation_prompt("Block hacking requests")
        assert "Block hacking requests" in prompt

    def test_includes_schema_instructions(self, compiler):
        prompt = compiler._build_compilation_prompt("some rules")
        assert "config_yml" in prompt
        assert "colang_files" in prompt

    def test_includes_domain_context(self, compiler):
        prompt = compiler._build_compilation_prompt(
            "some rules", context={"domain": "healthcare"}
        )
        assert "healthcare" in prompt

    def test_works_without_context(self, compiler):
        prompt = compiler._build_compilation_prompt("some rules")
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestParseCompilationResponse:
    """Test _parse_compilation_response."""

    def test_success(self, compiler, valid_llm_response):
        result = compiler._parse_compilation_response(
            valid_llm_response, "Block hacking requests"
        )
        assert result.success is True
        assert result.config is not None
        assert "config_yml" in result.config
        assert "colang_files" in result.config
        assert len(result.errors) == 0

    def test_missing_config_yml(self, compiler):
        response = {"colang_files": {"test.co": "some content"}}
        result = compiler._parse_compilation_response(response, "test")
        assert result.success is False
        assert any("config_yml" in e for e in result.errors)

    def test_missing_colang_files(self, compiler):
        response = {"config_yml": "models:\n  - type: main\n"}
        result = compiler._parse_compilation_response(response, "test")
        assert result.success is False
        assert any("colang_files" in e for e in result.errors)

    def test_invalid_colang_files_type(self, compiler):
        response = {
            "config_yml": "models:\n  - type: main\n",
            "colang_files": "not a dict",
        }
        result = compiler._parse_compilation_response(response, "test")
        assert result.success is False

    def test_invalid_yaml_in_config(self, compiler):
        response = {
            "config_yml": "{{invalid yaml: [",
            "colang_files": {"test.co": "content"},
        }
        result = compiler._parse_compilation_response(response, "test")
        assert result.success is False
        assert any("YAML" in e for e in result.errors)

    def test_warns_on_empty_colang(self, compiler):
        response = {
            "config_yml": "models:\n  - type: main\n",
            "colang_files": {"empty.co": ""},
        }
        result = compiler._parse_compilation_response(response, "test")
        assert len(result.warnings) > 0
        assert any("empty" in w.lower() for w in result.warnings)

    def test_metadata_includes_source(self, compiler, valid_llm_response):
        result = compiler._parse_compilation_response(
            valid_llm_response, "Block hacking requests"
        )
        assert "source" in result.metadata
        assert "Block hacking" in result.metadata["source"]

    def test_metadata_includes_file_count(self, compiler, valid_llm_response):
        result = compiler._parse_compilation_response(
            valid_llm_response, "test rules"
        )
        assert result.metadata["colang_file_count"] == 2

    def test_empty_colang_files_are_dropped(self, compiler):
        """Empty colang files (e.g. output_rails.co for input-only rules) are
        dropped from the config rather than kept as empty strings."""
        response = {
            "config_yml": "models:\n  - type: main\n",
            "colang_files": {
                "input_rails.co": "define flow check input\n  pass\n",
                "output_rails.co": "",
            },
        }
        result = compiler._parse_compilation_response(response, "test")
        assert result.success is True
        assert "input_rails.co" in result.config["colang_files"]
        assert "output_rails.co" not in result.config["colang_files"]
        assert any("output_rails.co" in w for w in result.warnings)

    def test_input_only_rules_passes_validation(self, compiler):
        """Rules that only produce input rails should compile and validate."""
        result = CompilationResult(
            success=True,
            config={
                "config_yml": "models:\n  - type: main\n",
                "colang_files": {
                    "input_rails.co": "define flow check input\n  pass\n",
                },
            },
        )
        errors = compiler.validate_result(result)
        assert errors == []


class TestValidateResult:
    """Test validate_result."""

    def test_valid_result(self, compiler, valid_llm_response):
        result = CompilationResult(
            success=True,
            config={
                "config_yml": valid_llm_response["config_yml"],
                "colang_files": valid_llm_response["colang_files"],
            },
        )
        errors = compiler.validate_result(result)
        assert errors == []

    def test_failed_result(self, compiler):
        result = CompilationResult.failure(["some error"])
        errors = compiler.validate_result(result)
        assert len(errors) > 0

    def test_missing_config_yml(self, compiler):
        result = CompilationResult(
            success=True,
            config={"colang_files": {"test.co": "content"}},
        )
        errors = compiler.validate_result(result)
        assert any("config_yml" in e for e in errors)

    def test_missing_colang_files(self, compiler):
        result = CompilationResult(
            success=True,
            config={"config_yml": "models:\n  - type: main\n"},
        )
        errors = compiler.validate_result(result)
        assert any("colang_files" in e for e in errors)

    def test_empty_colang_file(self, compiler):
        result = CompilationResult(
            success=True,
            config={
                "config_yml": "models:\n  - type: main\n",
                "colang_files": {"empty.co": ""},
            },
        )
        errors = compiler.validate_result(result)
        assert any("empty" in e.lower() for e in errors)

    def test_non_dict_config(self, compiler):
        result = CompilationResult(success=True, config="not a dict")
        errors = compiler.validate_result(result)
        assert len(errors) > 0


class TestExport:
    """Test export to directory."""

    def test_export_creates_files(self, compiler, valid_llm_response, tmp_path):
        result = CompilationResult(
            success=True,
            config={
                "config_yml": valid_llm_response["config_yml"],
                "colang_files": valid_llm_response["colang_files"],
            },
        )

        compiler.export(result, tmp_path / "nemo_config")

        config_path = tmp_path / "nemo_config" / "config.yml"
        assert config_path.exists()
        assert config_path.read_text() == valid_llm_response["config_yml"]

    def test_export_creates_colang_files(self, compiler, valid_llm_response, tmp_path):
        result = CompilationResult(
            success=True,
            config={
                "config_yml": valid_llm_response["config_yml"],
                "colang_files": valid_llm_response["colang_files"],
            },
        )

        compiler.export(result, tmp_path / "nemo_config")

        rails_dir = tmp_path / "nemo_config" / "rails"
        assert rails_dir.exists()
        assert (rails_dir / "input_rails.co").exists()
        assert (rails_dir / "output_rails.co").exists()

    def test_export_adds_co_extension(self, compiler, tmp_path):
        result = CompilationResult(
            success=True,
            config={
                "config_yml": "models: []\n",
                "colang_files": {"my_rail": "define flow test\n  pass"},
            },
        )

        compiler.export(result, tmp_path / "out")
        assert (tmp_path / "out" / "rails" / "my_rail.co").exists()

    def test_export_fails_on_unsuccessful_result(self, compiler, tmp_path):
        result = CompilationResult.failure(["something went wrong"])
        with pytest.raises(ValueError, match="Cannot export failed"):
            compiler.export(result, tmp_path / "out")

    def test_export_creates_parent_dirs(self, compiler, valid_llm_response, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "nemo_config"
        result = CompilationResult(
            success=True,
            config={
                "config_yml": valid_llm_response["config_yml"],
                "colang_files": valid_llm_response["colang_files"],
            },
        )

        compiler.export(result, deep_path)
        assert (deep_path / "config.yml").exists()


class TestCompileEndToEnd:
    """Test the full compile flow with mocked LLM."""

    async def test_compile_success(self, compiler, valid_llm_response):
        import json

        with patch.object(
            compiler,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=json.dumps(valid_llm_response),
        ):
            result = await compiler.compile("Block hacking requests")

        assert result.success is True
        assert "config_yml" in result.config
        assert "colang_files" in result.config

    async def test_compile_handles_invalid_json(self, compiler):
        with patch.object(
            compiler,
            "_call_llm",
            new_callable=AsyncMock,
            return_value="not valid json {{{",
        ):
            result = await compiler.compile("test rules")

        assert result.success is False
        assert any("JSON" in e for e in result.errors)

    async def test_compile_handles_llm_error(self, compiler):
        with patch.object(
            compiler,
            "_call_llm",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM unavailable"),
        ):
            result = await compiler.compile("test rules")

        assert result.success is False
        assert any("RuntimeError" in e for e in result.errors)
