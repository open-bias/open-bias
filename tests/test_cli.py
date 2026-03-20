"""Tests for opensentinel.cli commands."""

import sys
from io import StringIO
from unittest.mock import patch, MagicMock
from pathlib import Path

from click.testing import CliRunner

from opensentinel.cli import main


def _invoke(args):
    """Invoke CLI and capture both Click output and Rich stdout."""
    runner = CliRunner()
    # Rich writes to sys.stdout; CliRunner captures click.echo output.
    # We need to capture both.
    buf = StringIO()
    from opensentinel.cli_ui import console

    old_file = console.file
    console.file = buf
    try:
        result = runner.invoke(main, args)
    finally:
        console.file = old_file
    combined = result.output + buf.getvalue()
    return result, combined


class TestVersionCommand:
    def test_version_output(self):
        result, output = _invoke(["version"])
        assert result.exit_code == 0
        assert "Open Sentinel" in output

    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "osentinel" in result.output


class TestValidateCommand:
    def test_validate_missing_file(self):
        result, _ = _invoke(["validate", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_validate_valid_workflow(self):
        """Test validate with a mock workflow parser."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.yaml").write_text("name: test\nversion: '1.0'")

            mock_workflow = MagicMock()
            mock_workflow.name = "Test"
            mock_workflow.version = "1.0"
            mock_workflow.states = []
            mock_workflow.transitions = []
            mock_workflow.constraints = []
            mock_workflow.interventions = {}

            buf = StringIO()
            from opensentinel.cli_ui import console

            old_file = console.file
            console.file = buf
            try:
                with patch(
                    "opensentinel.policy.engines.fsm.workflow.parser.WorkflowParser.parse_file",
                    return_value=mock_workflow,
                ):
                    result = runner.invoke(main, ["validate", "test.yaml"])
            finally:
                console.file = old_file

            combined = result.output + buf.getvalue()
            assert result.exit_code == 0
            assert "Valid Workflow" in combined

    def test_validate_good_judge_config(self):
        """validate with a valid osentinel.yaml should show summary."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("osentinel.yaml").write_text(
                "engine: judge\n"
                "model: gpt-4o-mini\n"
                "policy:\n"
                '  - "Be professional"\n'
                '  - "No PII"\n'
                "tracing:\n  type: none\n"
            )

            buf = StringIO()
            from opensentinel.cli_ui import console

            old_file = console.file
            console.file = buf
            try:
                result = runner.invoke(main, ["validate", "osentinel.yaml"])
            finally:
                console.file = old_file

            combined = result.output + buf.getvalue()
            assert result.exit_code == 0
            assert "Valid Configuration" in combined

    @patch.dict("os.environ", {}, clear=True)
    def test_validate_judge_config_no_model(self):
        """validate with no model should fail with clear error."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("osentinel.yaml").write_text(
                "engine: judge\n"
                "policy:\n"
                '  - "Be professional"\n'
                "tracing:\n  type: none\n"
            )

            buf = StringIO()
            from opensentinel.cli_ui import console

            old_file = console.file
            console.file = buf
            try:
                result = runner.invoke(main, ["validate", "osentinel.yaml"])
            finally:
                console.file = old_file

            combined = result.output + buf.getvalue()
            assert result.exit_code != 0
            assert "No model configured" in combined

    def test_validate_judge_config_bad_rubric(self):
        """validate with a nonexistent default rubric should fail."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("osentinel.yaml").write_text(
                "engine: judge\n"
                "model: gpt-4o-mini\n"
                "judge:\n"
                "  default_rubric: nonexistent_rubric\n"
                "tracing:\n  type: none\n"
            )

            buf = StringIO()
            from opensentinel.cli_ui import console

            old_file = console.file
            console.file = buf
            try:
                result = runner.invoke(main, ["validate", "osentinel.yaml"])
            finally:
                console.file = old_file

            combined = result.output + buf.getvalue()
            assert result.exit_code != 0
            assert "nonexistent_rubric" in combined


class TestInitCommand:
    def test_init_no_interactive_flag(self):
        """Verify -i flag was removed."""
        result, _ = _invoke(["init", "-i"])
        assert result.exit_code != 0

    def test_init_quick_flag_exists(self):
        """Verify --quick flag works."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Mock run_init since it might try to write files or check env
            with patch("opensentinel.cli_init.run_init") as mock_run:
                result = runner.invoke(main, ["init", "--quick"])
                assert result.exit_code == 0
                mock_run.assert_called_once_with(quick=True)

    def test_init_non_tty_without_from(self):
        """Without --from and without TTY, should show error."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("opensentinel.cli_init.is_interactive", return_value=False):
                result = runner.invoke(main, ["init"])
                assert result.exit_code != 0


class TestServeCommand:
    def test_serve_missing_config(self):
        result, _ = _invoke(["serve", "--config", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_serve_no_yaml_prompts_init(self):
        """serve without an osentinel.yaml should tell user to run osentinel init."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            buf = StringIO()
            from opensentinel.cli_ui import console
            old_file = console.file
            console.file = buf
            try:
                result = runner.invoke(main, ["serve"])
            finally:
                console.file = old_file
            combined = result.output + buf.getvalue()
            assert result.exit_code != 0
            assert "osentinel init" in combined

    def test_serve_with_yaml_proceeds(self):
        """serve with an osentinel.yaml should pass the gate and attempt startup."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Write a minimal valid yaml
            Path("osentinel.yaml").write_text(
                "engine: judge\nmodel: gpt-4o-mini\nport: 4000\n"
                "policy:\n  fail_open: true\ntracing:\n  type: none\n"
            )
            # Mock start_proxy so we don't actually start a server
            with patch("opensentinel.proxy.server.start_proxy"):
                with patch("opensentinel.config.settings.SentinelSettings.validate"):
                    result = runner.invoke(main, ["serve"])
            # Should NOT fail with the init-gate error
            combined = result.output
            assert "osentinel init" not in combined or result.exit_code == 0


class TestCompileCommand:
    def test_compile_help(self):
        result, _ = _invoke(["compile", "--help"])
        assert result.exit_code == 0
        assert "POLICY" in result.output


    def test_compile_reads_model_from_yaml(self):
        """compile should use model from osentinel.yaml when --model is not given."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("osentinel.yaml").write_text("engine: judge\nmodel: gpt-4o\n")

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.errors = []
            mock_result.warnings = []
            mock_result.metadata = {}

            mock_compiler = MagicMock()
            mock_compiler.model = None
            mock_compiler.compile = MagicMock(return_value=mock_result)
            mock_compiler.validate_result = MagicMock(return_value=[])
            mock_compiler.export = MagicMock()

            mock_compiler_class = MagicMock(return_value=mock_compiler)

            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
                with patch("opensentinel.policy.registry.PolicyEngineRegistry.get", return_value=None):
                    with patch("opensentinel.policy.compiler.PolicyCompilerRegistry.get", return_value=mock_compiler_class):
                        result = runner.invoke(main, ["compile", "be professional"])

            # Model should have been set from yaml (gpt-4o), not gpt-4o-mini
            assert mock_compiler.model == "gpt-4o"

    def test_compile_no_yaml_calls_ensure_model(self):
        """compile with no yaml and no --model should call ensure_model_and_key."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.errors = []
            mock_result.warnings = []
            mock_result.metadata = {}

            mock_compiler = MagicMock()
            mock_compiler.model = None
            mock_compiler.compile = MagicMock(return_value=mock_result)
            mock_compiler.validate_result = MagicMock(return_value=[])
            mock_compiler.export = MagicMock()

            mock_compiler_class = MagicMock(return_value=mock_compiler)

            with patch("opensentinel.policy.registry.PolicyEngineRegistry.get", return_value=None):
                with patch("opensentinel.policy.compiler.PolicyCompilerRegistry.get", return_value=mock_compiler_class):
                    with patch("opensentinel.cli_init.ensure_model_and_key", return_value=("claude-3-haiku", None)) as mock_ensure:
                        result = runner.invoke(main, ["compile", "be professional"])

            mock_ensure.assert_called_once()
            assert mock_compiler.model == "claude-3-haiku"

    def test_compile_fails_without_api_key(self):
        """compile should fail with clear error when model requires a key that's missing."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("osentinel.yaml").write_text("engine: judge\nmodel: gpt-4o\n")

            # Ensure OPENAI_API_KEY is NOT set
            with patch.dict("os.environ", {}, clear=True):
                # Patch out HOME/PATH etc that SentinelSettings might need
                with patch.dict("os.environ", {"HOME": "/tmp"}, clear=False):
                    result = runner.invoke(main, ["compile", "be professional"])

            assert result.exit_code != 0
            assert "OPENAI_API_KEY" in result.output

    def test_compile_explicit_api_key_flag(self):
        """--api-key should be forwarded to the compiler regardless of provider."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("osentinel.yaml").write_text("engine: judge\nmodel: gpt-4o\n")

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.errors = []
            mock_result.warnings = []
            mock_result.metadata = {}

            mock_engine_cls = MagicMock()
            mock_engine_instance = MagicMock()
            mock_compiler = MagicMock()
            mock_compiler.compile = MagicMock(return_value=mock_result)
            mock_compiler.validate_result = MagicMock(return_value=[])
            mock_compiler.export = MagicMock()
            mock_engine_instance.get_compiler = MagicMock(return_value=mock_compiler)
            mock_engine_cls.return_value = mock_engine_instance

            with patch("opensentinel.policy.registry.PolicyEngineRegistry.get", return_value=mock_engine_cls):
                result = runner.invoke(
                    main, ["compile", "be professional", "--api-key", "sk-test-key"]
                )

            # get_compiler should have received the api_key
            mock_engine_instance.get_compiler.assert_called_once_with(
                model="gpt-4o", api_key="sk-test-key", base_url=None
            )

    def test_compile_gemini_resolves_key_from_env(self):
        """compile with gemini model should resolve key from GOOGLE_API_KEY."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("osentinel.yaml").write_text(
                "engine: judge\nmodel: gemini/gemini-2.5-flash\n"
            )

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.errors = []
            mock_result.warnings = []
            mock_result.metadata = {}

            mock_engine_cls = MagicMock()
            mock_engine_instance = MagicMock()
            mock_compiler = MagicMock()
            mock_compiler.compile = MagicMock(return_value=mock_result)
            mock_compiler.validate_result = MagicMock(return_value=[])
            mock_compiler.export = MagicMock()
            mock_engine_instance.get_compiler = MagicMock(return_value=mock_compiler)
            mock_engine_cls.return_value = mock_engine_instance

            with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-gemini-key"}):
                with patch("opensentinel.policy.registry.PolicyEngineRegistry.get", return_value=mock_engine_cls):
                    result = runner.invoke(main, ["compile", "be professional"])

            # get_compiler should have received the resolved gemini key
            mock_engine_instance.get_compiler.assert_called_once_with(
                model="gemini/gemini-2.5-flash",
                api_key="test-gemini-key",
                base_url=None,
            )


class TestHelpOutput:
    def test_main_help(self):
        result, _ = _invoke(["--help"])
        assert result.exit_code == 0
        assert "Open Sentinel" in result.output
        assert "init" in result.output
        assert "serve" in result.output
        assert "compile" in result.output
        assert "validate" in result.output
        assert "info" in result.output
        assert "version" in result.output
