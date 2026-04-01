"""Tests for openbias.cli commands."""

from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from openbias.cli import main


def _invoke(args):
    """Invoke CLI and capture both Click output and Rich stdout."""
    runner = CliRunner()
    # Rich writes to sys.stdout; CliRunner captures click.echo output.
    # We need to capture both.
    buf = StringIO()
    from openbias.cli_ui import console

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
        assert "Open Bias" in output

    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "openbias" in result.output


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
            from openbias.cli_ui import console

            old_file = console.file
            console.file = buf
            try:
                with patch(
                    "openbias.policy.engines.fsm.workflow.parser.WorkflowParser.parse_file",
                    return_value=mock_workflow,
                ):
                    result = runner.invoke(main, ["validate", "test.yaml"])
            finally:
                console.file = old_file

            combined = result.output + buf.getvalue()
            assert result.exit_code == 0
            assert "Valid Workflow" in combined

    def test_validate_good_judge_config(self):
        """validate with a valid openbias.yaml should show summary."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
                "    model: gpt-4o-mini\n"
                '    rules:\n'
                '      - "Be professional"\n'
                '      - "No PII"\n'
                ""
            )

            buf = StringIO()
            from openbias.cli_ui import console

            old_file = console.file
            console.file = buf
            try:
                result = runner.invoke(main, ["validate", "openbias.yaml"])
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
            Path("openbias.yaml").write_text(
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
                '    rules:\n'
                '      - "Be professional"\n'
                ""
            )

            buf = StringIO()
            from openbias.cli_ui import console

            old_file = console.file
            console.file = buf
            try:
                result = runner.invoke(main, ["validate", "openbias.yaml"])
            finally:
                console.file = old_file

            combined = result.output + buf.getvalue()
            assert result.exit_code != 0
            assert "No model configured" in combined

    def test_validate_judge_config_bad_rubric(self):
        """validate with a nonexistent default rubric should fail."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: behavior\n"
                "    type: judge\n"
                "    phase: post_call\n"
                "    model: gpt-4o-mini\n"
                "    default_rubric: nonexistent_rubric\n"
                ""
            )

            buf = StringIO()
            from openbias.cli_ui import console

            old_file = console.file
            console.file = buf
            try:
                result = runner.invoke(main, ["validate", "openbias.yaml"])
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
            with patch("openbias.cli_init.run_init") as mock_run:
                result = runner.invoke(main, ["init", "--quick"])
                assert result.exit_code == 0
                mock_run.assert_called_once_with(quick=True)

    def test_init_non_tty_without_from(self):
        """Without --from and without TTY, should show error."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("openbias.cli_init.is_interactive", return_value=False):
                result = runner.invoke(main, ["init"])
                assert result.exit_code != 0


class TestServeCommand:
    def test_serve_missing_config(self):
        result, _ = _invoke(["serve", "--config", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_serve_no_yaml_prompts_init(self):
        """serve without an openbias.yaml should tell user to run openbias init."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            buf = StringIO()
            from openbias.cli_ui import console
            old_file = console.file
            console.file = buf
            try:
                result = runner.invoke(main, ["serve"])
            finally:
                console.file = old_file
            combined = result.output + buf.getvalue()
            assert result.exit_code != 0
            assert "openbias init" in combined

    def test_serve_with_yaml_proceeds(self):
        """serve with an openbias.yaml should pass the gate and attempt startup."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Write a minimal valid yaml
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\nport: 4000\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
                "    rules:\n"
                "      - 'Be professional'\n"
            )
            # Mock start_proxy so we don't actually start a server
            with patch("openbias.proxy.server.start_proxy"):
                with patch("openbias.config.settings.Settings.validate"):
                    result = runner.invoke(main, ["serve"])
            # Should NOT fail with the init-gate error
            combined = result.output
            assert "openbias init" not in combined or result.exit_code == 0

    def test_serve_compiles_rules_before_start(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
                "    rules:\n"
                "      - 'Be professional'\n"
            )
            with patch("openbias.proxy.server.start_proxy"):
                with patch("openbias.config.settings.Settings.validate"):
                    with patch(
                        "openbias.policy.compiler.runtime.compile_runtime_config_for_evaluator",
                        new_callable=AsyncMock,
                    ) as mock_compile:
                        mock_compile.return_value = {"inline_rules": ["Be professional"]}
                        result = runner.invoke(main, ["serve"])

            assert result.exit_code == 0
            assert mock_compile.called

class TestTriggerCommand:
    def test_trigger_help(self):
        result, output = _invoke(["trigger", "--help"])
        assert result.exit_code == 0
        assert "usage" in output.lower() or "Usage" in output

    def test_trigger_no_config(self):
        """trigger without openbias.yaml should show error with openbias init hint."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            buf = StringIO()
            from openbias.cli_ui import console
            old_file = console.file
            console.file = buf
            try:
                result = runner.invoke(main, ["trigger"])
            finally:
                console.file = old_file
            combined = result.output + buf.getvalue()
            assert result.exit_code != 0
            assert "openbias init" in combined

    def test_trigger_success(self):
        """trigger with valid config should show ALLOW output."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
                "    rules:\n"
                "      - 'Be professional'\n"
                ""
            )

            from unittest.mock import AsyncMock

            mock_proxy = MagicMock()
            mock_proxy.initialize = AsyncMock()
            mock_proxy.completion = AsyncMock(return_value={
                "choices": [{"message": {"content": "Hello, I am an AI assistant."}}]
            })
            mock_callback = MagicMock()
            mock_callback.shutdown = AsyncMock()
            mock_callback._policy_engine = MagicMock()  # not None — prevents pass-through warning
            mock_proxy._callback = mock_callback

            mock_proxy_class = MagicMock(return_value=mock_proxy)

            mock_settings = MagicMock()
            mock_settings.evaluators = [MagicMock(type="judge")]
            mock_settings.proxy.default_model = "gpt-4o-mini"
            mock_settings.fail_action = "block"
            mock_settings.fail_open = True
            mock_settings.validate = MagicMock()

            mock_settings_class = MagicMock(return_value=mock_settings)

            buf = StringIO()
            from openbias.cli_ui import console
            old_file = console.file
            console.file = buf
            try:
                with patch("openbias.config.settings.Settings", mock_settings_class):
                    with patch("openbias.proxy.server.Proxy", mock_proxy_class):
                        result = runner.invoke(main, ["trigger"])
            finally:
                console.file = old_file

            combined = result.output + buf.getvalue()
            assert result.exit_code == 0
            assert "ALLOW" in combined

    def test_trigger_error(self):
        """trigger when completion raises should print error cleanly."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
                "    rules:\n"
                "      - 'Be professional'\n"
                ""
            )

            from unittest.mock import AsyncMock

            mock_proxy = MagicMock()
            mock_proxy.initialize = AsyncMock()
            mock_proxy.completion = AsyncMock(side_effect=RuntimeError("API call failed"))
            mock_callback = MagicMock()
            mock_callback.shutdown = AsyncMock()
            mock_callback._policy_engine = MagicMock()
            mock_proxy._callback = mock_callback

            mock_proxy_class = MagicMock(return_value=mock_proxy)

            mock_settings = MagicMock()
            mock_settings.evaluators = [MagicMock(type="judge")]
            mock_settings.proxy.default_model = "gpt-4o-mini"
            mock_settings.fail_action = "block"
            mock_settings.fail_open = True
            mock_settings.validate = MagicMock()

            mock_settings_class = MagicMock(return_value=mock_settings)

            buf = StringIO()
            from openbias.cli_ui import console
            old_file = console.file
            console.file = buf
            try:
                with patch("openbias.config.settings.Settings", mock_settings_class):
                    with patch("openbias.proxy.server.Proxy", mock_proxy_class):
                        result = runner.invoke(main, ["trigger"])
            finally:
                console.file = old_file

            combined = result.output + buf.getvalue()
            # Should exit 0 (trigger handles errors internally, prints them)
            assert result.exit_code == 0
            assert "Error" in combined or "API call failed" in combined

    def test_trigger_custom_message(self):
        """--message flag should be passed through to the completion call."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
                "    rules:\n"
                "      - 'Be professional'\n"
                ""
            )

            from unittest.mock import AsyncMock

            completion_calls = []

            async def fake_completion(**kwargs):
                completion_calls.append(kwargs)
                return {"choices": [{"message": {"content": "Understood."}}]}

            mock_proxy = MagicMock()
            mock_proxy.initialize = AsyncMock()
            mock_proxy.completion = fake_completion
            mock_callback = MagicMock()
            mock_callback.shutdown = AsyncMock()
            mock_callback._policy_engine = MagicMock()
            mock_proxy._callback = mock_callback

            mock_proxy_class = MagicMock(return_value=mock_proxy)

            mock_settings = MagicMock()
            mock_settings.evaluators = [MagicMock(type="judge")]
            mock_settings.proxy.default_model = "gpt-4o-mini"
            mock_settings.fail_action = "block"
            mock_settings.fail_open = True
            mock_settings.validate = MagicMock()

            mock_settings_class = MagicMock(return_value=mock_settings)

            custom_msg = "Tell me about the weather"

            buf = StringIO()
            from openbias.cli_ui import console
            old_file = console.file
            console.file = buf
            try:
                with patch("openbias.config.settings.Settings", mock_settings_class):
                    with patch("openbias.proxy.server.Proxy", mock_proxy_class):
                        result = runner.invoke(main, ["trigger", "--message", custom_msg])
            finally:
                console.file = old_file

            assert result.exit_code == 0
            assert len(completion_calls) == 1
            messages = completion_calls[0]["messages"]
            assert any(
                m.get("content") == custom_msg
                for m in messages
                if isinstance(m, dict)
            )


class TestHelpOutput:
    def test_main_help(self):
        result, _ = _invoke(["--help"])
        assert result.exit_code == 0
        assert "Open Bias" in result.output
        assert "init" in result.output
        assert "serve" in result.output
        assert "validate" in result.output
        assert "info" in result.output
        assert "version" in result.output
        assert "trigger" in result.output
