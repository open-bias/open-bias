"""Tests for openbias.cli commands."""

from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from openbias.cli import main


def _restore_policy_registries() -> None:
    from openbias.policy.compiler.registry import PolicyCompilerRegistry
    from openbias.policy.engines.fsm import FSMCompiler, FSMPolicyEngine
    from openbias.policy.engines.judge.compiler import JudgeRuntimeCompiler
    from openbias.policy.engines.judge.engine import JudgePolicyEngine
    from openbias.policy.engines.llm import LLMCompiler, LLMPolicyEngine
    from openbias.policy.engines.nemo import NemoCompiler, NemoGuardrailsPolicyEngine
    from openbias.policy.registry import PolicyEngineRegistry

    PolicyEngineRegistry.register("fsm", FSMPolicyEngine)
    PolicyEngineRegistry.register("judge", JudgePolicyEngine)
    PolicyEngineRegistry.register("llm", LLMPolicyEngine)
    PolicyEngineRegistry.register("nemo", NemoGuardrailsPolicyEngine)

    PolicyCompilerRegistry.register("fsm", FSMCompiler)
    PolicyCompilerRegistry.register("judge", JudgeRuntimeCompiler)
    PolicyCompilerRegistry.register("llm", LLMCompiler)
    PolicyCompilerRegistry.register("nemo", NemoCompiler)


@pytest.fixture(autouse=True)
def _restore_registries_before_each_test() -> None:
    _restore_policy_registries()


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


    def test_validate_good_judge_config(self):
        """validate with a valid openbias.yaml should show summary."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("rules.md").write_text("- Be professional\n- No PII\n")
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
                "    model: gpt-4o-mini\n"
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
            Path("rules.md").write_text("- Be professional\n")
            Path("openbias.yaml").write_text(
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
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

    def test_validate_judge_config_missing_rules_md(self):
        """validate fails when project rules.md is missing."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: behavior\n"
                "    type: judge\n"
                "    phase: post_call\n"
                "    model: gpt-4o-mini\n"
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
            assert "requires project rules.md" in combined

    def test_validate_llm_config_reports_evaluator_engine_type(self):
        """Evaluator-only configs should summarize the active evaluator type."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: synthesis\n"
                "    type: llm\n"
                "    phase: post_call\n"
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
            assert "llm" in combined
            assert "unknown" not in combined


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

    def test_init_quick_emphasizes_project_local_rules_md(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            buf = StringIO()
            from openbias.cli_ui import console

            old_file = console.file
            console.file = buf
            try:
                result = runner.invoke(main, ["init", "--quick"])
            finally:
                console.file = old_file

            combined = result.output + buf.getvalue()
            assert result.exit_code == 0
            assert "project-local evaluator policy file: rules.md" in combined
            assert "Edit project-local rules.md for your evaluator policy" in combined
            assert Path("rules.md").exists()

            generated_yaml = Path("openbias.yaml").read_text()
            assert "rules_file" not in generated_yaml
            assert "config_path" not in generated_yaml

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
            Path("rules.md").write_text("- Be professional\n")
            # Write a minimal valid yaml
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\nport: 4000\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
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
            Path("rules.md").write_text("- Be professional\n")
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
            )
            with patch("openbias.proxy.server.start_proxy"):
                with patch("openbias.config.settings.Settings.validate"):
                    with patch(
                        "openbias.policy.compiler.runtime.compile_runtime_config_for_evaluator",
                        new_callable=AsyncMock,
                    ) as mock_compile:
                        mock_compile.return_value = {"_compiled_rules": ["Be professional"]}
                        result = runner.invoke(main, ["serve"])

            assert result.exit_code == 0
            assert mock_compile.called

    def test_serve_compiles_rules_from_rules_md(self):
        """serve compiles evaluator config from project rules.md at startup."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("rules.md").write_text("- Be helpful\n- No secrets\n")
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: behavior\n"
                "    type: judge\n"
                "    phase: post_call\n"
            )
            with patch("openbias.proxy.server.start_proxy"):
                with patch("openbias.config.settings.Settings.validate"):
                    with patch(
                        "openbias.policy.compiler.runtime.compile_runtime_config_for_evaluator",
                        new_callable=AsyncMock,
                    ) as mock_compile:
                        mock_compile.return_value = {"_compiled_rules": ["Be helpful", "No secrets"]}
                        result = runner.invoke(main, ["serve"])

            assert result.exit_code == 0
            mock_compile.assert_called_once()
            call_kwargs = mock_compile.call_args
            assert call_kwargs.kwargs["evaluator_name"] == "behavior"

    def test_serve_compiles_multiple_evaluators(self):
        """serve compiles rules for each evaluator in sequence."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("rules.md").write_text("- No harmful content\n- Be professional\n")
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: pre-screen\n"
                "    type: judge\n"
                "    phase: pre_call\n"
                "  - name: post-eval\n"
                "    type: judge\n"
                "    phase: post_call\n"
            )
            with patch("openbias.proxy.server.start_proxy"):
                with patch("openbias.config.settings.Settings.validate"):
                    with patch(
                        "openbias.policy.compiler.runtime.compile_runtime_config_for_evaluator",
                        new_callable=AsyncMock,
                    ) as mock_compile:
                        mock_compile.return_value = {"_compiled_rules": ["No harmful content"]}
                        result = runner.invoke(main, ["serve"])

            assert result.exit_code == 0
            assert mock_compile.call_count == 2

    def test_serve_compilation_failure_exits(self):
        """serve exits with error when rules compilation fails."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("rules.md").write_text("- Be professional\n")
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
            )
            with patch("openbias.proxy.server.start_proxy"):
                with patch("openbias.config.settings.Settings.validate"):
                    with patch(
                        "openbias.policy.compiler.runtime.compile_runtime_config_for_evaluator",
                        new_callable=AsyncMock,
                        side_effect=ValueError("Failed to compile rules"),
                    ):
                        result = runner.invoke(main, ["serve"])

            assert result.exit_code != 0

    def test_serve_auto_discovers_rules_md(self):
        """serve auto-discovers rules.md when no explicit rules are set."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("rules.md").write_text("Auto rule one\n\nAuto rule two\n")
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
            )
            with patch("openbias.proxy.server.start_proxy"):
                with patch("openbias.config.settings.Settings.validate"):
                    with patch(
                        "openbias.policy.compiler.runtime.compile_runtime_config_for_evaluator",
                        new_callable=AsyncMock,
                    ) as mock_compile:
                        mock_compile.return_value = {"_compiled_rules": ["Auto rule one", "Auto rule two"]}
                        result = runner.invoke(main, ["serve"])

            assert result.exit_code == 0
            # Compilation was called because rules.md was auto-discovered
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


class TestReplayCommand:
    def test_replay_requires_trace(self):
        result, _ = _invoke(["replay"])
        assert result.exit_code != 0

    def test_replay_delegates_to_cli_replay_module(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("rules.md").write_text("- Be professional\n")
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: behavior\n"
                "    type: judge\n"
                "    phase: post_call\n"
            )
            Path("trace.jsonl").write_text(
                '{"id":"trace-1","session_id":"sess-1","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}]}\n'
            )

            with patch("openbias.cli_replay.run_replay") as mock_run_replay:
                result = runner.invoke(main, ["replay", "--trace", "trace.jsonl"])
                assert result.exit_code == 0
                mock_run_replay.assert_called_once()


class TestImproveCommand:
    def test_improve_requires_trace(self):
        result, _ = _invoke(["improve", "--instruction", "tighten the policy"])
        assert result.exit_code != 0

    def test_improve_requires_instruction(self):
        result, _ = _invoke(["improve"])
        assert result.exit_code != 0

    def test_improve_delegates_to_cli_improve_module(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("rules.md").write_text("- Be professional\n")
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: behavior\n"
                "    type: judge\n"
                "    phase: post_call\n"
            )
            Path("trace.jsonl").write_text(
                '{"id":"trace-1","session_id":"sess-1","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}]}\n'
            )

            with patch("openbias.cli_improve.run_improve") as mock_run_improve:
                result = runner.invoke(
                    main,
                    ["improve", "--trace", "trace.jsonl", "--instruction", "tighten the policy"],
                )
                assert result.exit_code == 0
                mock_run_improve.assert_called_once()


class TestLegacyImprovementCommands:
    def test_compare_command_is_removed(self):
        result, output = _invoke(["compare"])
        assert result.exit_code != 0
        assert "No such command 'compare'" in output

    def test_review_pack_command_is_removed(self):
        result, output = _invoke(["review-pack"])
        assert result.exit_code != 0
        assert "No such command 'review-pack'" in output

class TestTriggerCommand:
    def test_trigger_success(self):
        """trigger with valid config should show ALLOW output."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("rules.md").write_text("- Be professional\n")
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
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
            Path("rules.md").write_text("- Be professional\n")
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
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
            Path("rules.md").write_text("- Be professional\n")
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: safety\n"
                "    type: judge\n"
                "    phase: post_call\n"
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


class TestEvalCommand:
    def test_eval_delegates_to_cli_eval_module(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("rules.md").write_text("- Be professional\n")
            Path("openbias.yaml").write_text(
                "model: gpt-4o-mini\n"
                "evaluators:\n"
                "  - name: behavior\n"
                "    type: judge\n"
                "    phase: post_call\n"
            )
            suite_dir = Path("evals/suites")
            suite_dir.mkdir(parents=True)
            (suite_dir / "smoke.yaml").write_text(
                "name: smoke\n"
                "cases:\n"
                "  - id: safe\n"
                "    tags: [safe]\n"
                "    labels:\n"
                "      violation: false\n"
                "      detection_scope: either\n"
                "      detect_at_turn: null\n"
                "      repair_expected: null\n"
                "      repair_verified_at_turn: null\n"
                "    messages:\n"
                "      - role: user\n"
                "        content: hello\n",
                encoding="utf-8",
            )

            with patch("openbias.cli_eval.run_eval") as mock_run_eval:
                result = runner.invoke(
                    main,
                    ["eval", "--config", "openbias.yaml", "--suite", "evals/suites"],
                )
                assert result.exit_code == 0
                mock_run_eval.assert_called_once_with(
                    config=Path("openbias.yaml"),
                    suite_paths=(Path("evals/suites"),),
                    json_output=None,
                    verbose=False,
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
