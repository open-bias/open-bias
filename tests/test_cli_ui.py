"""Tests for openbias.cli_ui module."""

from io import StringIO
from unittest.mock import patch

from openbias.cli_ui import (
    _panel_width,
    banner,
    config_panel,
    console,
    dim,
    error,
    heading,
    is_interactive,
    key_value,
    make_table,
    next_steps,
    success,
    warning,
    yaml_preview,
)


def _capture_output(fn, *args, **kwargs):
    """Capture Rich console output by temporarily redirecting."""
    buf = StringIO()
    original_file = console.file
    console.file = buf
    try:
        fn(*args, **kwargs)
    finally:
        console.file = original_file
    return buf.getvalue()


class TestOutputHelpers:
    def test_banner_contains_version(self):
        output = _capture_output(banner, "1.2.3")
        assert "Open Bias" in output
        assert "1.2.3" in output

    def test_banner_contains_description(self):
        output = _capture_output(banner, "0.1.0")
        assert "Reliability layer" in output

    def test_heading_plain(self):
        output = _capture_output(heading, "Test Section")
        assert "Test Section" in output

    def test_heading_with_step(self):
        output = _capture_output(heading, "Engine", step=2, total=5)
        assert "2/5" in output
        assert "Engine" in output

    def test_success_message(self):
        output = _capture_output(success, "All good")
        assert "\u2713" in output
        assert "All good" in output

    def test_error_message(self):
        output = _capture_output(error, "Something broke")
        assert "\u2717" in output
        assert "Something broke" in output

    def test_error_with_hint(self):
        output = _capture_output(error, "Failed", hint="Try this instead")
        assert "Failed" in output
        assert "Try this instead" in output

    def test_warning_message(self):
        output = _capture_output(warning, "Watch out")
        assert "!" in output
        assert "Watch out" in output

    def test_dim_message(self):
        output = _capture_output(dim, "Secondary info")
        assert "Secondary info" in output

    def test_key_value(self):
        output = _capture_output(key_value, "Name", "sentinel")
        assert "Name" in output
        assert "sentinel" in output

    def test_next_steps(self):
        output = _capture_output(next_steps, ["step one", "step two"])
        assert "Next steps" in output
        assert "1." in output
        assert "step one" in output
        assert "2." in output
        assert "step two" in output

    def test_yaml_preview(self):
        yaml_content = "engine: judge\nmodel: gpt-4o"
        output = _capture_output(yaml_preview, yaml_content, title="test.yaml")
        assert "engine" in output
        assert "judge" in output

    def test_yaml_preview_truncates_long_content(self):
        # 40 lines should get truncated to 30
        yaml_content = "\n".join([f"key_{i}: value_{i}" for i in range(40)])
        output = _capture_output(yaml_preview, yaml_content)
        assert "truncated" in output

    def test_config_panel(self):
        output = _capture_output(
            config_panel, "Test Panel", {"Engine": "judge", "Port": "4000"}
        )
        assert "Test Panel" in output
        assert "Engine" in output
        assert "judge" in output
        assert "Port" in output

    def test_make_table(self):
        output = _capture_output(
            make_table, "States", ["Name", "Type"], [["start", "initial"], ["end", "terminal"]]
        )
        assert "States" in output
        assert "start" in output
        assert "end" in output


class TestPanelWidth:
    def test_panel_width_capped_at_80(self):
        width = _panel_width()
        assert width <= 80

    def test_panel_width_positive(self):
        width = _panel_width()
        assert width > 0


class TestInputHelpers:
    """Test that input wrappers handle None (Ctrl-C) correctly."""

    @patch("openbias.cli_ui.questionary")
    def test_select_exits_on_none(self, mock_q):
        mock_q.select.return_value.ask.return_value = None
        import pytest

        with pytest.raises(SystemExit):
            from openbias.cli_ui import select

            select("Pick one", [{"name": "a", "value": "a"}])

    @patch("openbias.cli_ui.questionary")
    def test_confirm_exits_on_none(self, mock_q):
        mock_q.confirm.return_value.ask.return_value = None
        import pytest

        with pytest.raises(SystemExit):
            from openbias.cli_ui import confirm

            confirm("Sure?")

    @patch("openbias.cli_ui.questionary")
    def test_text_exits_on_none(self, mock_q):
        mock_q.text.return_value.ask.return_value = None
        import pytest

        with pytest.raises(SystemExit):
            from openbias.cli_ui import text

            text("Enter something")

    @patch("openbias.cli_ui.questionary")
    def test_select_returns_value(self, mock_q):
        mock_q.select.return_value.ask.return_value = "judge"
        from openbias.cli_ui import select

        result = select("Pick engine", [{"name": "judge", "value": "judge"}])
        assert result == "judge"

    @patch("openbias.cli_ui.questionary")
    def test_confirm_returns_bool(self, mock_q):
        mock_q.confirm.return_value.ask.return_value = True
        from openbias.cli_ui import confirm

        result = confirm("Sure?")
        assert result is True

    @patch("openbias.cli_ui.questionary")
    def test_text_returns_string(self, mock_q):
        mock_q.text.return_value.ask.return_value = "hello"
        from openbias.cli_ui import text

        result = text("Enter something")
        assert result == "hello"


class TestIsInteractive:
    @patch("openbias.cli_ui.sys")
    def test_interactive_when_tty(self, mock_sys):
        mock_sys.stdin.isatty.return_value = True
        assert is_interactive() is True

    @patch("openbias.cli_ui.sys")
    def test_not_interactive_when_not_tty(self, mock_sys):
        mock_sys.stdin.isatty.return_value = False
        assert is_interactive() is False
