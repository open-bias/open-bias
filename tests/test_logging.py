"""Tests for openbias.logging — configure_logging, RequestContextFilter, formatters."""

import json
import logging

import pytest

from openbias.logging import (
    ColoredFormatter,
    JsonFormatter,
    RequestContextFilter,
    configure_logging,
    session_id_var,
    request_id_var,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Restore root logger state after each test."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)


@pytest.fixture(autouse=True)
def _reset_context_vars():
    """Clear context vars before each test."""
    t1 = session_id_var.set("")
    t2 = request_id_var.set("")
    yield
    session_id_var.reset(t1)
    request_id_var.reset(t2)


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def test_sets_root_logger_level_info_by_default(self):
        configure_logging()
        assert logging.getLogger().level == logging.INFO

    def test_debug_flag_overrides_level(self):
        configure_logging(debug=True)
        assert logging.getLogger().level == logging.DEBUG

    def test_log_level_parameter(self):
        configure_logging(log_level="WARNING")
        assert logging.getLogger().level == logging.WARNING

    def test_text_format_uses_colored_formatter(self):
        configure_logging(log_format="text")
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert any(isinstance(h.formatter, ColoredFormatter) for h in stream_handlers)

    def test_json_format_uses_json_formatter(self):
        configure_logging(log_format="json")
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert any(isinstance(h.formatter, JsonFormatter) for h in stream_handlers)

    def test_preserves_non_stream_handlers(self):
        """Non-StreamHandler handlers (e.g. SpanEventManager) must survive reconfiguration."""
        root = logging.getLogger()
        custom_handler = logging.FileHandler("/dev/null")
        root.addHandler(custom_handler)

        configure_logging()

        assert custom_handler in root.handlers
        # Clean up
        root.removeHandler(custom_handler)
        custom_handler.close()


# ---------------------------------------------------------------------------
# RequestContextFilter
# ---------------------------------------------------------------------------


class TestRequestContextFilter:
    def test_injects_session_id_and_request_id(self):
        session_id_var.set("sess-123")
        request_id_var.set("req-456")

        f = RequestContextFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        result = f.filter(record)

        assert result is True
        assert record.session_id == "sess-123"  # type: ignore[attr-defined]
        assert record.request_id == "req-456"  # type: ignore[attr-defined]

    def test_defaults_to_empty_strings(self):
        f = RequestContextFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        f.filter(record)

        assert record.session_id == ""  # type: ignore[attr-defined]
        assert record.request_id == ""  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    def test_produces_parseable_json(self):
        session_id_var.set("sess-abc")
        request_id_var.set("req-def")

        formatter = JsonFormatter()
        filt = RequestContextFilter()

        record = logging.LogRecord(
            name="openbias.test", level=logging.INFO, pathname="test.py",
            lineno=42, msg="test message", args=(), exc_info=None,
        )
        filt.filter(record)
        output = formatter.format(record)

        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "openbias.test"
        assert parsed["message"] == "test message"
        assert parsed["session_id"] == "sess-abc"
        assert parsed["request_id"] == "req-def"
        assert "timestamp" in parsed

    def test_includes_exception_info(self):
        formatter = JsonFormatter()
        filt = RequestContextFilter()

        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="test.py",
            lineno=1, msg="error", args=(), exc_info=exc_info,
        )
        filt.filter(record)
        output = formatter.format(record)

        parsed = json.loads(output)
        assert "exception" in parsed
        assert "boom" in parsed["exception"]
