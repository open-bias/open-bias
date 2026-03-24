"""
Unified logging configuration for Open Bias.

Both the CLI commands and the Proxy server call ``configure_logging()`` so that
logging is configured exactly once, in the same way, regardless of entry point.

Features:
- ``ColoredFormatter`` for human-friendly console output (text mode)
- JSON-lines formatter for production / structured-log pipelines (json mode)
- ``RequestContextFilter`` that injects ``session_id`` and ``request_id``
  from ``contextvars`` into every ``LogRecord`` automatically
"""

import contextvars
import json as _json
import logging
import sys
from typing import Literal

import litellm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context variables — set in hooks.py, read by RequestContextFilter
# ---------------------------------------------------------------------------

session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id_var", default=""
)
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id_var", default=""
)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class RequestContextFilter(logging.Filter):
    """Inject ``session_id`` and ``request_id`` from *contextvars* into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = session_id_var.get("")  # type: ignore[attr-defined]
        record.request_id = request_id_var.get("")  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class ColoredFormatter(logging.Formatter):
    """ANSI-colored formatter for human-readable console output."""

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - [%(session_id)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class JsonFormatter(logging.Formatter):
    """Single-line JSON formatter for structured logging pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "session_id": getattr(record, "session_id", ""),
            "request_id": getattr(record, "request_id", ""),
        }
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return _json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def configure_logging(
    *,
    debug: bool = False,
    log_level: str = "INFO",
    log_format: Literal["text", "json"] = "text",
    litellm_verbose: bool = False,
) -> None:
    """Configure logging for the entire process.

    Args:
        debug: When True, overrides *log_level* to DEBUG and emits a warning
            about ``OBIAS_LITELLM_VERBOSE``.
        log_level: Root logger level (e.g. ``"INFO"``, ``"DEBUG"``).
            Ignored when *debug* is True.
        log_format: ``"text"`` for colored console output, ``"json"`` for
            structured JSON lines.
        litellm_verbose: When True, enables ``litellm.set_verbose``.
    """
    level = logging.DEBUG if debug else getattr(logging, log_level, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(ColoredFormatter())

    # Attach the context filter so session_id/request_id are always available
    context_filter = RequestContextFilter()
    handler.addFilter(context_filter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Replace existing handlers to avoid duplicates across repeated calls,
    # but preserve non-StreamHandler handlers (SpanEventManager, TriggerLogCollector, etc.)
    preserved = [
        h for h in root_logger.handlers
        if not isinstance(h, logging.StreamHandler)
        or type(h) is not logging.StreamHandler  # exact type, not subclass
    ]
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    for h in preserved:
        root_logger.addHandler(h)

    if debug:
        logger.warning(
            "Debug mode enabled. Set OBIAS_LITELLM_VERBOSE=true to also enable "
            "LiteLLM verbose logging (WARNING: this logs full request payloads "
            "including API keys)."
        )

    if litellm_verbose:
        litellm.set_verbose = True
