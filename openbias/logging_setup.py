"""
Unified logging configuration for Open Bias.

Both the CLI commands and the Proxy server call `setup_logging()` so that
logging is configured exactly once, in the same way, regardless of entry point.
"""

import logging
import sys

import litellm

logger = logging.getLogger(__name__)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds ANSI colors to log output."""

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logging(
    *,
    debug: bool = False,
    log_level: str = "INFO",
    litellm_verbose: bool = False,
) -> None:
    """Configure logging for the entire process.

    Args:
        debug: When True, overrides *log_level* to DEBUG and emits a warning
            about ``OBIAS_LITELLM_VERBOSE``.
        log_level: Root logger level (e.g. ``"INFO"``, ``"DEBUG"``).
            Ignored when *debug* is True.
        litellm_verbose: When True, enables ``litellm.set_verbose``.
    """
    level = logging.DEBUG if debug else getattr(logging, log_level, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Replace existing handlers to avoid duplicates across repeated calls.
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    if debug:
        logger.warning(
            "Debug mode enabled. Set OBIAS_LITELLM_VERBOSE=true to also enable "
            "LiteLLM verbose logging (WARNING: this logs full request payloads "
            "including API keys)."
        )

    if litellm_verbose:
        litellm.set_verbose = True
