"""Tests for Proxy._setup_logging() and generate_litellm_config() behaviour."""

from unittest.mock import patch

import litellm
import pytest
import yaml

from openbias.config.settings import Settings
from openbias.proxy.server import Proxy

# The warning is now emitted by the shared openbias.logging module.
_LOGGER_PATCH_TARGET = "openbias.logging.logger"


@pytest.fixture(autouse=True)
def reset_litellm_verbose():
    """Restore litellm.set_verbose to its original value after each test."""
    original = litellm.set_verbose
    yield
    litellm.set_verbose = original


def _make_proxy(**kwargs) -> Proxy:
    settings = Settings(**kwargs)
    return Proxy(settings)


# ---------------------------------------------------------------------------
# debug=True must NOT enable litellm.set_verbose
# ---------------------------------------------------------------------------


def test_debug_mode_does_not_enable_litellm_verbose():
    """Enabling debug must not set litellm.set_verbose=True."""
    litellm.set_verbose = False
    proxy = _make_proxy(debug=True, litellm_verbose=False)
    proxy._setup_logging()
    assert litellm.set_verbose is False


# ---------------------------------------------------------------------------
# debug=True must emit a warning about OBIAS_LITELLM_VERBOSE
# ---------------------------------------------------------------------------


def test_debug_mode_emits_warning():
    proxy = _make_proxy(debug=True, litellm_verbose=False)
    with patch(_LOGGER_PATCH_TARGET) as mock_logger:
        proxy._setup_logging()
    mock_logger.warning.assert_called_once()
    call_args = mock_logger.warning.call_args[0][0]
    assert "OBIAS_LITELLM_VERBOSE" in call_args


# ---------------------------------------------------------------------------
# debug=False must NOT emit the warning
# ---------------------------------------------------------------------------


def test_no_warning_when_debug_false():
    proxy = _make_proxy(debug=False, litellm_verbose=False)
    with patch(_LOGGER_PATCH_TARGET) as mock_logger:
        proxy._setup_logging()
    mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# litellm_verbose=True (without debug) must enable litellm.set_verbose
# ---------------------------------------------------------------------------


def test_litellm_verbose_flag_enables_verbose():
    litellm.set_verbose = False
    proxy = _make_proxy(debug=False, litellm_verbose=True)
    proxy._setup_logging()
    assert litellm.set_verbose is True


# ---------------------------------------------------------------------------
# Both debug=True and litellm_verbose=True: verbose enabled, warning present
# ---------------------------------------------------------------------------


def test_debug_and_litellm_verbose_both_set():
    litellm.set_verbose = False
    proxy = _make_proxy(debug=True, litellm_verbose=True)
    with patch(_LOGGER_PATCH_TARGET) as mock_logger:
        proxy._setup_logging()
    assert litellm.set_verbose is True
    mock_logger.warning.assert_called_once()
    call_args = mock_logger.warning.call_args[0][0]
    assert "OBIAS_LITELLM_VERBOSE" in call_args


# ---------------------------------------------------------------------------
# Settings: litellm_verbose defaults to False
# ---------------------------------------------------------------------------


def test_settings_litellm_verbose_default():
    settings = Settings()
    assert settings.litellm_verbose is False


# ---------------------------------------------------------------------------
# Settings: litellm_verbose respects env var OBIAS_LITELLM_VERBOSE
# ---------------------------------------------------------------------------


def test_settings_litellm_verbose_from_env(monkeypatch):
    monkeypatch.setenv("OBIAS_LITELLM_VERBOSE", "true")
    settings = Settings()
    assert settings.litellm_verbose is True


# ---------------------------------------------------------------------------
# generate_litellm_config: master_key handling
# ---------------------------------------------------------------------------


def test_generate_litellm_config_includes_master_key_when_set():
    """master_key must appear in general_settings when explicitly configured."""
    proxy = _make_proxy(proxy={"master_key": "sk-my-secret-key"})
    config = yaml.safe_load(proxy.generate_litellm_config())
    assert config["general_settings"].get("master_key") == "sk-my-secret-key"


def test_generate_litellm_config_omits_master_key_when_none():
    """master_key must not appear in general_settings when not configured (None)."""
    proxy = _make_proxy()  # master_key defaults to None
    config = yaml.safe_load(proxy.generate_litellm_config())
    assert "master_key" not in config["general_settings"]


def test_generate_litellm_config_omits_master_key_when_empty_string():
    """master_key must not appear in general_settings when set to empty string."""
    proxy = _make_proxy(proxy={"master_key": ""})
    config = yaml.safe_load(proxy.generate_litellm_config())
    assert "master_key" not in config["general_settings"]


def test_log_policy_config_reports_llm_engine():
    proxy = _make_proxy()

    with patch.object(
        Settings,
        "get_policy_config",
        return_value={"type": "llm", "config": {"llm_model": "gpt-4o-mini"}},
    ):
        with patch("openbias.proxy.server.logger") as mock_logger:
            proxy._log_policy_config()

    mock_logger.info.assert_called_once()
    assert "LLM" in mock_logger.info.call_args.args[0]
