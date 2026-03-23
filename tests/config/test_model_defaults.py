
import os
import pytest
from openbias.config.settings import Settings
from openbias.cli_init import detect_available_model


class TestDetectAvailableModel:
    """Tests for detect_available_model utility (used by CLI, not settings)."""

    def test_no_keys_returns_none(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        model, provider, env_var = detect_available_model()
        assert model is None
        assert provider is None
        assert env_var is None

    def test_openai_key_detects_gpt(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")

        model, provider, _ = detect_available_model()
        assert "gpt" in model
        assert provider == "OpenAI"

    def test_gemini_key_detects_gemini(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")

        model, provider, _ = detect_available_model()
        assert "gemini" in model
        assert provider == "Google Gemini"

    def test_anthropic_key_detects_claude(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key")

        model, provider, _ = detect_available_model()
        assert "anthropic" in model
        assert provider == "Anthropic"

    def test_openai_takes_priority(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")
        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")

        model, _, _ = detect_available_model()
        assert "gpt" in model


class TestModelDefaults:
    """Settings does NOT auto-detect models — that is the CLI's responsibility."""

    def test_no_keys_and_no_model_raises_on_validate(self, monkeypatch):
        """validate() raises when no model and no keys are set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("TOGETHERAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        settings = Settings(_env_file=None)

        with pytest.raises(ValueError, match="No LLM API keys detected"):
            settings.validate()

    def test_api_key_auto_detects_model(self, monkeypatch):
        """API keys should auto-populate proxy.default_model via _auto_detect_model."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")

        settings = Settings(_env_file=None)

        # Model auto-detected from available API key
        assert settings.proxy.default_model == "gemini/gemini-2.5-flash"

    def test_explicit_model_overrides_autodetect(self, monkeypatch):
        """Explicitly set model should be preserved."""
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")

        settings = Settings(
            proxy={"default_model": "custom/model"},
            _env_file=None,
        )

        assert settings.proxy.default_model == "custom/model"
