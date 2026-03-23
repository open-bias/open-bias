
import os
from pathlib import Path
import pytest
import yaml
from openbias.config.settings import Settings

class TestConfigValidation:

    def test_invalid_yaml_raises_error(self, tmp_path):
        """Test that invalid YAML syntax raises an error when loaded."""
        config_path = tmp_path / "bad_config.yaml"
        # Write invalid YAML
        with open(config_path, "w") as f:
            f.write("engine: judge\npolicy:\n  - 'missing quote")
            
        with pytest.raises(Exception, match="while scanning a quoted scalar"):
            Settings(_config_path=str(config_path))

    def test_missing_policy_file_raises_error(self, tmp_path):
        """Test that referencing a non-existent policy file raises ValueError during validation."""
        config_path = tmp_path / "config.yaml"
        # Correct structure for config_path
        config = {
            "engine": "nemo",
            "nemo": {
                "config_path": "non_existent.yaml"
            }
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Disable .env loading to prevent pollution
        settings = Settings(_config_path=str(config_path), _env_file=None)
        
        with pytest.raises(ValueError, match="Policy configuration file not found"):
            settings.validate()

    def test_missing_api_key_raises_error(self, tmp_path, monkeypatch):
        """Test that missing API key for default model raises ValueError."""
        # Ensure no accidental keys provided by env
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        
        settings = Settings(
            # Override to ensure fallback logic doesn't pick up something else
            proxy={"default_model": "gpt-4o-mini"},
            openai_api_key=None,
            _env_file=None # Important: prevent reading real .env
        )
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            settings.validate()

    def test_valid_config_passes(self, tmp_path, monkeypatch):
        """Test that valid config with API keys passes validation."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        
        settings = Settings(
            proxy={"default_model": "gpt-4o-mini"}
        )
        
        # Should not raise
        settings.validate()
        
    def test_env_sync(self, monkeypatch):
        """Test that keys provided in settings are synced to os.environ for downstream libs."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        # Instantiate settings with explicit key and no .env file interference
        settings = Settings(openai_api_key="sk-explicit-test", _env_file=None)
        
        # Check if it was pushed to environ
        assert os.environ.get("OPENAI_API_KEY") == "sk-explicit-test"

    def test_default_config_is_judge_and_valid(self, monkeypatch):
        """Test that default configuration without any file uses judge engine and passes validation if API key and model present."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        settings = Settings(
            proxy={"default_model": "gpt-4o-mini"},
            _env_file=None,
        )
        assert settings.policy.engine.type == "judge"

        # Should not raise
        settings.validate()

    def test_fsm_engine_passes_without_api_keys(self, monkeypatch):
        """FSM engine is local-only and must not require any LLM API keys."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        settings = Settings(
            policy={"engine": {"type": "fsm"}},
            _env_file=None,
        )
        assert settings.policy.engine.type == "fsm"

        # Should not raise even with no keys or model
        settings.validate()

    def test_non_fsm_engine_still_requires_model(self):
        """Non-FSM engines (e.g. judge) must still fail validation without a model/API key."""
        settings = Settings(
            policy={"engine": {"type": "judge"}},
            openai_api_key=None,
            anthropic_api_key=None,
            google_api_key=None,
            gemini_api_key=None,
            openrouter_api_key=None,
            _env_file=None,
        )
        assert settings.policy.engine.type == "judge"

        with pytest.raises(ValueError, match="No LLM API keys detected"):
            settings.validate()
