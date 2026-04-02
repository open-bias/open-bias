
import os
import pytest
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

    def test_default_config_is_valid_with_api_key(self, monkeypatch):
        """Test that default configuration with an API key passes validation."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        settings = Settings(
            proxy={"default_model": "gpt-4o-mini"},
            _env_file=None,
        )

        # Should not raise
        settings.validate()

    def test_fsm_engine_passes_without_api_keys(self, monkeypatch):
        """FSM engine is local-only and must not require any LLM API keys."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        settings = Settings(
            evaluators=[
                {"name": "workflow", "type": "fsm", "phase": "post_call"},
            ],
            _env_file=None,
        )

        # Should not raise even with no keys or model
        settings.validate()

    def test_no_api_key_raises_error(self):
        """Non-FSM engines (e.g. judge) must fail validation without a model/API key."""
        settings = Settings(
            proxy={"default_model": "gpt-4o-mini"},
            openai_api_key=None,
            anthropic_api_key=None,
            google_api_key=None,
            gemini_api_key=None,
            openrouter_api_key=None,
            _env_file=None,
        )

        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            settings.validate()

    @pytest.mark.parametrize(
        ("disallowed_key", "value"),
        [
            ("rules", ["Do not leak secrets"]),
            ("rules_file", "./rules.md"),
            ("workflow", {"states": []}),
        ],
    )
    def test_validate_rejects_user_authored_policy_keys_in_evaluator_config(
        self, disallowed_key, value
    ):
        """Direct Settings config must enforce the same rules.md-only contract as YAML."""
        settings = Settings(
            evaluators=[
                {
                    "name": "behavior",
                    "type": "judge",
                    "phase": "post_call",
                    "config": {disallowed_key: value},
                }
            ],
            proxy={"default_model": "gpt-4o-mini"},
            openai_api_key="sk-test-123",
            _env_file=None,
        )

        with pytest.raises(ValueError, match=rf"`{disallowed_key}` is not allowed"):
            settings.validate()

    def test_validate_rejects_user_authored_runtime_config_path(self, tmp_path):
        """Validation should reject engine-native runtime artifacts as user config."""
        runtime_dir = tmp_path / "compiled-nemo"
        runtime_dir.mkdir()
        settings = Settings(
            evaluators=[
                {
                    "name": "rails",
                    "type": "nemo",
                    "phase": "post_call",
                    "config": {"config_path": str(runtime_dir)},
                }
            ],
            proxy={"default_model": "gpt-4o-mini"},
            openai_api_key="sk-test-123",
            _env_file=None,
        )

        with pytest.raises(ValueError, match="project-local `rules.md`"):
            settings.validate()
