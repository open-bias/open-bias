import os
import pytest
from opensentinel.config.settings import SentinelSettings, YamlConfigSource


def test_osntl_config_env_var_discovery():
    """Verify that OSNTL_CONFIG is still used to find the config file path."""
    # This is handled manually in YamlConfigSource, so it should still work
    os.environ["OSNTL_CONFIG"] = "/tmp/nonexistent.yaml"
    
    settings = SentinelSettings()
    # It won't fail to initialize, but YamlConfigSource will have attempted to load it
    # We can check its internal state or just ensure no other OSNTL_ vars are picked up
    
    os.environ["OSNTL_DEBUG"] = "true"
    settings = SentinelSettings()
    assert settings.debug is True  # OSNTL_ prefix maps to settings fields

    del os.environ["OSNTL_CONFIG"]
    del os.environ["OSNTL_DEBUG"]


def test_standard_api_keys_work():
    """Verify that standard API keys are still picked up without OSNTL_ prefix."""
    os.environ["OPENAI_API_KEY"] = "sk-test-123"
    settings = SentinelSettings()
    assert settings.openai_api_key == "sk-test-123"
    del os.environ["OPENAI_API_KEY"]


class TestInlinePolicyMapping:
    """Tests for inline policy handling in YamlConfigSource._map_to_settings()."""

    def _build_source(self, yaml_data):
        """Create a YamlConfigSource with injected YAML data."""
        source = YamlConfigSource.__new__(YamlConfigSource)
        source._yaml_data = yaml_data
        source._config_file = None
        return source

    def test_policy_list_maps_to_inline_policy(self):
        """A list of rules should map to policy.engine.config.inline_policy."""
        source = self._build_source({
            "engine": "judge",
            "policy": ["No PII", "Be helpful"],
        })
        result = source._map_to_settings()
        assert result["policy"]["engine"]["config"]["inline_policy"] == ["No PII", "Be helpful"]
        assert "config_path" not in result["policy"]["engine"]

    def test_policy_dict_maps_to_inline_policy(self):
        """A dict policy should map to policy.engine.config.inline_policy."""
        source = self._build_source({
            "engine": "judge",
            "policy": {"rules": ["No PII"]},
        })
        result = source._map_to_settings()
        assert result["policy"]["engine"]["config"]["inline_policy"] == {"rules": ["No PII"]}

    def test_policy_string_maps_to_config_path(self):
        """A string policy should still map to config_path (backward compat)."""
        source = self._build_source({
            "engine": "judge",
            "policy": "./policy.yaml",
        })
        result = source._map_to_settings()
        assert result["policy"]["engine"]["config_path"] == "./policy.yaml"
        assert "config" not in result["policy"]["engine"] or \
               "inline_policy" not in result["policy"]["engine"].get("config", {})


class TestInterventionDefaults:

    def test_default_strategy_is_user_message_inject(self):
        """PolicyConfig defaults to user_message_inject strategy."""
        settings = SentinelSettings()
        assert settings.policy.default_strategy == "user_message_inject"


class TestGetPolicyConfig:
    """Tests for SentinelSettings.get_policy_config() model injection."""

    def test_injects_models_from_global_default_model(self):
        """When judge engine has no explicit model, inject from proxy.default_model."""
        settings = SentinelSettings()
        settings.proxy.default_model = "gpt-4o"
        settings.policy.engine.type = "judge"
        settings.policy.engine.config = {}

        result = settings.get_policy_config()
        assert result["config"]["models"] == [
            {"name": "primary", "model": "gpt-4o"}
        ]

    def test_no_injection_when_models_already_set(self):
        """When judge engine already has models, don't override."""
        settings = SentinelSettings()
        settings.proxy.default_model = "gpt-4o"
        settings.policy.engine.type = "judge"
        settings.policy.engine.config = {
            "models": [{"name": "primary", "model": "claude-sonnet-4-6"}]
        }

        result = settings.get_policy_config()
        assert result["config"]["models"] == [
            {"name": "primary", "model": "claude-sonnet-4-6"}
        ]

    def test_no_injection_for_non_judge_engine(self):
        """Non-judge engines should not get models injected."""
        settings = SentinelSettings()
        settings.proxy.default_model = "gpt-4o"
        settings.policy.engine.type = "fsm"
        settings.policy.engine.config = {}

        result = settings.get_policy_config()
        assert "models" not in result["config"]

    def test_no_injection_when_no_default_model(self):
        """When there's no global default_model, don't inject."""
        settings = SentinelSettings()
        settings.proxy.default_model = None
        settings.policy.engine.type = "judge"
        settings.policy.engine.config = {}

        result = settings.get_policy_config()
        assert "models" not in result["config"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))

