import pytest

from openbias.config.settings import (
    EvaluatorConfig,
    Settings,
    YamlConfigSource,
)


def test_obias_config_env_var_discovery(monkeypatch):
    """Verify that OBIAS_CONFIG is still used to find the config file path."""
    # This is handled manually in YamlConfigSource, so it should still work
    monkeypatch.setenv("OBIAS_CONFIG", "/tmp/nonexistent.yaml")

    settings = Settings()
    # It won't fail to initialize, but YamlConfigSource will have attempted to load it
    # We can check its internal state or just ensure no other OBIAS_ vars are picked up

    monkeypatch.setenv("OBIAS_DEBUG", "true")
    settings = Settings()
    assert settings.debug is True  # OBIAS_ prefix maps to settings fields


def test_standard_api_keys_work(monkeypatch):
    """Verify that standard API keys are still picked up without OBIAS_ prefix."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    settings = Settings()
    assert settings.openai_api_key == "sk-test-123"


# =========================================================================
# EvaluatorConfig tests
# =========================================================================


class TestEvaluatorConfig:
    """Tests for the new EvaluatorConfig model."""

    def test_defaults(self):
        cfg = EvaluatorConfig(name="safety")
        assert cfg.name == "safety"
        assert cfg.type == "judge"
        assert cfg.phase == "post_call"
        assert cfg.config == {}

    def test_custom_values(self):
        cfg = EvaluatorConfig(
            name="pre_check",
            type="custom",
            phase="pre_call",
            config={"threshold": 0.8},
        )
        assert cfg.name == "pre_check"
        assert cfg.type == "custom"
        assert cfg.phase == "pre_call"
        assert cfg.config == {"threshold": 0.8}

    def test_phase_literal_validation(self):
        with pytest.raises(Exception):
            EvaluatorConfig(name="bad", phase="invalid")


# =========================================================================
# Flat field tests
# =========================================================================


class TestFlatFields:
    """Tests for the new flat evaluator-pipeline fields on Settings."""

    def test_flat_defaults(self):
        settings = Settings()
        assert settings.mode == "async"
        assert settings.fail_action == "intervene"
        assert settings.strategy == "user_message_inject"
        assert settings.session_ttl == 3600
        assert settings.max_sessions == 10000
        assert settings.fail_open is True
        assert settings.hook_timeout_seconds == 30.0
        assert settings.evaluators == []

    def test_evaluators_list(self):
        settings = Settings(
            evaluators=[
                {"name": "safety", "type": "judge", "phase": "post_call"},
                {"name": "pre_check", "type": "custom", "phase": "pre_call"},
            ]
        )
        assert len(settings.evaluators) == 2
        assert settings.evaluators[0].name == "safety"
        assert settings.evaluators[1].phase == "pre_call"


# =========================================================================
# New evaluator-based YAML format tests
# =========================================================================


class TestEvaluatorYamlMapping:
    """Tests for the new evaluator-based YAML parsing path in _map_evaluators()."""

    def _build_source(self, yaml_data, config_file=None):
        """Create a YamlConfigSource with injected YAML data."""
        source = YamlConfigSource.__new__(YamlConfigSource)
        source._yaml_data = yaml_data
        source._config_file = config_file
        return source

    def test_global_settings_map_to_top_level(self):
        """Global pipeline settings map directly to top-level result keys."""
        source = self._build_source({
            "mode": "sync",
            "fail_action": "block",
            "strategy": "system_prompt_append",
            "session_ttl": 7200,
            "max_sessions": 5000,
            "fail_open": False,
            "hook_timeout_seconds": 60.0,
            "evaluators": [],
        })
        result = source._map_evaluators(source._yaml_data)
        assert result["mode"] == "sync"
        assert result["fail_action"] == "block"
        assert result["strategy"] == "system_prompt_append"
        assert result["session_ttl"] == 7200
        assert result["max_sessions"] == 5000
        assert result["fail_open"] is False
        assert result["hook_timeout_seconds"] == 60.0
        # Should NOT have policy key
        assert "policy" not in result

    def test_proxy_fields_mapped(self):
        """port, host, model map to proxy settings."""
        source = self._build_source({
            "port": 5000,
            "host": "127.0.0.1",
            "model": "gpt-4o",
            "evaluators": [],
        })
        result = source._map_evaluators(source._yaml_data)
        assert result["proxy"]["port"] == 5000
        assert result["proxy"]["host"] == "127.0.0.1"
        assert result["proxy"]["default_model"] == "gpt-4o"

    def test_debug_and_log_level_mapped(self):
        """debug and log_level pass through."""
        source = self._build_source({
            "debug": True,
            "log_level": "DEBUG",
            "evaluators": [],
        })
        result = source._map_evaluators(source._yaml_data)
        assert result["debug"] is True
        assert result["log_level"] == "DEBUG"

    def test_tracing_mapped_to_otel(self):
        """tracing section maps to otel in new format too."""
        source = self._build_source({
            "evaluators": [],
            "tracing": {"type": "otlp", "endpoint": "http://jaeger:4317"},
        })
        result = source._map_evaluators(source._yaml_data)
        assert result["otel"]["exporter_type"] == "otlp"
        assert result["otel"]["endpoint"] == "http://jaeger:4317"

    def test_judge_evaluator_basic(self):
        """Basic judge evaluator is parsed correctly."""
        source = self._build_source({
            "evaluators": [
                {"name": "safety", "type": "judge", "phase": "post_call"},
            ],
        })
        result = source._map_evaluators(source._yaml_data)
        assert len(result["evaluators"]) == 1
        ev = result["evaluators"][0]
        assert ev["name"] == "safety"
        assert ev["type"] == "judge"
        assert ev["phase"] == "post_call"
        assert ev["config"] == {}

    def test_judge_rules_key_passes_through(self):
        """Judge evaluator with canonical rules key is preserved in config."""
        source = self._build_source({
            "evaluators": [
                {
                    "name": "safety",
                    "type": "judge",
                    "phase": "pre_call",
                    "rules": ["No harmful content", "No PII leaks"],
                },
            ],
        })
        result = source._map_evaluators(source._yaml_data)
        ev = result["evaluators"][0]
        assert ev["config"]["rules"] == ["No harmful content", "No PII leaks"]

    def test_judge_rules_file_resolved(self):
        """rules_file is resolved relative to the config file."""
        from pathlib import Path
        config_file = Path("/etc/openbias/openbias.yaml")
        source = self._build_source({
            "evaluators": [
                {
                    "name": "behavior",
                    "type": "judge",
                    "phase": "post_call",
                    "rules_file": "./rules.md",
                },
            ],
        }, config_file=config_file)
        result = source._map_evaluators(source._yaml_data)
        ev = result["evaluators"][0]
        assert ev["config"]["rules_file"] == "/etc/openbias/rules.md"

    def test_judge_extra_keys_in_config(self):
        """Extra keys on a judge evaluator go into config dict."""
        source = self._build_source({
            "evaluators": [
                {
                    "name": "safety",
                    "type": "judge",
                    "phase": "post_call",
                    "pass_threshold": 0.6,
                    "temperature": 0.0,
                },
            ],
        })
        result = source._map_evaluators(source._yaml_data)
        ev = result["evaluators"][0]
        assert ev["config"]["pass_threshold"] == 0.6
        assert ev["config"]["temperature"] == 0.0

    def test_fsm_evaluator_rules_file_resolved(self):
        """FSM evaluator with rules_file gets resolved relative to config file."""
        from pathlib import Path
        config_file = Path("/etc/openbias/openbias.yaml")
        source = self._build_source(
            {
                "evaluators": [
                    {
                        "name": "workflow",
                        "type": "fsm",
                        "phase": "post_call",
                        "rules_file": "./workflow-rules.md",
                    },
                ],
            },
            config_file=config_file,
        )
        result = source._map_evaluators(source._yaml_data)
        ev = result["evaluators"][0]
        assert ev["config"]["rules_file"] == "/etc/openbias/workflow-rules.md"

    def test_nemo_evaluator_rules_file_resolved(self):
        """NeMo evaluator with rules_file gets resolved relative to config file."""
        from pathlib import Path
        config_file = Path("/etc/openbias/openbias.yaml")
        source = self._build_source(
            {
                "evaluators": [
                    {
                        "name": "nemo-rails",
                        "type": "nemo",
                        "phase": "post_call",
                        "rules_file": "./nemo-rules.md",
                    },
                ],
            },
            config_file=config_file,
        )
        result = source._map_evaluators(source._yaml_data)
        ev = result["evaluators"][0]
        assert ev["config"]["rules_file"] == "/etc/openbias/nemo-rules.md"

    def test_fsm_evaluator_extra_keys(self):
        """FSM evaluator extra keys go into config."""
        source = self._build_source({
            "evaluators": [
                {
                    "name": "workflow",
                    "type": "fsm",
                    "phase": "post_call",
                    "rules_file": "/abs/workflow-rules.md",
                    "max_steps": 10,
                },
            ],
        })
        result = source._map_evaluators(source._yaml_data)
        ev = result["evaluators"][0]
        assert ev["config"]["rules_file"] == "/abs/workflow-rules.md"
        assert ev["config"]["max_steps"] == 10

    def test_multiple_evaluators(self):
        """Multiple evaluators (pre_call + post_call) are parsed correctly."""
        source = self._build_source({
            "evaluators": [
                {
                    "name": "pre-screen",
                    "type": "judge",
                    "phase": "pre_call",
                    "rules": ["No harmful content"],
                },
                {
                    "name": "post-eval",
                    "type": "judge",
                    "phase": "post_call",
                    "rules_file": "/path/to/post-rules.md",
                },
                {
                    "name": "workflow",
                    "type": "fsm",
                    "phase": "post_call",
                    "rules_file": "/path/to/workflow-rules.md",
                },
            ],
        })
        result = source._map_evaluators(source._yaml_data)
        assert len(result["evaluators"]) == 3

        pre = result["evaluators"][0]
        assert pre["name"] == "pre-screen"
        assert pre["phase"] == "pre_call"
        assert pre["config"]["rules"] == ["No harmful content"]

        post = result["evaluators"][1]
        assert post["name"] == "post-eval"
        assert post["config"]["rules_file"] == "/path/to/post-rules.md"

        fsm = result["evaluators"][2]
        assert fsm["type"] == "fsm"
        assert fsm["config"]["rules_file"] == "/path/to/workflow-rules.md"

    @pytest.mark.parametrize("legacy_key", ["policy", "policies", "rubric", "workflow"])
    def test_legacy_top_level_keys_fail_fast(self, legacy_key):
        source = self._build_source({"evaluators": [], legacy_key: "legacy"})
        with pytest.raises(ValueError, match=f"Legacy key `{legacy_key}` is no longer supported"):
            source._map_evaluators(source._yaml_data)

    @pytest.mark.parametrize("legacy_key", ["policy", "policies", "rubric", "workflow"])
    def test_legacy_evaluator_keys_fail_fast(self, legacy_key):
        source = self._build_source({
            "evaluators": [
                {"name": "safety", "type": "judge", "phase": "post_call", legacy_key: "legacy"}
            ]
        })
        with pytest.raises(ValueError, match=f"Legacy key `{legacy_key}` is no longer supported"):
            source._map_evaluators(source._yaml_data)

    def test_no_policy_key_in_result(self):
        """New format path should NOT populate result['policy']."""
        source = self._build_source({
            "mode": "async",
            "fail_action": "intervene",
            "evaluators": [
                {"name": "safety", "type": "judge", "phase": "post_call"},
            ],
        })
        result = source._map_evaluators(source._yaml_data)
        assert "policy" not in result


# =========================================================================
# get_policy_config() tests
# =========================================================================


class TestGetPolicyConfig:
    """Tests for Settings.get_policy_config() using the evaluators-based format."""

    def test_empty_evaluators_returns_default_judge(self):
        """When evaluators is empty, return default judge config."""
        settings = Settings()
        result = settings.get_policy_config()
        assert result["type"] == "judge"
        assert result["enabled"] is True
        assert result["config"] == {}
        assert result["config_path"] is None

    def test_judge_evaluator_no_models_injects_proxy_model(self):
        """When judge evaluator has no models, inject from proxy.default_model."""
        settings = Settings(
            evaluators=[
                {"name": "safety", "type": "judge", "phase": "post_call"},
            ],
            proxy={"default_model": "gpt-4o-mini"},
        )
        result = settings.get_policy_config()
        assert result["type"] == "judge"
        assert result["config"]["models"] == [
            {"name": "primary", "model": "gpt-4o-mini"}
        ]

    def test_judge_evaluator_models_already_set_no_override(self):
        """When judge evaluator already has models set, don't override."""
        settings = Settings(
            evaluators=[
                {
                    "name": "safety",
                    "type": "judge",
                    "phase": "post_call",
                    "config": {
                        "models": [{"name": "primary", "model": "anthropic/claude-sonnet-4-5"}]
                    },
                },
            ],
            proxy={"default_model": "gpt-4o-mini"},
        )
        result = settings.get_policy_config()
        assert result["config"]["models"] == [
            {"name": "primary", "model": "anthropic/claude-sonnet-4-5"}
        ]

    def test_non_judge_evaluator_no_models_injection(self):
        """When evaluator is fsm type, no models injection occurs."""
        settings = Settings(
            evaluators=[
                {
                    "name": "workflow",
                    "type": "fsm",
                    "phase": "post_call",
                    "config": {"rules_file": "/abs/workflow-rules.md"},
                },
            ],
            proxy={"default_model": "gpt-4o-mini"},
        )
        result = settings.get_policy_config()
        assert result["type"] == "fsm"
        assert "models" not in result["config"]
        assert result["config_path"] is None


# =========================================================================
# async + block normalization tests
# =========================================================================


class TestAsyncBlockNormalization:
    """Verify that async + block is normalized to intervene at config time."""

    def test_async_block_normalized_to_intervene(self):
        with pytest.warns(UserWarning, match="fail_action='block' has no effect in async mode"):
            settings = Settings(mode="async", fail_action="block")
        assert settings.fail_action == "intervene"

    def test_sync_block_unchanged(self):
        settings = Settings(mode="sync", fail_action="block")
        assert settings.fail_action == "block"

    def test_async_shadow_unchanged(self):
        settings = Settings(mode="async", fail_action="shadow")
        assert settings.fail_action == "shadow"

    def test_async_intervene_unchanged(self):
        settings = Settings(mode="async", fail_action="intervene")
        assert settings.fail_action == "intervene"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
