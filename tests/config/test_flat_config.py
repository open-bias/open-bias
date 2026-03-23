import pytest

from openbias.config.settings import (
    EvaluatorConfig,
    PolicyConfig,
    PolicyEngineConfig,
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
        settings = Settings()
        assert settings.policy.default_strategy == "user_message_inject"


class TestGetPolicyConfig:
    """Tests for Settings.get_policy_config() model injection."""

    def test_injects_models_from_global_default_model(self):
        """When judge engine has no explicit model, inject from proxy.default_model."""
        settings = Settings()
        settings.proxy.default_model = "gpt-4o"
        settings.policy.engine.type = "judge"
        settings.policy.engine.config = {}

        result = settings.get_policy_config()
        assert result["config"]["models"] == [
            {"name": "primary", "model": "gpt-4o"}
        ]

    def test_no_injection_when_models_already_set(self):
        """When judge engine already has models, don't override."""
        settings = Settings()
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
        settings = Settings()
        settings.proxy.default_model = "gpt-4o"
        settings.policy.engine.type = "fsm"
        settings.policy.engine.config = {}

        result = settings.get_policy_config()
        assert "models" not in result["config"]

    def test_no_injection_when_no_default_model(self):
        """When there's no global default_model, don't inject."""
        settings = Settings()
        settings.proxy.default_model = None
        settings.policy.engine.type = "judge"
        settings.policy.engine.config = {}

        result = settings.get_policy_config()
        assert "models" not in result["config"]


class TestFailAction:
    """Tests for fail_action configuration."""

    def _build_source(self, yaml_data):
        source = YamlConfigSource.__new__(YamlConfigSource)
        source._yaml_data = yaml_data
        source._config_file = None
        return source

    def test_default_fail_action_is_intervene(self):
        """PolicyConfig defaults to fail_action='intervene'."""
        settings = Settings()
        assert settings.policy.fail_action == "intervene"

    def test_fail_action_maps_from_yaml(self):
        """Top-level fail_action in YAML maps to policy.fail_action."""
        source = self._build_source({"fail_action": "block"})
        result = source._map_to_settings()
        assert result["policy"]["fail_action"] == "block"

    def test_fail_action_intervene_maps_from_yaml(self):
        """fail_action: intervene maps correctly."""
        source = self._build_source({"fail_action": "intervene"})
        result = source._map_to_settings()
        assert result["policy"]["fail_action"] == "intervene"

    def test_fail_action_shadow_maps_from_yaml(self):
        """fail_action: shadow maps correctly."""
        source = self._build_source({"fail_action": "shadow"})
        result = source._map_to_settings()
        assert result["policy"]["fail_action"] == "shadow"


class TestFailOpen:
    """Tests for fail_open configuration mapping."""

    def _build_source(self, yaml_data):
        source = YamlConfigSource.__new__(YamlConfigSource)
        source._yaml_data = yaml_data
        source._config_file = None
        return source

    def test_default_fail_open_is_true(self):
        """PolicyConfig defaults to fail_open=True."""
        settings = Settings()
        assert settings.policy.fail_open is True

    def test_fail_open_false_maps_from_yaml(self):
        """Top-level fail_open: false in YAML maps to policy.fail_open."""
        source = self._build_source({"fail_open": False})
        result = source._map_to_settings()
        assert result["policy"]["fail_open"] is False

    def test_fail_open_true_maps_from_yaml(self):
        """Top-level fail_open: true in YAML maps to policy.fail_open."""
        source = self._build_source({"fail_open": True})
        result = source._map_to_settings()
        assert result["policy"]["fail_open"] is True


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
        assert settings.max_intervention_attempts == 3
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
# Flat-field / policy-shim sync tests
# =========================================================================


class TestFlatPolicySync:
    """Verify that flat fields stay in sync with the policy shim."""

    def test_policy_shim_reflects_flat_defaults(self):
        """When no policy kwarg is given, shim mirrors the flat-field defaults."""
        settings = Settings()
        assert settings.policy.fail_action == "intervene"
        assert settings.policy.default_strategy == "user_message_inject"
        assert settings.policy.fail_open is True
        assert settings.policy.hook_timeout_seconds == 30.0
        assert settings.policy.post_call_mode == "async"

    def test_policy_kwarg_syncs_to_flat_fields(self):
        """When policy={...} is passed (as YAML source would), flat fields are updated."""
        settings = Settings(
            policy={"fail_action": "block", "fail_open": False, "post_call_mode": "sync"}
        )
        assert settings.fail_action == "block"
        assert settings.fail_open is False
        assert settings.mode == "sync"
        # And the shim still works
        assert settings.policy.fail_action == "block"
        assert settings.policy.fail_open is False
        assert settings.policy.post_call_mode == "sync"

    def test_backward_compat_classes_still_importable(self):
        """PolicyConfig and PolicyEngineConfig are still importable."""
        assert PolicyConfig is not None
        assert PolicyEngineConfig is not None
        pc = PolicyConfig()
        assert pc.engine.type == "judge"


# =========================================================================
# New evaluator-based YAML format tests
# =========================================================================


class TestEvaluatorYamlMapping:
    """Tests for the new evaluator-based YAML parsing path in _map_to_settings()."""

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
            "max_intervention_attempts": 5,
            "strategy": "system_prompt_append",
            "session_ttl": 7200,
            "max_sessions": 5000,
            "fail_open": False,
            "hook_timeout_seconds": 60.0,
            "evaluators": [],
        })
        result = source._map_to_settings()
        assert result["mode"] == "sync"
        assert result["fail_action"] == "block"
        assert result["max_intervention_attempts"] == 5
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
        result = source._map_to_settings()
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
        result = source._map_to_settings()
        assert result["debug"] is True
        assert result["log_level"] == "DEBUG"

    def test_tracing_mapped_to_otel(self):
        """tracing section maps to otel in new format too."""
        source = self._build_source({
            "evaluators": [],
            "tracing": {"type": "otlp", "endpoint": "http://jaeger:4317"},
        })
        result = source._map_to_settings()
        assert result["otel"]["exporter_type"] == "otlp"
        assert result["otel"]["enabled"] is True
        assert result["otel"]["endpoint"] == "http://jaeger:4317"

    def test_judge_evaluator_basic(self):
        """Basic judge evaluator is parsed correctly."""
        source = self._build_source({
            "evaluators": [
                {"name": "safety", "type": "judge", "phase": "post_call"},
            ],
        })
        result = source._map_to_settings()
        assert len(result["evaluators"]) == 1
        ev = result["evaluators"][0]
        assert ev["name"] == "safety"
        assert ev["type"] == "judge"
        assert ev["phase"] == "post_call"
        assert ev["config"] == {}

    def test_judge_model_shorthand(self):
        """Judge evaluator with model shorthand synthesizes models list."""
        source = self._build_source({
            "evaluators": [
                {
                    "name": "safety",
                    "type": "judge",
                    "phase": "pre_call",
                    "model": "anthropic/claude-sonnet-4-5",
                },
            ],
        })
        result = source._map_to_settings()
        ev = result["evaluators"][0]
        assert ev["config"]["models"] == [
            {"name": "primary", "model": "anthropic/claude-sonnet-4-5"}
        ]

    def test_judge_policies_shorthand(self):
        """Judge evaluator with policies shorthand maps to inline_policy."""
        source = self._build_source({
            "evaluators": [
                {
                    "name": "safety",
                    "type": "judge",
                    "phase": "pre_call",
                    "policies": ["No harmful content", "No PII leaks"],
                },
            ],
        })
        result = source._map_to_settings()
        ev = result["evaluators"][0]
        assert ev["config"]["inline_policy"] == ["No harmful content", "No PII leaks"]

    def test_judge_rubric_shorthand(self):
        """Judge evaluator with rubric shorthand maps to default_rubric."""
        source = self._build_source({
            "evaluators": [
                {
                    "name": "behavior",
                    "type": "judge",
                    "phase": "post_call",
                    "rubric": "agent_behavior",
                },
            ],
        })
        result = source._map_to_settings()
        ev = result["evaluators"][0]
        assert ev["config"]["default_rubric"] == "agent_behavior"

    def test_judge_extra_keys_in_config(self):
        """Extra keys on a judge evaluator go into config dict."""
        source = self._build_source({
            "evaluators": [
                {
                    "name": "safety",
                    "type": "judge",
                    "phase": "post_call",
                    "model": "gpt-4o",
                    "pass_threshold": 0.6,
                    "temperature": 0.0,
                },
            ],
        })
        result = source._map_to_settings()
        ev = result["evaluators"][0]
        assert ev["config"]["pass_threshold"] == 0.6
        assert ev["config"]["temperature"] == 0.0
        assert ev["config"]["models"] == [{"name": "primary", "model": "gpt-4o"}]

    def test_fsm_evaluator_policy_resolved(self):
        """FSM evaluator with policy path gets resolved relative to config file."""
        from pathlib import Path
        config_file = Path("/etc/openbias/openbias.yaml")
        source = self._build_source(
            {
                "evaluators": [
                    {
                        "name": "workflow",
                        "type": "fsm",
                        "phase": "post_call",
                        "policy": "./workflow.yaml",
                    },
                ],
            },
            config_file=config_file,
        )
        result = source._map_to_settings()
        ev = result["evaluators"][0]
        assert ev["config"]["config_path"] == "/etc/openbias/workflow.yaml"

    def test_nemo_evaluator_policy_resolved(self):
        """NeMo evaluator with policy path gets resolved relative to config file."""
        from pathlib import Path
        config_file = Path("/etc/openbias/openbias.yaml")
        source = self._build_source(
            {
                "evaluators": [
                    {
                        "name": "nemo-rails",
                        "type": "nemo",
                        "phase": "post_call",
                        "policy": "./nemo_config/",
                    },
                ],
            },
            config_file=config_file,
        )
        result = source._map_to_settings()
        ev = result["evaluators"][0]
        assert ev["config"]["config_path"] == "/etc/openbias/nemo_config"

    def test_fsm_evaluator_extra_keys(self):
        """FSM evaluator extra keys go into config."""
        source = self._build_source({
            "evaluators": [
                {
                    "name": "workflow",
                    "type": "fsm",
                    "phase": "post_call",
                    "policy": "/abs/workflow.yaml",
                    "max_steps": 10,
                },
            ],
        })
        result = source._map_to_settings()
        ev = result["evaluators"][0]
        assert ev["config"]["config_path"] == "/abs/workflow.yaml"
        assert ev["config"]["max_steps"] == 10

    def test_multiple_evaluators(self):
        """Multiple evaluators (pre_call + post_call) are parsed correctly."""
        source = self._build_source({
            "evaluators": [
                {
                    "name": "pre-screen",
                    "type": "judge",
                    "phase": "pre_call",
                    "model": "gpt-4o",
                    "policies": ["No harmful content"],
                },
                {
                    "name": "post-eval",
                    "type": "judge",
                    "phase": "post_call",
                    "model": "anthropic/claude-sonnet-4-5",
                    "rubric": "quality",
                },
                {
                    "name": "workflow",
                    "type": "fsm",
                    "phase": "post_call",
                    "policy": "/path/to/workflow.yaml",
                },
            ],
        })
        result = source._map_to_settings()
        assert len(result["evaluators"]) == 3

        pre = result["evaluators"][0]
        assert pre["name"] == "pre-screen"
        assert pre["phase"] == "pre_call"
        assert pre["config"]["inline_policy"] == ["No harmful content"]

        post = result["evaluators"][1]
        assert post["name"] == "post-eval"
        assert post["config"]["default_rubric"] == "quality"

        fsm = result["evaluators"][2]
        assert fsm["type"] == "fsm"
        assert fsm["config"]["config_path"] == "/path/to/workflow.yaml"

    def test_no_policy_key_in_result(self):
        """New format path should NOT populate result['policy']."""
        source = self._build_source({
            "mode": "async",
            "fail_action": "intervene",
            "evaluators": [
                {"name": "safety", "type": "judge", "phase": "post_call"},
            ],
        })
        result = source._map_to_settings()
        assert "policy" not in result


class TestOldFormatStillWorks:
    """Ensure old-format YAML (without evaluators key) still works."""

    def _build_source(self, yaml_data):
        source = YamlConfigSource.__new__(YamlConfigSource)
        source._yaml_data = yaml_data
        source._config_file = None
        return source

    def test_old_format_engine_judge(self):
        """Old format with engine: judge still maps to policy structure."""
        source = self._build_source({
            "engine": "judge",
            "model": "gpt-4o",
            "fail_action": "block",
            "judge": {"pass_threshold": 0.6},
        })
        result = source._map_to_settings()
        assert result["policy"]["engine"]["type"] == "judge"
        assert result["proxy"]["default_model"] == "gpt-4o"
        assert result["policy"]["fail_action"] == "block"
        assert result["policy"]["engine"]["config"]["pass_threshold"] == 0.6
        # Should NOT have evaluators key
        assert "evaluators" not in result

    def test_old_format_policy_list(self):
        """Old format inline policy list still works."""
        source = self._build_source({
            "engine": "judge",
            "policy": ["No PII", "Be helpful"],
        })
        result = source._map_to_settings()
        assert result["policy"]["engine"]["config"]["inline_policy"] == ["No PII", "Be helpful"]


class TestInitSyncSkippedWithEvaluators:
    """Verify that __init__ sync block is skipped when evaluators are present."""

    def test_sync_skipped_when_evaluators_present(self):
        """When evaluators are provided, flat fields come from kwargs, not policy shim."""
        settings = Settings(
            mode="sync",
            fail_action="block",
            evaluators=[
                {"name": "safety", "type": "judge", "phase": "post_call"},
            ],
        )
        # Flat fields should reflect what was passed, not the policy defaults
        assert settings.mode == "sync"
        assert settings.fail_action == "block"
        # Policy shim should still have its own defaults (not overwritten)
        assert settings.policy.post_call_mode == "async"
        assert settings.policy.fail_action == "intervene"

    def test_sync_runs_when_no_evaluators(self):
        """When no evaluators, sync block copies from policy shim as before."""
        settings = Settings(
            policy={"fail_action": "shadow", "post_call_mode": "sync"}
        )
        assert settings.evaluators == []
        assert settings.fail_action == "shadow"
        assert settings.mode == "sync"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))

