"""
Tests for YamlConfigSource._map_to_settings() — full YAML schema coverage.

DEPRECATED: _map_to_settings() was removed as part of the backward-compatibility
removal. The old single-engine YAML format (engine:, judge:, policy: top-level keys)
is no longer supported. Use the evaluators-based format instead.

These tests are skipped pending removal or replacement with evaluator-format tests.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="REMOVED: _map_to_settings() deleted — old single-engine YAML format no longer supported"
)

from openbias.config.settings import YamlConfigSource


def _build_source(yaml_data):
    """Create a YamlConfigSource with injected YAML data (no file I/O)."""
    source = YamlConfigSource.__new__(YamlConfigSource)
    source._yaml_data = yaml_data
    source._config_file = None
    return source


def _get_engine_config(result):
    """Extract the engine config dict from a mapped result."""
    return result.get("policy", {}).get("engine", {}).get("config", {})


# =========================================================================
# Global / top-level keys
# =========================================================================


class TestGlobalKeys:
    def test_engine_maps_to_policy_engine_type(self):
        result = _build_source({"engine": "judge"})._map_to_settings()
        assert result["policy"]["engine"]["type"] == "judge"

    def test_model_maps_to_proxy_default_model(self):
        result = _build_source({"model": "gemini/gemini-2.5-flash"})._map_to_settings()
        assert result["proxy"]["default_model"] == "gemini/gemini-2.5-flash"

    def test_port_maps_to_proxy_port(self):
        result = _build_source({"port": 5000})._map_to_settings()
        assert result["proxy"]["port"] == 5000

    def test_host_maps_to_proxy_host(self):
        result = _build_source({"host": "127.0.0.1"})._map_to_settings()
        assert result["proxy"]["host"] == "127.0.0.1"

    def test_debug_maps_directly(self):
        result = _build_source({"debug": True})._map_to_settings()
        assert result["debug"] is True

    def test_log_level_maps_directly(self):
        result = _build_source({"log_level": "DEBUG"})._map_to_settings()
        assert result["log_level"] == "DEBUG"

    def test_empty_yaml_returns_empty(self):
        result = _build_source({})._map_to_settings()
        assert result == {}

    def test_none_yaml_returns_empty(self):
        result = _build_source(None)._map_to_settings()
        assert result == {}


# =========================================================================
# Judge engine section
# =========================================================================


class TestJudgeMapping:
    def test_judge_model_creates_models_list(self):
        result = _build_source(
            {
                "engine": "judge",
                "judge": {"model": "gpt-4o-mini"},
            }
        )._map_to_settings()
        cfg = _get_engine_config(result)
        assert cfg["models"] == [{"name": "primary", "model": "gpt-4o-mini"}]

    def test_judge_passthrough_keys(self):
        """Keys beyond model should pass through directly."""
        result = _build_source(
            {
                "engine": "judge",
                "judge": {
                    "fail_action": "intervene",
                    "pre_call_enabled": True,
                    "pre_call_rubric": "safety",
                    "default_rubric": "custom_rubric",
                    "conversation_rubric": "convo_rubric",
                    "conversation_eval_interval": 3,
                    "custom_rubrics_path": "./rubrics/",
                },
            }
        )._map_to_settings()
        cfg = _get_engine_config(result)

        assert cfg["fail_action"] == "intervene"
        assert cfg["pre_call_enabled"] is True
        assert cfg["pre_call_rubric"] == "safety"
        assert cfg["default_rubric"] == "custom_rubric"
        assert cfg["conversation_rubric"] == "convo_rubric"
        assert cfg["conversation_eval_interval"] == 3
        assert cfg["custom_rubrics_path"] == "./rubrics/"

    def test_judge_advanced_models_list(self):
        """Advanced multi-model config should pass through."""
        models = [
            {"name": "primary", "model": "gpt-4o-mini", "temperature": 0.0},
            {"name": "secondary", "model": "anthropic/claude-sonnet-4-5"},
        ]
        result = _build_source(
            {
                "engine": "judge",
                "judge": {"models": models},
            }
        )._map_to_settings()
        cfg = _get_engine_config(result)
        assert cfg["models"] == models


# =========================================================================
# LLM engine section
# =========================================================================


class TestLLMMapping:
    def test_llm_model_renames_to_llm_model(self):
        """llm.model -> llm_model (what LLMPolicyEngine.initialize expects)."""
        result = _build_source(
            {
                "engine": "llm",
                "llm": {"model": "gemini/gemini-2.5-flash"},
            }
        )._map_to_settings()
        cfg = _get_engine_config(result)
        assert cfg["llm_model"] == "gemini/gemini-2.5-flash"
        assert "model" not in cfg

    def test_llm_passthrough_keys(self):
        result = _build_source(
            {
                "engine": "llm",
                "llm": {
                    "temperature": 0.1,
                    "max_tokens": 2048,
                    "timeout": 15.0,
                    "confident_threshold": 0.9,
                    "uncertain_threshold": 0.4,
                    "temporal_weight": 0.6,
                    "cooldown_turns": 3,
                    "max_constraints_per_batch": 10,
                },
            }
        )._map_to_settings()
        cfg = _get_engine_config(result)

        assert cfg["temperature"] == 0.1
        assert cfg["max_tokens"] == 2048
        assert cfg["timeout"] == 15.0
        assert cfg["confident_threshold"] == 0.9
        assert cfg["uncertain_threshold"] == 0.4
        assert cfg["temporal_weight"] == 0.6
        assert cfg["cooldown_turns"] == 3
        assert cfg["max_constraints_per_batch"] == 10


# =========================================================================
# NeMo engine section
# =========================================================================


class TestNemoMapping:
    def test_nemo_fail_closed(self):
        result = _build_source(
            {
                "engine": "nemo",
                "nemo": {"fail_closed": True},
            }
        )._map_to_settings()
        cfg = _get_engine_config(result)
        assert cfg["fail_closed"] is True

    def test_nemo_rails(self):
        result = _build_source(
            {
                "engine": "nemo",
                "nemo": {"rails": ["input", "output"]},
            }
        )._map_to_settings()
        cfg = _get_engine_config(result)
        assert cfg["rails"] == ["input", "output"]


# =========================================================================
# Composite engine section
# =========================================================================


# =========================================================================
# FSM engine section (future extensibility)
# =========================================================================


class TestFSMMapping:
    def test_fsm_empty_section_no_crash(self):
        """Empty fsm: section should not break anything."""
        result = _build_source(
            {
                "engine": "fsm",
                "fsm": {},
            }
        )._map_to_settings()
        # Should have engine type but no config entries from fsm section
        assert result["policy"]["engine"]["type"] == "fsm"

    def test_fsm_future_keys_pass_through(self):
        """Any future keys added to fsm: should pass through."""
        result = _build_source(
            {
                "engine": "fsm",
                "fsm": {"some_future_setting": 42},
            }
        )._map_to_settings()
        cfg = _get_engine_config(result)
        assert cfg["some_future_setting"] == 42


# =========================================================================
# Tracing section
# =========================================================================


class TestTracingMapping:
    def test_tracing_otlp(self):
        result = _build_source(
            {
                "tracing": {
                    "type": "otlp",
                    "endpoint": "http://jaeger:4317",
                    "service_name": "my-service",
                },
            }
        )._map_to_settings()
        otel = result["otel"]
        assert otel["exporter_type"] == "otlp"
        assert otel["enabled"] is True
        assert otel["endpoint"] == "http://jaeger:4317"
        assert otel["service_name"] == "my-service"

    def test_tracing_none_disables(self):
        result = _build_source({"tracing": {"type": "none"}})._map_to_settings()
        assert result["otel"]["enabled"] is False

    def test_tracing_console(self):
        result = _build_source({"tracing": {"type": "console"}})._map_to_settings()
        assert result["otel"]["exporter_type"] == "console"
        assert result["otel"]["enabled"] is True

    def test_tracing_insecure(self):
        result = _build_source(
            {
                "tracing": {"type": "otlp", "insecure": False},
            }
        )._map_to_settings()
        assert result["otel"]["insecure"] is False

    def test_tracing_redact_content(self):
        result = _build_source(
            {"tracing": {"type": "otlp", "redact_content": True}}
        )._map_to_settings()
        assert result["otel"]["redact_content"] is True

    def test_tracing_redact_content_false(self):
        result = _build_source(
            {"tracing": {"type": "otlp", "redact_content": False}}
        )._map_to_settings()
        assert result["otel"]["redact_content"] is False

    def test_tracing_langfuse_complete(self):
        result = _build_source(
            {
                "tracing": {
                    "type": "langfuse",
                    "endpoint": "http://localhost:4317",
                    "service_name": "test-svc",
                    "langfuse_public_key": "pk-test-123",
                    "langfuse_secret_key": "sk-test-456",
                    "langfuse_host": "https://us.cloud.langfuse.com",
                },
            }
        )._map_to_settings()
        otel = result["otel"]
        assert otel["exporter_type"] == "langfuse"
        assert otel["enabled"] is True
        assert otel["endpoint"] == "http://localhost:4317"
        assert otel["service_name"] == "test-svc"
        assert otel["langfuse_public_key"] == "pk-test-123"
        assert otel["langfuse_secret_key"] == "sk-test-456"
        assert otel["langfuse_host"] == "https://us.cloud.langfuse.com"


# =========================================================================
# Intervention section
# =========================================================================


class TestInterventionMapping:
    def test_intervention_nested_under_fsm(self):
        result = _build_source(
            {
                "engine": "fsm",
                "fsm": {
                    "intervention": {
                        "default_strategy": "user_message_inject",
                        "max_intervention_attempts": 5,
                        "include_headers": False,
                    },
                },
            }
        )._map_to_settings()
        cfg = _get_engine_config(result)
        assert cfg["intervention"]["default_strategy"] == "user_message_inject"
        assert cfg["intervention"]["max_intervention_attempts"] == 5
        assert cfg["intervention"]["include_headers"] is False

    def test_intervention_nested_under_llm(self):
        result = _build_source(
            {
                "engine": "llm",
                "llm": {
                    "intervention": {
                        "default_strategy": "user_message_inject",
                        "max_intervention_attempts": 3,
                    },
                },
            }
        )._map_to_settings()
        cfg = _get_engine_config(result)
        assert cfg["intervention"]["default_strategy"] == "user_message_inject"
        assert cfg["intervention"]["max_intervention_attempts"] == 3


# =========================================================================
# Classifier section
# =========================================================================


class TestClassifierMapping:
    def test_classifier_nested_under_fsm(self):
        result = _build_source(
            {
                "engine": "fsm",
                "fsm": {
                    "classifier": {
                        "model_name": "all-MiniLM-L12-v2",
                        "backend": "onnx",
                        "similarity_threshold": 0.85,
                        "cache_embeddings": False,
                        "device": "cuda",
                    },
                },
            }
        )._map_to_settings()
        cfg = _get_engine_config(result)
        assert cfg["classifier"]["model_name"] == "all-MiniLM-L12-v2"
        assert cfg["classifier"]["backend"] == "onnx"
        assert cfg["classifier"]["similarity_threshold"] == 0.85
        assert cfg["classifier"]["cache_embeddings"] is False
        assert cfg["classifier"]["device"] == "cuda"



# =========================================================================
# Combined / integration-style tests
# =========================================================================


class TestCombinedYAML:
    def test_full_judge_config(self):
        """Test a realistic full judge openbias.yaml."""
        result = _build_source(
            {
                "engine": "judge",
                "model": "gemini/gemini-2.5-flash",
                "port": 4000,
                "judge": {
                    "fail_action": "block",
                },
                "policy": ["No PII", "Be professional"],
                "tracing": {"type": "none"},
            }
        )._map_to_settings()

        assert result["policy"]["engine"]["type"] == "judge"
        assert result["proxy"]["default_model"] == "gemini/gemini-2.5-flash"
        assert result["proxy"]["port"] == 4000
        cfg = _get_engine_config(result)
        assert cfg["inline_policy"] == ["No PII", "Be professional"]
        assert cfg["fail_action"] == "block"
        assert result["otel"]["enabled"] is False

    def test_full_llm_config(self):
        """Test a realistic full LLM engine openbias.yaml."""
        result = _build_source(
            {
                "engine": "llm",
                "model": "gpt-4o",
                "port": 5000,
                "policy": "./workflow.yaml",
                "llm": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "cooldown_turns": 3,
                    "intervention": {
                        "default_strategy": "user_message_inject",
                        "max_intervention_attempts": 5,
                    },
                },
            }
        )._map_to_settings()

        assert result["policy"]["engine"]["type"] == "llm"
        assert result["proxy"]["default_model"] == "gpt-4o"
        assert result["proxy"]["port"] == 5000
        assert result["policy"]["engine"]["config_path"] == "./workflow.yaml"
        cfg = _get_engine_config(result)
        assert cfg["llm_model"] == "gpt-4o-mini"
        assert cfg["temperature"] == 0.1
        assert cfg["cooldown_turns"] == 3
        assert cfg["intervention"]["default_strategy"] == "user_message_inject"
        assert cfg["intervention"]["max_intervention_attempts"] == 5

    def test_model_and_judge_model_coexist(self):
        """Top-level model sets proxy default; judge.model overrides for the engine."""
        result = _build_source(
            {
                "engine": "judge",
                "model": "gpt-4o",
                "judge": {"model": "gpt-4o-mini"},
            }
        )._map_to_settings()

        assert result["proxy"]["default_model"] == "gpt-4o"
        cfg = _get_engine_config(result)
        assert cfg["models"] == [{"name": "primary", "model": "gpt-4o-mini"}]
