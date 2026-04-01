"""
Tests for YamlConfigSource._map_evaluators() — tracing, intervention, and classifier
YAML schema coverage using the evaluator-based format.
"""

import pytest

from openbias.config.settings import YamlConfigSource


def _build_source(yaml_data):
    """Create a YamlConfigSource with injected YAML data (no file I/O)."""
    source = YamlConfigSource.__new__(YamlConfigSource)
    source._yaml_data = yaml_data
    source._config_file = None
    return source


# =========================================================================
# Tracing section
# =========================================================================


class TestTracingMapping:
    def _map(self, data):
        return _build_source(data)._map_evaluators(data)

    def test_tracing_otlp(self):
        result = self._map({
            "evaluators": [],
            "tracing": {
                "type": "otlp",
                "endpoint": "http://jaeger:4317",
                "service_name": "my-service",
            },
        })
        otel = result["otel"]
        assert otel["exporter_type"] == "otlp"
        assert otel["endpoint"] == "http://jaeger:4317"
        assert otel["service_name"] == "my-service"

    def test_tracing_not_configured(self):
        """No tracing section means otel stays at defaults (disabled)."""
        result = self._map({"evaluators": []})
        assert "otel" not in result

    def test_tracing_console(self):
        result = self._map({"evaluators": [], "tracing": {"type": "console"}})
        assert result["otel"]["exporter_type"] == "console"

    def test_tracing_insecure(self):
        result = self._map(
            {"evaluators": [], "tracing": {"type": "otlp", "insecure": False}}
        )
        assert result["otel"]["insecure"] is False

    def test_tracing_redact_content(self):
        result = self._map(
            {"evaluators": [], "tracing": {"type": "otlp", "redact_content": True}}
        )
        assert result["otel"]["redact_content"] is True

    def test_tracing_redact_content_false(self):
        result = self._map(
            {"evaluators": [], "tracing": {"type": "otlp", "redact_content": False}}
        )
        assert result["otel"]["redact_content"] is False

    def test_tracing_langfuse_complete(self):
        data = {
            "evaluators": [],
            "tracing": {
                "type": "langfuse",
                "endpoint": "http://localhost:4317",
                "service_name": "test-svc",
                "langfuse_public_key": "pk-test-123",
                "langfuse_secret_key": "sk-test-456",
                "langfuse_host": "https://us.cloud.langfuse.com",
            },
        }
        result = _build_source(data)._map_evaluators(data)
        otel = result["otel"]
        assert otel["exporter_type"] == "langfuse"
        assert otel["endpoint"] == "http://localhost:4317"
        assert otel["service_name"] == "test-svc"
        assert otel["langfuse_public_key"] == "pk-test-123"
        assert otel["langfuse_secret_key"] == "sk-test-456"
        assert otel["langfuse_host"] == "https://us.cloud.langfuse.com"


# =========================================================================
# Intervention fields (now flat top-level keys in evaluator format)
# =========================================================================


class TestInterventionMapping:
    def test_intervention_flat_fields(self):
        """Intervention settings are flat top-level keys in the evaluator format."""
        data = {
            "evaluators": [],
            "strategy": "user_message_inject",
        }
        result = _build_source(data)._map_evaluators(data)
        assert result["strategy"] == "user_message_inject"

    def test_intervention_defaults_not_present_when_omitted(self):
        """Omitted intervention fields are not forced into result."""
        data = {"evaluators": []}
        result = _build_source(data)._map_evaluators(data)
        assert "strategy" not in result


# =========================================================================
# Classifier — passed via evaluator config dict
# =========================================================================


class TestClassifierMapping:
    def test_classifier_in_fsm_evaluator_config(self):
        """Classifier config can be passed inside an fsm evaluator's config dict."""
        data = {
            "evaluators": [
                {
                    "name": "workflow",
                    "type": "fsm",
                    "phase": "post_call",
                    "policy": "/abs/workflow.yaml",
                    "classifier": {
                        "model_name": "all-MiniLM-L12-v2",
                        "backend": "onnx",
                        "similarity_threshold": 0.85,
                        "cache_embeddings": False,
                        "device": "cuda",
                    },
                },
            ]
        }
        result = _build_source(data)._map_evaluators(data)
        ev = result["evaluators"][0]
        assert ev["config"]["classifier"]["model_name"] == "all-MiniLM-L12-v2"
        assert ev["config"]["classifier"]["backend"] == "onnx"
        assert ev["config"]["classifier"]["similarity_threshold"] == 0.85
        assert ev["config"]["classifier"]["cache_embeddings"] is False
        assert ev["config"]["classifier"]["device"] == "cuda"
