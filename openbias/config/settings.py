"""
Open Bias configuration management using Pydantic Settings.

Configuration can be provided via:
1. openbias.yaml config file (primary — see config/schema.yaml for full reference)
2. Environment variables (ONLY for API keys like OPENAI_API_KEY, GEMINI_API_KEY, etc.)
3. .env file
4. Direct instantiation

Priority (highest wins): openbias.yaml > API keys > defaults

The simplified openbias.yaml format:
    model: gemini/gemini-2.5-flash   # optional — auto-detected from API keys
    port: 4000
    evaluators:
      - name: safety
        type: judge
        policies:
          - "Must NOT provide financial advice"
          - "Be professional and helpful"
    tracing:
      type: none

For the complete YAML schema reference, see:
    openbias/config/schema.yaml
"""

import contextvars
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, model_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
)

logger = logging.getLogger(__name__)




class OTelConfig(BaseModel):
    """OpenTelemetry tracing configuration.

    Tracing is auto-enabled when a ``type`` is explicitly set or when
    Langfuse API keys are provided.  There is no manual ``enabled`` flag.

    Supported exporters:
    - otlp: Standard OTLP endpoint (Jaeger, Zipkin, etc.)
    - langfuse: Langfuse's OTLP endpoint (requires public_key/secret_key)
    - console: Print traces to console (for debugging)
    """

    model_config = ConfigDict(populate_by_name=True)

    endpoint: str = "http://localhost:4317"
    service_name: str = "openbias"
    exporter_type: Literal["otlp", "langfuse", "console"] | None = None
    insecure: bool = True  # Use insecure connection (no TLS) for local dev

    # Langfuse-specific settings (used when exporter_type="langfuse")
    langfuse_public_key: str | None = Field(None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        "https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )

    # When True, user/assistant message content is replaced with "[REDACTED]" in traces
    redact_content: bool = False

    @property
    def enabled(self) -> bool:
        """Tracing is on when a type is set or Langfuse keys are present."""
        if self.exporter_type is not None:
            return True
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def resolved_exporter_type(self) -> str:
        """Return the effective exporter type, inferring langfuse from keys."""
        if self.exporter_type is not None:
            return self.exporter_type
        if self.langfuse_public_key and self.langfuse_secret_key:
            return "langfuse"
        return "otlp"


class ProxyConfig(BaseModel):
    """LiteLLM proxy server configuration."""

    host: str = "0.0.0.0"
    port: int = 4000
    workers: int = 1
    timeout: int = 600
    master_key: str | None = None
    # Model routing
    default_model: str | None = None
    model_list: list[dict] = Field(default_factory=list)


class ClassifierConfig(BaseModel):
    """State classifier configuration."""

    # Model for semantic similarity
    model_name: str = "all-MiniLM-L6-v2"
    # Use ONNX backend for faster inference (<50ms)
    backend: Literal["pytorch", "onnx"] = "pytorch"
    # Minimum similarity score to consider a match
    similarity_threshold: float = 0.7
    # Cache embeddings for workflow states
    cache_embeddings: bool = True
    # Device for inference
    device: str = "cpu"


class EvaluatorConfig(BaseModel):
    """Configuration for a single evaluator in the pipeline."""
    name: str
    type: str = "judge"
    phase: Literal["pre_call", "post_call"] = "post_call"
    config: dict[str, Any] = Field(default_factory=dict)


class YamlConfigSource(PydanticBaseSettingsSource):
    """Custom settings source that reads from a openbias.yaml config file.

    Discovers config at:
    1. Explicit path passed via _config_path init kwarg
    2. $OBIAS_CONFIG env var
    3. ./openbias.yaml
    4. ./openbias.yml

    Maps simplified YAML keys to the nested Settings structure.
    """

    def __init__(
        self, settings_cls: type[BaseSettings], config_path: str | None = None
    ):
        super().__init__(settings_cls)
        self._config_path = config_path
        self._yaml_data: dict[str, Any] | None = None
        self._load()

    def _discover_config_file(self) -> Path | None:
        """Find the config file to load."""
        if self._config_path:
            p = Path(self._config_path)
            return p if p.is_file() else None

        env_path = os.environ.get("OBIAS_CONFIG")
        if env_path:
            p = Path(env_path)
            return p if p.is_file() else None

        for name in ("openbias.yaml", "openbias.yml"):
            p = Path(name)
            if p.is_file():
                return p

        return None

    def _resolve_path(self, path_str: Any) -> Any:
        """Resolve a path string relative to the config file location."""
        if not isinstance(path_str, str) or not self._config_file:
            return path_str
            
        p = Path(path_str)
        if p.is_absolute():
            return path_str
            
        # Resolve relative to config file directory
        return str(self._config_file.parent / p)

    def _load(self) -> None:
        """Load and parse the YAML config file."""
        path = self._discover_config_file()
        if path is None:
            self._yaml_data = {}
            self._config_file = None
            return
            
        self._config_file = path

        try:
            import yaml

            with open(path) as f:
                data = yaml.safe_load(f)
            self._yaml_data = data if isinstance(data, dict) else {}
            logger.debug(f"Loaded config from {path}")
        except Exception as e:
            # If a specific config path was provided, we must not silent-fail
            if self._config_path:
                logger.error(f"Failed to load config from {self._config_path}: {e}")
                raise

            logger.warning(f"Failed to load config from {path}: {e}")
            self._yaml_data = {}

    # Keys that are handled specially and should NOT be passed through
    # to engine config as generic keys.
    _RESERVED_TOPLEVEL_KEYS = frozenset(
        {
            "port",
            "host",
            "debug",
            "log_level",
            "log_format",
            "model",
            "tracing",
            "eval",
            "fail_action",
            "fail_open",
            "hook_timeout_seconds",
            "mode",
            "evaluators",
            "strategy",
            "session_ttl",
            "max_sessions",
        }
    )

    # Keys extracted from each evaluator entry as EvaluatorConfig fields
    # (everything else goes into the config dict).
    _EVALUATOR_FIELD_KEYS = frozenset({"name", "type", "phase"})

    def _map_common_fields(self, data: dict[str, Any], result: dict[str, Any]) -> None:
        """Map proxy, debug/log_level, and tracing fields shared by both mapping paths."""
        # Proxy fields
        if "port" in data:
            result.setdefault("proxy", {})["port"] = data["port"]
        if "host" in data:
            result.setdefault("proxy", {})["host"] = data["host"]
        if "model" in data:
            result.setdefault("proxy", {})["default_model"] = data["model"]

        # Direct passthrough
        if "debug" in data:
            result["debug"] = data["debug"]
        if "log_level" in data:
            result["log_level"] = data["log_level"]
        if "log_format" in data:
            result["log_format"] = data["log_format"]

        # Tracing (tracing.* -> otel.*)
        tracing_cfg = data.get("tracing", {})
        if isinstance(tracing_cfg, dict) and tracing_cfg:
            otel = result.setdefault("otel", {})
            if "type" in tracing_cfg:
                otel["exporter_type"] = tracing_cfg["type"]
            for k in ("endpoint", "service_name", "insecure",
                      "langfuse_public_key", "langfuse_secret_key",
                      "langfuse_host", "redact_content"):
                if k in tracing_cfg:
                    otel[k] = tracing_cfg[k]

    def _map_evaluators(self, data: dict[str, Any]) -> dict[str, Any]:
        """Map new evaluator-based YAML format to Settings structure.

        When ``evaluators`` key is present, this path is used exclusively.
        It populates flat pipeline fields directly (no policy shim).
        """
        result: dict[str, Any] = {}

        # Direct top-level pipeline fields
        _FLAT_KEYS = (
            "mode", "fail_action", "strategy",
            "session_ttl", "max_sessions", "fail_open", "hook_timeout_seconds",
        )
        for key in _FLAT_KEYS:
            if key in data:
                result[key] = data[key]

        # Shared proxy, debug/log_level, and tracing mapping
        self._map_common_fields(data, result)

        # Build evaluators list
        evaluators: list[dict[str, Any]] = []
        for entry in data.get("evaluators", []):
            if not isinstance(entry, dict):
                continue

            ev: dict[str, Any] = {
                "name": entry.get("name", "unnamed"),
                "type": entry.get("type", "judge"),
                "phase": entry.get("phase", "post_call"),
            }

            # Collect remaining keys into config
            config: dict[str, Any] = {}
            ev_type = ev["type"]

            for k, v in entry.items():
                if k in self._EVALUATOR_FIELD_KEYS:
                    continue

                if ev_type == "judge":
                    if k == "policies":
                        config["inline_policy"] = v
                        continue
                    if k == "rubric":
                        config["default_rubric"] = v
                        continue

                if ev_type in ("fsm", "nemo") and k == "policy":
                    config["config_path"] = self._resolve_path(v)
                    continue

                config[k] = v

            ev["config"] = config
            evaluators.append(ev)

        result["evaluators"] = evaluators
        return result

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        mapped = self._map_evaluators(self._yaml_data or {})
        value = mapped.get(field_name)
        return value, field_name, value is not None

    def __call__(self) -> dict[str, Any]:
        return self._map_evaluators(self._yaml_data or {})


_config_path_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_config_path_var", default=None
)


class Settings(BaseSettings):
    """
    Main Open Bias configuration.

    Configuration can be provided via:
    1. openbias.yaml config file (primary — see config/schema.yaml for full reference)
    2. Environment variables (ONLY for API keys like OPENAI_API_KEY, GEMINI_API_KEY, etc.)
    3. .env file
    4. Direct instantiation

    Priority (highest wins): openbias.yaml > API keys > defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="OBIAS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,  # Allow initializing with field names even if aliases are set
    )

    # _config_path is passed via contextvars to settings_customise_sources

    # General settings
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"
    litellm_verbose: bool = False

    # Component configurations
    otel: OTelConfig = Field(default_factory=OTelConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)

    # --- Evaluator pipeline (flattened from former PolicyConfig) ---
    mode: Literal["sync", "async"] = "async"
    fail_action: Literal["intervene", "block", "shadow"] = "intervene"
    strategy: Literal["system_prompt_append", "user_message_inject"] = "user_message_inject"
    session_ttl: int = 3600
    max_sessions: int = 10000
    fail_open: bool = True
    hook_timeout_seconds: float = 30.0
    evaluators: list[EvaluatorConfig] = Field(default_factory=list)

    # API Keys (loaded from env vars or .env file)
    # We use validation_alias to map standard keys to these fields
    openai_api_key: str | None = Field(None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(None, validation_alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(None, validation_alias="GOOGLE_API_KEY")
    gemini_api_key: str | None = Field(None, validation_alias="GEMINI_API_KEY")
    groq_api_key: str | None = Field(None, validation_alias="GROQ_API_KEY")
    togetherai_api_key: str | None = Field(None, validation_alias="TOGETHERAI_API_KEY")
    openrouter_api_key: str | None = Field(None, validation_alias="OPENROUTER_API_KEY")

    @model_validator(mode="after")
    def _normalize_async_block(self) -> "Settings":
        """Normalize async + block to intervene — block can't work in async mode."""
        if self.mode == "async" and self.fail_action == "block":
            warnings.warn(
                "fail_action='block' has no effect in async mode (the response is "
                "already sent). Normalizing to 'intervene'.",
                UserWarning,
                stacklevel=2,
            )
            self.fail_action = "intervene"
        return self

    def __init__(self, _config_path: str | None = None, **kwargs: Any):
        _token = _config_path_var.set(_config_path)
        try:
            super().__init__(**kwargs)
        finally:
            _config_path_var.reset(_token)

        # Sync API keys to os.environ for downstream libraries (LiteLLM, LangChain)
        # This allows us to use .env files without explicit load_dotenv() in CLI
        self._sync_env_var("OPENAI_API_KEY", self.openai_api_key)
        self._sync_env_var("ANTHROPIC_API_KEY", self.anthropic_api_key)
        self._sync_env_var("GOOGLE_API_KEY", self.google_api_key)
        self._sync_env_var("GEMINI_API_KEY", self.gemini_api_key)
        self._sync_env_var("GROQ_API_KEY", self.groq_api_key)
        self._sync_env_var("TOGETHERAI_API_KEY", self.togetherai_api_key)
        self._sync_env_var("OPENROUTER_API_KEY", self.openrouter_api_key)

        # Auto-detect default model from available API keys when not explicitly set
        if not self.proxy.default_model:
            self.proxy.default_model = self._auto_detect_model()

    def _auto_detect_model(self) -> str | None:
        """Pick a default model based on which API key is available.

        Priority:
          1. OpenAI  -> gpt-4o-mini
          2. Gemini  -> gemini/gemini-2.5-flash
          3. Anthropic -> anthropic/claude-sonnet-4-5
          4. Groq    -> groq/llama3-8b-8192
          5. Together -> together_ai/meta-llama/Llama-3-8b-chat-hf
          6. OpenRouter -> openrouter/auto
        """
        if self.openai_api_key:
            return "gpt-4o-mini"
        if self.google_api_key or self.gemini_api_key:
            return "gemini/gemini-2.5-flash"
        if self.anthropic_api_key:
            return "anthropic/claude-sonnet-4-5"
        if self.groq_api_key:
            return "groq/llama3-8b-8192"
        if self.togetherai_api_key:
            return "together_ai/meta-llama/Llama-3-8b-chat-hf"
        if self.openrouter_api_key:
            return "openrouter/auto"
        return None

    def _sync_env_var(self, key: str, value: str | None) -> None:
        """Set env var if present in settings but missing in os.environ."""
        if value and not os.getenv(key):
            os.environ[key] = value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert YAML config source before env vars.

        Priority (highest first): init > yaml > env > dotenv > file_secret
        """
        yaml_source = YamlConfigSource(settings_cls, config_path=_config_path_var.get())
        return (
            init_settings,
            yaml_source,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    @staticmethod
    def inject_default_model(
        evaluator_type: str, config: dict[str, Any], default_model: str | None
    ) -> None:
        """Inject ``default_model`` into *config* when the engine needs one.

        Mutates *config* in place.  No-op when *default_model* is falsy or the
        engine already has an explicit model configured.
        """
        if not default_model:
            return
        if evaluator_type == "judge" and not config.get("models"):
            config["models"] = [{"name": "primary", "model": default_model}]
        elif evaluator_type == "llm" and not config.get("llm_model"):
            config["llm_model"] = default_model

    def get_policy_config(self) -> dict[str, Any]:
        """
        Get policy engine configuration.

        Returns:
            Configuration dict ready for PolicyEngineRegistry.create_and_initialize()
        """
        if not self.evaluators:
            return {
                "type": "judge",
                "enabled": True,
                "config": {},
                "config_path": None,
            }

        evaluator = self.evaluators[0]
        config = dict(evaluator.config)
        self.inject_default_model(
            evaluator.type, config, self.proxy.default_model
        )

        return {
            "type": evaluator.type,
            "enabled": True,
            "config": config,
            "config_path": config.get("config_path"),
        }

    def get_model_list(self) -> list[dict]:
        """Get model list for LiteLLM router using wildcard routing.

        Returns wildcard entries for providers whose API keys are present,
        allowing LiteLLM to dynamically route any model from those providers.
        """
        # If explicitly configured, use that
        if self.proxy.model_list:
            return self.proxy.model_list

        # Provider wildcard configurations: (model_name, litellm_model, required_env_vars)
        # required_env_vars can be a string or list of strings (any match = enabled)
        providers = [
            ("openai/*", "openai/*", "OPENAI_API_KEY"),
            ("anthropic/*", "anthropic/*", "ANTHROPIC_API_KEY"),
            ("gemini/*", "gemini/*", ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
            ("groq/*", "groq/*", "GROQ_API_KEY"),
            ("together_ai/*", "together_ai/*", "TOGETHERAI_API_KEY"),
            ("openrouter/*", "openrouter/*", "OPENROUTER_API_KEY"),
        ]

        model_list = []
        for model_name, litellm_model, env_vars in providers:
            # Check if any required env var is set
            env_var_list = env_vars if isinstance(env_vars, list) else [env_vars]
            if any(os.environ.get(var) for var in env_var_list):
                model_list.append(
                    {
                        "model_name": model_name,
                        "litellm_params": {"model": litellm_model},
                    }
                )

        return model_list

    def validate(self) -> None:
        """Validate configuration logic."""
        if self.evaluators:
            evaluator = self.evaluators[0]
            config_path = evaluator.config.get("config_path")
            if config_path and not Path(config_path).exists():
                raise ValueError(f"Policy configuration file not found: {config_path}")

            # FSM engines don't require API keys
            if all(ev.type == "fsm" for ev in self.evaluators):
                return

        default_model = self.proxy.default_model

        if not default_model:
            raise ValueError(
                "No LLM API keys detected. Please set one of OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or GEMINI_API_KEY."
            )

        if "gpt" in default_model and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found (required for OpenAI models)")
        if "gemini" in default_model and not (
            self.google_api_key or self.gemini_api_key
        ):
            raise ValueError("GOOGLE_API_KEY not found (required for Gemini models)")
        if "claude" in default_model and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found (required for Anthropic models)"
            )
