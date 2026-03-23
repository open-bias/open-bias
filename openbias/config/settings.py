"""
Open Bias configuration management using Pydantic Settings.

Configuration can be provided via:
1. openbias.yaml config file (primary — see config/schema.yaml for full reference)
2. Environment variables (ONLY for API keys like OPENAI_API_KEY, GEMINI_API_KEY, etc.)
3. .env file
4. Direct instantiation

Priority (highest wins): openbias.yaml > API keys > defaults

The simplified openbias.yaml format:
    engine: judge
    model: gemini/gemini-2.5-flash   # optional — auto-detected from API keys
    port: 4000
    judge:
      pass_threshold: 0.6
    policy:
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
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
)

logger = logging.getLogger(__name__)




class OTelConfig(BaseModel):
    """OpenTelemetry tracing configuration.

    Supports multiple exporters:
    - otlp: Standard OTLP endpoint (Jaeger, Zipkin, etc.)
    - langfuse: Langfuse's OTLP endpoint (requires public_key/secret_key)
    - console: Print traces to console (for debugging)
    - none: Disable tracing
    """

    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = False
    endpoint: str = "http://localhost:4317"
    service_name: str = "openbias"
    exporter_type: Literal["otlp", "langfuse", "console", "none"] = "none"
    insecure: bool = True  # Use insecure connection (no TLS) for local dev

    # Langfuse-specific settings (used when exporter_type="langfuse")
    langfuse_public_key: str | None = Field(None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        "https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )

    # When True, user/assistant message content is replaced with "[REDACTED]" in traces
    redact_content: bool = False


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


class PolicyEngineConfig(BaseModel):
    """Configuration for a single policy engine.

    The 'type' field accepts any engine registered via @register_engine
    in the PolicyEngineRegistry (see openbias/policy/registry.py).

    Built-in engine types include: fsm, nemo, llm, judge.
    Custom engines are automatically supported once registered.

    The 'config' field contains engine-specific configuration.
    """

    # Accepts any registered engine type — not hard-coded to a fixed set.
    # See PolicyEngineRegistry.list_engines() for available types at runtime.
    type: str = "judge"
    enabled: bool = True
    # Unified configuration path
    config_path: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, **kwargs):
        """Custom dump to merge config_path into config dict for engines."""
        data = super().model_dump(**kwargs)
        # If data.get("config_path") is set, use it to populate/override config dict
        if data.get("config_path"):
            data["config"]["config_path"] = data["config_path"]
        return data


class PolicyConfig(BaseModel):
    """Policy system configuration.

    Supports multiple policy engines with different mechanisms:
    - FSM for workflow enforcement
    - NeMo Guardrails for content moderation
    - Judge for LLM-based evaluation

    Examples:
        # Judge (default)
        policy:
          engine:
            type: judge

        # FSM only
        policy:
          engine:
            type: fsm
            config:
              workflow_path: ./workflow.yaml

        # NeMo only
        policy:
          engine:
            type: nemo
            config:
              config_path: ./nemo_config/
    """

    # Primary engine configuration
    engine: PolicyEngineConfig = Field(default_factory=PolicyEngineConfig)

    # Default strategy when not specified in workflow
    default_strategy: Literal[
        "system_prompt_append", "user_message_inject"
    ] = "user_message_inject"

    # What happens when a policy violation is detected.
    # "intervene": modify the next request to steer the agent (default)
    # "block": reject the request outright
    # "shadow": log findings but take no action
    fail_action: Literal["intervene", "block", "shadow"] = "intervene"

    # Fallback behavior when engine evaluation fails
    # True = allow request on error (fail open)
    # False = deny request on error (fail closed)
    fail_open: bool = True

    # Maximum time (seconds) any hook is allowed to run before fail-open timeout.
    # Set generously since interceptor checks may involve LLM calls.
    hook_timeout_seconds: float = 30.0

    # POST_CALL checker execution mode.
    # "async" (default): checkers run in background, results deferred to next request.
    #   Zero latency impact but cannot enforce on the current response.
    # "sync": checkers block the response, enabling BLOCK/INTERVENE on current response.
    #   Adds latency (depends on engine) but enables real-time enforcement.
    post_call_mode: Literal["sync", "async"] = "async"


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
            "engine",
            "policy",
            "port",
            "host",
            "debug",
            "log_level",
            "model",
            "tracing",
            "eval",
            "post_call_mode",
            "fail_action",
            "fail_open",
            # Engine-specific sections are handled below
            "judge",
            "llm",
            "fsm",
            "nemo",
        }
    )

    # Keys within judge: that receive special handling
    _JUDGE_SPECIAL_KEYS = frozenset({"model"})

    # Keys within llm: that need renaming for the engine config
    _LLM_KEY_RENAMES = {"model": "llm_model"}

    def _map_to_settings(self) -> dict[str, Any]:
        """Map simplified YAML keys to nested Settings structure.

        See openbias/config/schema.yaml for the full reference of
        supported keys and their mapping behavior.
        """
        if not self._yaml_data:
            return {}

        data = self._yaml_data
        result: dict[str, Any] = {}

        # engine -> policy.engine.type
        if "engine" in data:
            result.setdefault("policy", {}).setdefault("engine", {})["type"] = data[
                "engine"
            ]

        # model -> proxy.default_model
        if "model" in data:
            result.setdefault("proxy", {})["default_model"] = data["model"]

        # policy -> config_path (string) or inline_policy (list/dict)
        if "policy" in data:
            policy_val = data["policy"]
            if isinstance(policy_val, str):
                resolved_path = self._resolve_path(policy_val)
                result.setdefault("policy", {}).setdefault("engine", {})[
                    "config_path"
                ] = resolved_path
            elif isinstance(policy_val, (list, dict)):
                engine_config = (
                    result.setdefault("policy", {})
                    .setdefault("engine", {})
                    .setdefault("config", {})
                )
                engine_config["inline_policy"] = policy_val

        # post_call_mode -> policy.post_call_mode
        if "post_call_mode" in data:
            result.setdefault("policy", {})["post_call_mode"] = data[
                "post_call_mode"
            ]

        # fail_action -> policy.fail_action
        if "fail_action" in data:
            result.setdefault("policy", {})["fail_action"] = data["fail_action"]

        # fail_open -> policy.fail_open
        if "fail_open" in data:
            result.setdefault("policy", {})["fail_open"] = data["fail_open"]

        # port -> proxy.port
        if "port" in data:
            result.setdefault("proxy", {})["port"] = data["port"]

        # host -> proxy.host
        if "host" in data:
            result.setdefault("proxy", {})["host"] = data["host"]

        # debug
        if "debug" in data:
            result["debug"] = data["debug"]

        # log_level
        if "log_level" in data:
            result["log_level"] = data["log_level"]

        # -----------------------------------------------------------------
        # Engine-specific sections -> policy.engine.config.*
        # Each engine's YAML section passes all keys into the config dict
        # that the engine's initialize() receives.
        # -----------------------------------------------------------------
        engine_type = data.get("engine", "judge")

        # judge.* -> policy.engine.config.*
        judge_cfg = data.get("judge", {})
        if engine_type == "judge" and isinstance(judge_cfg, dict) and judge_cfg:
            engine_config = (
                result.setdefault("policy", {})
                .setdefault("engine", {})
                .setdefault("config", {})
            )

            # Special: judge.model -> models list entry
            if "model" in judge_cfg:
                engine_config["models"] = [
                    {"name": "primary", "model": judge_cfg["model"]}
                ]

            # Pass through all remaining keys directly
            for k, v in judge_cfg.items():
                if k not in self._JUDGE_SPECIAL_KEYS:
                    engine_config[k] = v

        # llm.* -> policy.engine.config.*
        llm_cfg = data.get("llm", {})
        if engine_type == "llm" and isinstance(llm_cfg, dict) and llm_cfg:
            engine_config = (
                result.setdefault("policy", {})
                .setdefault("engine", {})
                .setdefault("config", {})
            )
            for k, v in llm_cfg.items():
                # Rename 'model' -> 'llm_model' to match what LLMPolicyEngine expects
                mapped_key = self._LLM_KEY_RENAMES.get(k, k)
                engine_config[mapped_key] = v

        # nemo.* -> policy.engine.config.*
        nemo_cfg = data.get("nemo", {})
        if engine_type == "nemo" and isinstance(nemo_cfg, dict) and nemo_cfg:
            engine_config = (
                result.setdefault("policy", {})
                .setdefault("engine", {})
                .setdefault("config", {})
            )
            for k, v in nemo_cfg.items():
                if k == "config_path":
                    v = self._resolve_path(v)
                engine_config[k] = v

        # fsm.* -> policy.engine.config.*
        fsm_cfg = data.get("fsm", {})
        if engine_type == "fsm" and isinstance(fsm_cfg, dict) and fsm_cfg:
            engine_config = (
                result.setdefault("policy", {})
                .setdefault("engine", {})
                .setdefault("config", {})
            )
            for k, v in fsm_cfg.items():
                if k == "workflow_path" or k == "config_path":
                    v = self._resolve_path(v)
                engine_config[k] = v

        # -----------------------------------------------------------------
        # Tracing / observability
        # -----------------------------------------------------------------

        # tracing.* -> otel.*
        tracing_cfg = data.get("tracing", {})
        if isinstance(tracing_cfg, dict) and tracing_cfg:
            otel = result.setdefault("otel", {})
            if "type" in tracing_cfg:
                tracing_type = tracing_cfg["type"]
                otel["exporter_type"] = tracing_type
                if tracing_type == "none":
                    otel["enabled"] = False
                else:
                    otel["enabled"] = True
            if "endpoint" in tracing_cfg:
                otel["endpoint"] = tracing_cfg["endpoint"]
            if "service_name" in tracing_cfg:
                otel["service_name"] = tracing_cfg["service_name"]
            if "insecure" in tracing_cfg:
                otel["insecure"] = tracing_cfg["insecure"]
            # Langfuse-specific settings
            if "langfuse_public_key" in tracing_cfg:
                otel["langfuse_public_key"] = tracing_cfg["langfuse_public_key"]
            if "langfuse_secret_key" in tracing_cfg:
                otel["langfuse_secret_key"] = tracing_cfg["langfuse_secret_key"]
            if "langfuse_host" in tracing_cfg:
                otel["langfuse_host"] = tracing_cfg["langfuse_host"]
            if "redact_content" in tracing_cfg:
                otel["redact_content"] = tracing_cfg["redact_content"]

        return result

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        mapped = self._map_to_settings()
        value = mapped.get(field_name)
        return value, field_name, value is not None

    def __call__(self) -> dict[str, Any]:
        return self._map_to_settings()


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
    litellm_verbose: bool = False

    # Component configurations
    otel: OTelConfig = Field(default_factory=OTelConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)

    # Policy engine configuration
    policy: PolicyConfig = Field(default_factory=PolicyConfig)

    # API Keys (loaded from env vars or .env file)
    # We use validation_alias to map standard keys to these fields
    openai_api_key: str | None = Field(None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(None, validation_alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(None, validation_alias="GOOGLE_API_KEY")
    gemini_api_key: str | None = Field(None, validation_alias="GEMINI_API_KEY")
    groq_api_key: str | None = Field(None, validation_alias="GROQ_API_KEY")
    togetherai_api_key: str | None = Field(None, validation_alias="TOGETHERAI_API_KEY")
    openrouter_api_key: str | None = Field(None, validation_alias="OPENROUTER_API_KEY")

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

        Priority mirrors detect_available_model() in cli_init.py:
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

    def get_policy_config(self) -> dict[str, Any]:
        """
        Get policy engine configuration.

        Returns:
            Configuration dict ready for PolicyEngineRegistry.create_and_initialize()
        """
        engine_config = self.policy.engine.model_dump()

        # For judge engines: ensure models list is populated from global default_model
        # if no explicit judge.model was configured. Engine should never resolve models itself.
        if (
            engine_config.get("type") == "judge"
            and not engine_config["config"].get("models")
            and self.proxy.default_model
        ):
            engine_config["config"]["models"] = [
                {"name": "primary", "model": self.proxy.default_model}
            ]

        return engine_config

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
        # 1. Check if policy config path exists
        # Check both the unified 'config_path' field and within the 'config' dictionary
        policy_config_path = self.policy.engine.config_path or self.policy.engine.config.get("config_path")
        
        if policy_config_path and not Path(policy_config_path).exists():
            raise ValueError(
                f"Policy configuration file not found: {policy_config_path}"
            )

        # FSM engine is purely local — no LLM model or API keys required.
        if self.policy.engine.type == "fsm":
            return

        default_model = self.proxy.default_model

        if not default_model:
            # If no model is set (and auto-detection failed because no keys were found),
            # we must enforce that the user provides at least one key or sets a model manually.
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
