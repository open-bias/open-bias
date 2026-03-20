# Configuration Reference

OpenSentinel reads configuration from three sources, applied in this order (highest priority wins):

1. `osentinel.yaml` (or `osentinel.yml`) in the working directory
2. API key environment variables (e.g., `OPENAI_API_KEY`)
3. Built-in defaults

API keys are always read from environment variables or `.env` files. Never put keys in YAML.

## Config File Discovery

OpenSentinel looks for the config file in this order:

1. Explicit path via `osentinel serve --config path/to/config.yaml`
2. `$OSNTL_CONFIG` environment variable
3. `./osentinel.yaml` in the current directory
4. `./osentinel.yml` in the current directory

If none are found, all settings use defaults.

## Minimal Config

```yaml
engine: judge
policy:
  - "No financial advice"
  - "Be professional"
```

This uses the judge engine with inline rules, auto-detected model, default port 4000.

## Global Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `engine` | string | `judge` | Policy engine type: `judge`, `fsm`, `llm`, `nemo` |
| `model` | string | auto-detected | Default LLM model. Auto-detected from whichever API key is present. Engines can override in their own section. |
| `fail_action` | string | `intervene` | What happens on policy violation: `intervene` (modify next request) or `block` (reject request) |
| `port` | int | `4000` | Proxy server port |
| `host` | string | `0.0.0.0` | Proxy server bind address |
| `debug` | bool | `false` | Enable debug logging |
| `log_level` | string | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Model auto-detection priority: `OPENAI_API_KEY` -> `gpt-4o-mini`, `GOOGLE_API_KEY`/`GEMINI_API_KEY` -> `gemini/gemini-2.5-flash`, `ANTHROPIC_API_KEY` -> `anthropic/claude-sonnet-4-5`.

## Policy

The `policy` key accepts three forms:

**File path** (string) -- passed as `config_path` to the engine:
```yaml
policy: ./customer_support.yaml
```

**Inline rules** (list) -- judge engine only:
```yaml
policy:
  - "No financial advice"
  - "Be professional"
```

**Inline rules with dicts** (list of dicts) -- judge engine only:
```yaml
policy:
  - rule: "No financial advice"
  - rule: "Be professional"
```

## Judge Engine

Set `engine: judge` at the top level. Configure under the `judge:` section.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `judge.model` | string | global `model` | LLM model for evaluation. Overrides the global model setting. |
| `judge.fail_action` | string | `intervene` | What happens on violation: `intervene` or `block`. Overrides the global `fail_action` for the judge engine. |
| `judge.pre_call_enabled` | bool | `false` | Evaluate requests before forwarding to the LLM |
| `judge.pre_call_rubric` | string | `safety` | Which rubric to use for pre-call evaluation |
| `judge.default_rubric` | string | `agent_behavior` | Default rubric for per-turn evaluation |
| `judge.conversation_rubric` | string | `conversation_policy` | Rubric for multi-turn conversation evaluation |
| `judge.custom_rubrics_path` | string | -- | Path to directory containing custom rubric YAML files |
| `judge.conversation_eval_interval` | int | `5` | Run conversation-level evaluation every N turns |

## LLM Engine

Set `engine: llm` at the top level. Requires `policy:` pointing to a workflow YAML file. Configure under the `llm:` section.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm.model` | string | global `model` | LLM model for state classification |
| `llm.temperature` | float | `0.0` | LLM temperature |
| `llm.max_tokens` | int | `1024` | Maximum tokens per LLM call |
| `llm.timeout` | float | `10.0` | Request timeout in seconds |
| `llm.confident_threshold` | float | `0.8` | Confidence above which a state classification is accepted |
| `llm.uncertain_threshold` | float | `0.5` | Below this, classification is rejected |
| `llm.temporal_weight` | float | `0.55` | Weight for temporal signals in drift detection |
| `llm.cooldown_turns` | int | `2` | Minimum turns between constraint re-evaluations |
| `llm.max_constraints_per_batch` | int | `5` | Maximum constraints evaluated per batch |

### LLM Engine Intervention Settings

```yaml
llm:
  intervention:
    default_strategy: user_message_inject   # system_prompt_append | user_message_inject | hard_block
    max_intervention_attempts: 3
    include_headers: true
```

## FSM Engine

Set `engine: fsm` at the top level. Requires `policy:` pointing to a workflow YAML file. Configure under the `fsm:` section.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fsm.classifier.model_name` | string | `all-MiniLM-L6-v2` | Sentence-transformers model for embedding-based state classification |
| `fsm.classifier.backend` | string | `pytorch` | Inference backend: `pytorch` or `onnx` |
| `fsm.classifier.similarity_threshold` | float | `0.7` | Minimum cosine similarity for a state match |
| `fsm.classifier.cache_embeddings` | bool | `true` | Cache computed embeddings for workflow states |
| `fsm.classifier.device` | string | `cpu` | Inference device: `cpu` or `cuda` |

### FSM Engine Intervention Settings

```yaml
fsm:
  intervention:
    default_strategy: system_prompt_append   # system_prompt_append | user_message_inject | hard_block
    max_intervention_attempts: 3
    include_headers: true
```

## NeMo Guardrails Engine

Set `engine: nemo` at the top level. Requires `policy:` pointing to a NeMo Guardrails config directory. Configure under the `nemo:` section.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `nemo.fail_closed` | bool | `false` | If true, block on NeMo evaluation errors. If false (default), warn and allow. |
| `nemo.rails` | list | all configured | Which rails to enable. Omit to use all rails from NeMo config. |

## Tracing

Configure under the `tracing:` section. Tracing uses OpenTelemetry spans.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tracing.type` | string | `otlp` | Exporter type: `otlp`, `langfuse`, `console`, `none` |
| `tracing.endpoint` | string | `http://localhost:4317` | OTLP endpoint URL |
| `tracing.service_name` | string | `opensentinel` | Service name in traces |

### Langfuse

When `tracing.type: langfuse`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tracing.langfuse_public_key` | string | -- | Langfuse public key |
| `tracing.langfuse_secret_key` | string | -- | Langfuse secret key |
| `tracing.langfuse_host` | string | `https://cloud.langfuse.com` | Langfuse host URL |

## Environment Variables

Environment variables are primarily used for LLM API keys. Generic settings should be configured via `osentinel.yaml`.

### Config File Discovery

You can override the configuration file path via the environment:

| Variable | Description |
|----------|-------------|
| `OSNTL_CONFIG` | Path to `osentinel.yaml` |

### API Keys

API keys bypass the `OSNTL_` prefix. Set them directly:

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` | Google (Gemini) |
| `GEMINI_API_KEY` | Google (Gemini, alternative) |
| `GROQ_API_KEY` | Groq |
| `TOGETHERAI_API_KEY` | Together AI |
| `OPENROUTER_API_KEY` | OpenRouter |

If multiple keys are present, the auto-detected model uses the first one found in the order above.

### Langfuse via Environment

While most settings are in YAML, Langfuse keys are also supported via environment variables for convenience:

| Variable | YAML Equivalent |
|----------|-----------------|
| `LANGFUSE_PUBLIC_KEY` | `tracing.langfuse_public_key` |
| `LANGFUSE_SECRET_KEY` | `tracing.langfuse_secret_key` |
| `LANGFUSE_HOST` | `tracing.langfuse_host` |

## .env File

OpenSentinel reads `.env` files automatically. API keys found in `.env` are synced to `os.environ` so downstream libraries (LiteLLM, etc.) can use them without explicit `load_dotenv()` calls.

See `.env.example` in the repository root for a template.

## Config Validation

The `osentinel serve` command validates configuration at startup:

- Checks that referenced policy files exist on disk
- Verifies that the required API key is present for the configured model
- Applies engine defaults before engine-specific overrides

If validation fails, the server prints the error and exits with code 1. Use `--debug` for a full traceback.

## Full Example

```yaml
engine: judge
model: gemini/gemini-2.5-flash
port: 4000
debug: false
fail_action: intervene

policy: ./policy.yaml

judge:
  model: anthropic/claude-sonnet-4-5
  pre_call_enabled: false

tracing:
  type: none
```
