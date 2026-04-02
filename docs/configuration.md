# Configuration Reference

OpenBias reads configuration from three sources, applied in this order (highest priority wins):

1. `openbias.yaml` (or `openbias.yml`) in the working directory
2. API key environment variables (e.g., `OPENAI_API_KEY`)
3. Built-in defaults

API keys are always read from environment variables or `.env` files. Never put keys in YAML.

## Config File Discovery

OpenBias looks for the config file in this order:

1. Explicit path via `openbias serve --config path/to/config.yaml`
2. `$OBIAS_CONFIG` environment variable
3. `./openbias.yaml` in the current directory
4. `./openbias.yml` in the current directory

If none are found, all settings use defaults.

## Minimal Config

```yaml
evaluators:
  - type: judge
    rules:
      - "No financial advice"
      - "Be professional"
```

This uses a single judge evaluator with inline rules, auto-detected model, default port 4000.

## Global Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | string | `async` | Evaluation mode: `sync` (blocking) or `async` (non-blocking, default) |
| `fail_action` | string | `intervene` | What happens on a rules violation: `intervene` (modify next request), `block` (reject request), or `shadow` (log only). **Note:** `block` is automatically normalized to `intervene` when `mode: async`, since async evaluation cannot block a response that has already been sent. |
| `strategy` | string | `user_message_inject` | Intervention strategy: `system_prompt_append` or `user_message_inject` |
| `session_ttl` | int | -- | Session time-to-live in seconds |
| `max_sessions` | int | -- | Maximum concurrent sessions |
| `model` | string | auto-detected | Default LLM model for the proxy target. Auto-detected from whichever API key is present. Evaluators can override with their own `model` key. |
| `port` | int | `4000` | Proxy server port |
| `host` | string | `0.0.0.0` | Proxy server bind address |
| `debug` | bool | `false` | Enable debug logging |
| `log_level` | string | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Model auto-detection priority: `OPENAI_API_KEY` -> `gpt-4o-mini`, `GOOGLE_API_KEY`/`GEMINI_API_KEY` -> `gemini/gemini-2.5-flash`, `ANTHROPIC_API_KEY` -> `anthropic/claude-sonnet-4-5`.

## Evaluator Pipeline

The `evaluators:` key defines an ordered list of evaluators that run against each request or response. Each evaluator is an independent rules check; all evaluators run regardless of whether earlier ones flag a violation.

### Standard Fields

Every evaluator entry supports these three fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `unnamed` | A unique label for this evaluator, used in logs and traces |
| `type` | string | `judge` | Engine type: `judge`, `fsm`, `llm`, or `nemo` |
| `phase` | string | `post_call` | When to run: `pre_call` (before the LLM responds) or `post_call` (after) |

### Canonical Rule Source

All evaluators compile from the project-local `rules.md` file. Keep rule text there rather than setting evaluator-specific `rules`, `rules_file`, `workflow`, or `config_path` keys in `openbias.yaml`.

Legacy user-facing keys like `policy`, `policies`, and `rubric` are no longer supported.

## Judge Engine

The judge engine evaluates responses against rules using a separate LLM as judge.

```yaml
evaluators:
  - name: content-policy
    type: judge
    phase: post_call
    model: anthropic/claude-sonnet-4-5
    # default_rules: agent_behavior
    # custom_rules_path: ./rules/
    # verbose: true
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | global `model` | LLM model for evaluation (shorthand for `config.models`). Overrides the global model setting. |
| `default_rules` | string | `agent_behavior` | Default rule set for per-turn evaluation. |
| `custom_rules_path` | string | -- | Path to directory containing custom rule set YAML files |
| `verbose` | bool | `false` | Log the raw judge prompt and response |

Pre-call evaluation is controlled by setting `phase: pre_call` on the evaluator entry. For conversation-scope evaluation, configure a separate evaluator with a conversation rule set.

## LLM Engine

The LLM engine uses a language model for state classification and constraint evaluation.

```yaml
evaluators:
  - name: llm-guard
    type: llm
    phase: post_call
    model: anthropic/claude-sonnet-4-5
    temperature: 0.0
    max_tokens: 1024
    # timeout: 10.0
    # confident_threshold: 0.8
    # uncertain_threshold: 0.5
    # temporal_weight: 0.55
    # cooldown_turns: 2
    # max_constraints_per_batch: 5
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | global `model` | LLM model for state classification (shorthand for `config.models`). |
| `temperature` | float | `0.0` | LLM temperature |
| `max_tokens` | int | `1024` | Maximum tokens per LLM call |
| `timeout` | float | `10.0` | Request timeout in seconds |
| `confident_threshold` | float | `0.8` | Confidence above which a state classification is accepted |
| `uncertain_threshold` | float | `0.5` | Below this, classification is rejected |
| `temporal_weight` | float | `0.55` | Weight for temporal signals in drift detection |
| `cooldown_turns` | int | `2` | Minimum turns between constraint re-evaluations |
| `max_constraints_per_batch` | int | `5` | Maximum constraints evaluated per batch |

## FSM Engine

The FSM engine enforces deterministic workflow rules using a finite state machine compiled from rules.

```yaml
evaluators:
  - name: workflow-guard
    type: fsm
    phase: post_call
    # classifier:
    #   model_name: all-MiniLM-L6-v2
    #   backend: pytorch
    #   similarity_threshold: 0.7
    #   cache_embeddings: true
    #   device: cpu
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `classifier.model_name` | string | `all-MiniLM-L6-v2` | Sentence-transformers model for embedding-based state classification |
| `classifier.backend` | string | `pytorch` | Inference backend: `pytorch` or `onnx` |
| `classifier.similarity_threshold` | float | `0.7` | Minimum cosine similarity for a state match |
| `classifier.cache_embeddings` | bool | `true` | Cache computed embeddings for workflow states |
| `classifier.device` | string | `cpu` | Inference device: `cpu` or `cuda` |

## NeMo Guardrails Engine

The NeMo engine integrates NVIDIA NeMo Guardrails for input/output rail enforcement.

```yaml
evaluators:
  - name: nemo-rails
    type: nemo
    phase: post_call
    # fail_closed: false
    # rails:
    #   - input
    #   - output
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fail_closed` | bool | `false` | If `true`, block on NeMo evaluation errors. If `false` (default), warn and allow. |
| `rails` | list | all configured | Which rails to enable. Omit to use all rails from the NeMo config. |

## Tracing

Configure under the `tracing:` section. Tracing uses OpenTelemetry spans.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tracing.type` | string | `none` | Exporter type: `otlp`, `langfuse`, `console`, `none`. Tracing is enabled when type is not `none`. |
| `tracing.endpoint` | string | `http://localhost:4317` | OTLP endpoint URL |
| `tracing.service_name` | string | `openbias` | Service name in traces |
| `tracing.redact_content` | bool | `false` | Strip prompts and completions from trace spans |

### Langfuse

When `tracing.type: langfuse`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tracing.langfuse_public_key` | string | -- | Langfuse public key |
| `tracing.langfuse_secret_key` | string | -- | Langfuse secret key |
| `tracing.langfuse_host` | string | `https://cloud.langfuse.com` | Langfuse host URL |

## Environment Variables

Environment variables are primarily used for LLM API keys. Generic settings should be configured via `openbias.yaml`.

### Config File Discovery

You can override the configuration file path via the environment:

| Variable | Description |
|----------|-------------|
| `OBIAS_CONFIG` | Path to `openbias.yaml` |

### API Keys

API keys bypass the `OBIAS_` prefix. Set them directly:

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

OpenBias reads `.env` files automatically. API keys found in `.env` are synced to `os.environ` so downstream libraries (LiteLLM, etc.) can use them without explicit `load_dotenv()` calls.

See `.env.example` in the repository root for a template.

## Eval

Configure offline scenario-based evaluation under the `eval:` section. Used by `openbias eval`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `eval.scenarios` | list of strings | `[]` | Glob patterns for scenario JSON files, resolved relative to the config file |

```yaml
eval:
  scenarios:
    - ./eval/scenarios/*.json
    - ./eval/scenarios/**/*.json    # recursive
```

The evaluators under test are determined by the top-level `evaluators` list.

## Config Validation

The `openbias serve` command validates configuration at startup:

- Checks that referenced rules files exist on disk
- Verifies that the required API key is present for the configured model (skipped for `fsm`, which is local-only)
- Applies defaults before evaluator-specific overrides
- Eagerly initializes all evaluators in the pipeline, failing fast on bad configuration instead of deferring errors to the first request

If validation fails, the server prints the error and exits with code 1. Use `--debug` for a full traceback.

## Full Example

```yaml
mode: async
fail_action: intervene
strategy: user_message_inject

model: gemini/gemini-2.5-flash
port: 4000
debug: false

evaluators:
  - name: safety-screen
    type: judge
    phase: pre_call
    model: anthropic/claude-sonnet-4-5

  - name: behavior-eval
    type: judge
    phase: post_call
    custom_rules_path: ./rules/

  - name: workflow-guard
    type: fsm
    phase: post_call
    rules:
      - "Verify identity before account changes"
      - "Never disclose internal pricing policy"

tracing:
  type: otlp
  endpoint: http://localhost:4317
  service_name: openbias
  redact_content: false
```
