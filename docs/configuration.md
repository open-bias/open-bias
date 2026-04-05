# Configuration Reference

Open Bias can enforce rules with no `openbias.yaml` at all. If you have a project-local `rules.md`, the CLI can synthesize a working default evaluator and start the proxy immediately.

Use `openbias.yaml` when you want explicit project settings such as pinned models, multiple evaluators, tracing, replay boundary selection, or committed offline eval/improvement workflows.

## Zero-Config Defaults

When you run `openbias serve`, `openbias trigger`, `openbias validate`, or `openbias info` without a config file, Open Bias resolves:

- a synthesized `judge` evaluator
- `mode: sync`
- `fail_action: intervene`
- `strategy: user_message_inject`
- proxy port `4000`
- tracing disabled
- project-local `rules.md` as the canonical authored policy source

You still need one provider API key in the environment and a local `rules.md` file.

## Effective Settings Sources

Open Bias resolves runtime behavior from these sources:

1. CLI flags for the active command (for example `openbias serve --port 8080`)
2. `openbias.yaml` (or `openbias.yml`) when present
3. API key environment variables and `.env` files for provider credentials
4. Built-in defaults / synthesized evaluator settings

API keys are always read from environment variables or `.env` files. Never put keys in YAML.

## Config File Discovery

OpenBias looks for the config file in this order:

1. Explicit path via `openbias serve --config path/to/config.yaml`
2. `$OBIAS_CONFIG` environment variable
3. `./openbias.yaml` in the current directory
4. `./openbias.yml` in the current directory

If none are found, all settings use defaults.

## Minimal Config

This file is optional. If you do want to commit project settings, the smallest useful config is:

```yaml
evaluators:
  - type: judge
```

```md
- No financial advice.
- Be professional.
```

This uses a single judge evaluator, project-local `rules.md`, runtime-compiled rules, an auto-detected model, and the default port 4000.

## Command Expectations

These CLI commands can run from synthesized defaults plus local `rules.md`:

- `openbias serve`
- `openbias trigger`
- `openbias validate`
- `openbias info`

These commands currently require `openbias.yaml` because they depend on committed offline project settings:

- `openbias eval`
- `openbias replay`
- `openbias improve`

## Global Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | string | `async` | Evaluation mode: `sync` (blocking) or `async` (non-blocking, default) |
| `fail_action` | string | `intervene` | What happens after a rule violation is detected: `intervene` (queue a corrective intervention for the next turn), `block` (reject request), or `shadow` (log only). **Note:** `block` is automatically normalized to `intervene` when `mode: async`, since async evaluation cannot block a response that has already been sent. |
| `strategy` | string | `user_message_inject` | Request-time intervention strategy. Supported values are `system_prompt_append` and `user_message_inject`; `user_message_inject` is the default. |
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

The judge engine evaluates runtime-compiled rules using one or more separate LLM judges.

```yaml
evaluators:
  - name: content-policy
    type: judge
    phase: post_call
    # aggregation_mode: majority
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `models` | list[object] | injected from global default model when omitted | Explicit judge model list for multi-judge evaluation. Each entry follows `{name, model, temperature?, max_tokens?, timeout?}`. |
| `aggregation_mode` | string | `majority` | How to aggregate binary results when multiple judges evaluate the same compiled rule. Supported values: `majority`, `all`, `any`. |

Pre-call evaluation is controlled by setting `phase: pre_call` on the evaluator entry. Policy text still comes from project `rules.md`; Open Bias compiles that file into `_compiled_rules` internally, then judges each compiled rule independently with a binary pass/fail result. If you need to pin specific judge models instead of using the global default model, set `models` explicitly on the evaluator.

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
| `rails` | list | all configured | Which rails to enable. Omit to use every rail from the compiled NeMo runtime config generated from project `rules.md`. |

## Tracing

Configure under the `tracing:` section. Tracing uses OpenTelemetry spans.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tracing.type` | string | `none` | Exporter type: `otlp`, `langfuse`, `console`, `jsonl`, `none`. Tracing is enabled when type is not `none`. |
| `tracing.endpoint` | string | `http://localhost:4317` | OTLP endpoint URL |
| `tracing.path` | string | -- | When `tracing.type: jsonl`, append replayable trace cases to this path. Supports file paths such as `.openbias/traces/%Y-%m-%d.jsonl` or directory paths such as `.openbias/traces/`. |

### Langfuse

When `tracing.type: langfuse`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tracing.langfuse_public_key` | string | -- | Langfuse public key |
| `tracing.langfuse_secret_key` | string | -- | Langfuse secret key |
| `tracing.langfuse_host` | string | `https://cloud.langfuse.com` | Langfuse host URL |

### JSONL Trace Sink

When `tracing.type: jsonl`, Open Bias writes one replayable request/response pair per line to a local JSONL dataset. This is intended for offline replay and policy improvement workflows.

```yaml
tracing:
  type: jsonl
  path: .openbias/traces/%Y-%m-%d.jsonl
```

`path` may be either:

- a file path ending in `.jsonl`
- a directory path, in which case Open Bias writes `YYYY-MM-DD.jsonl` inside it

These datasets are the handoff point into the continuous-improvement loop:

- `openbias replay --trace ...` re-runs the current policy against captured traffic
- `openbias improve --trace ... --instruction ...` generates variants, replays them, and writes reviewer-facing artifacts

Replay and improvement share the same `replay.boundary` setting. They evaluate one boundary only: `response` by default, or `request` when you want request-side detection.

Variant generation happens inside `openbias improve`, but human review still decides whether any generated policy should replace `rules.md`.

## Environment Variables

Environment variables are only supported for config discovery and secrets. Generic runtime settings such as `debug`, `log_level`, `port`, and `litellm_verbose` should be configured in `openbias.yaml` or passed explicitly via CLI flags / `Settings(...)`.

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

### Langfuse Keys

Langfuse credentials can also be set via environment variables (or `.env`):

| Variable | YAML Equivalent |
|----------|-----------------|
| `OBIAS_OTEL__LANGFUSE_PUBLIC_KEY` | `tracing.langfuse_public_key` |
| `OBIAS_OTEL__LANGFUSE_SECRET_KEY` | `tracing.langfuse_secret_key` |
| `OBIAS_OTEL__LANGFUSE_HOST` | `tracing.langfuse_host` |

## .env File

OpenBias reads `.env` files automatically. API keys found in `.env` are synced to `os.environ` so downstream libraries (LiteLLM, etc.) can use them without explicit `load_dotenv()` calls.

See `.env.example` in the repository root for a template.

## Eval

Configure offline native-suite evaluation under the `eval:` section. `openbias eval` uses these paths when you do not pass `--suite` explicitly.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `eval.suites` | list of strings | `[]` | Native suite files, directories, or glob patterns, resolved relative to the config file |

```yaml
eval:
  suites:
    - ./evals/suites
    - ./evals/smoke/*.yaml
```

If `eval.suites` is omitted, `openbias eval` falls back to `evals/suites/` in the current project.

The evaluators under test are determined by the top-level `evaluators` list.

## Replay

Configure offline detection replay under the `replay:` section. `openbias replay` and `openbias improve` both use this boundary selector.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `replay.boundary` | string | `response` | Which trace boundary to evaluate offline: `request` or `response` |

```yaml
replay:
  boundary: response
```

Offline replay and improvement currently build a single engine from the first entry in `evaluators` for simplicity. If you configure multiple evaluators, reorder the list so the one you want exercised offline is listed first.

## Config Validation

The `openbias serve` command validates configuration at startup, whether settings came from YAML or synthesized defaults:

- Checks that required project policy files such as `rules.md` are present when compilation is needed
- Verifies that the required API key is present for the configured model (skipped for `fsm`, which is local-only)
- Applies defaults before evaluator-specific overrides
- Eagerly initializes all evaluators in the pipeline, failing fast on bad configuration instead of deferring errors to the first request

If validation fails, the server prints the error and exits with code 1. Use `--debug` for a full traceback.

## Full Example

Open Bias now supports only these two request-time strategies:

- `user_message_inject` (default): insert guidance as a synthetic user note on the request that carries the intervention.
- `system_prompt_append`: append guidance to the system prompt on that request using a `<system-reminder>` wrapper. This strategy does not register response-cleanup markers.

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

  - name: workflow-guard
    type: fsm
    phase: post_call

tracing:
  type: otlp
  endpoint: http://localhost:4317
```
