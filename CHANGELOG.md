# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.4.0

### Breaking
- Replaced top-level `engine`, `policy`, `judge`, `fsm`, `llm`, `nemo` config keys with `evaluators` list — each evaluator entry specifies `name`, `type`, and `phase`
- Removed `PolicyEngineConfig`, `PolicyConfig`, and `_map_to_settings()`
- Interceptor constructor now takes `pre_call_evaluators` and `post_call_evaluators` instead of a single `engines` list
- `rules_file` and inline `rules` keys removed from evaluator config — `RULES.md` is now the sole policy input
- `ViolationRecord` no longer includes `rule_id`, `rule_name`, or `evidence` fields
- `rubric_name` replaced by `rules_source` in judge evaluator config

### Removed
- Removed `max_intervention_attempts` and intervention escalation — interventions remain interventions regardless of frequency
- `ResponseModificationStrategy` removed — use sync intervention instead

### Added
- Multi-evaluator pipeline with phase-based execution (pre_call / post_call)
- YAML shorthand mappings for evaluator config (`model`) plus automatic compilation from project-local `RULES.md`
- Top-level `mode`, `strategy`, `session_ttl`, `max_sessions` settings
- `openbias trigger` — evaluate a single message against policy with structured output
- `openbias replay` — replay a JSONL trace dataset against a policy offline
- `openbias improve` — generate and replay policy variants, recommend one for review
- Judge engine: per-rule binary evaluation, `target_role` support, multi-judge verdict aggregation
- Sync post-call enforcement (block/intervene inline without async defer)
- Zero-config `openbias init` with synthesized defaults, RULES.md gate, and preset rules library
- `log_format` setting (`text`/`json`) for structured logging
- JSONL trace sink for replayable policy traces (`tracing.sink: jsonl`)
- `async+block` config automatically normalized to `intervene`

### Changed
- `checker` renamed to `evaluator` in all metadata keys

## 0.3.0

### Breaking
- Renamed package from `opensentinel` to `openbias`
- Removed composite engine
- Removed reliability modes, ensemble, and multi-threshold from judge engine
- Rewrote FSM schema and compiler (deterministic pipeline, simplified constraints)
- Removed `mode` and `pass_threshold` config options

### Added
- Eval framework: `openbias eval` with per-engine scenario runners, metrics, and reporting
- Shadow `fail_action` mode — log violations without blocking
- Trace content redaction
- Session TTL, LRU eviction, and per-session async task caps
- Eager evaluator engine initialization at startup
- Tool call awareness in judge evaluations
- Session ID validation to prevent log/OTEL injection

### Fixed
- Dozens of correctness fixes across all engines (judge scoring, FSM transitions, LLM drift detection, NeMo rail activation)
- Interceptor race conditions, session lifecycle, and async result handling

## 0.2.0

### Added

- **`openbias init -q`**: Quick (non-interactive) init — auto-detects your API key and writes a minimal config in one shot.
- **CLI output formatting**: Rich-formatted console output for all CLI commands (headings, YAML previews, success/error indicators).
- **Model auto-detection**: Automatically resolves the best LLM model from whichever API key is present (`OPENAI_API_KEY` → `gpt-4o-mini`, `GEMINI_API_KEY` → `gemini/gemini-2.5-flash`, etc.).
- **Model & API-key validation**: `openbias serve` and `openbias init` now validate that the required API key exists for the configured model before starting.
- **YAML as single source of truth**: `openbias.yaml` is now the primary configuration surface. Removed the `OBIAS_*` environment-variable prefix; API keys are still read from env vars / `.env`.
- **Path resolution**: Project-local policy files and other relative paths are resolved relative to the config file location.
- **Config validation at startup**: `openbias serve` checks that referenced rules files exist and that the required API key is present; exits with a clear error if not.
- **API-key syncing**: Keys loaded from `.env` are synced into `os.environ` so downstream libraries (LiteLLM, LangChain) work without explicit `load_dotenv()`.
- **Langfuse config**: Langfuse keys are configured via `openbias.yaml` under `tracing:`.
- **Automatic rules compilation**: `openbias serve`, `trigger`, and `eval` automatically compile rules into engine-native config at startup. No separate compile step needed.
- **`docs/configuration.md`**: Full configuration reference for `openbias.yaml`.

### Changed

- **Default engine** changed from `nemo` to `judge` — works out of the box with project-local `RULES.md`; no external config directory required.
- **Tracing disabled by default**: `tracing.enabled` now defaults to `false` and `exporter_type` defaults to `none` to avoid noisy OTLP connection errors on first run.
- **`proxy.default_model`** defaults to `None` instead of eagerly auto-detecting; the model is resolved at startup via YAML or auto-detection.
- **Intervention merge logic** refactored for consistency across FSM and LLM engines.
- **Rules compiler** refactored into per-engine modules (`fsm`, `llm`, `judge`, `nemo`).
- **Docs updated**: `developing.md`, `examples/README.md`, and `README.md` updated to reflect YAML-first configuration.

### Fixed

- Judge engine: score clamping, criterion failure checks, JSON validation, timezone-aware timestamps.
- Session ID propagation for internal LLM calls in the Judge engine.
- `intervention` and `classifier` YAML sections now correctly map to `Settings`.

## 0.1.0 (alpha)

Initial release.

### Added

- Transparent LLM proxy built on LiteLLM. Point any OpenAI-compatible client at it with a one-line `base_url` change.
- **Judge engine**: scores responses against plain-English rules using a sidecar LLM. Binary pass/fail evaluation with configurable fail action. Async by default -- zero latency on the critical path.
- **FSM engine**: enforces agent behavior as a finite state machine. Three-tier classification cascade (tool call matching, regex, embedding similarity). LTL-lite temporal constraint evaluation.
- **LLM engine**: classifies conversation state and detects drift using LLM-based reasoning.
- **NeMo engine**: integrates NVIDIA NeMo Guardrails for content safety and dialog rails.
- **Composite engine**: runs multiple engines in parallel, merges results (most restrictive wins).
- **Rules compiler**: translates natural language rules to engine-specific config automatically during serve startup.
- **CLI**: `openbias init`, `openbias serve`, `openbias trigger`, `openbias eval`, `openbias validate`, `openbias info`.
- **OpenTelemetry tracing**: spans for every proxy call, rules evaluation, and intervention. Console, OTLP, and Langfuse backends.
- Fail-open design: hook failures pass the request through unmodified. Only intentional blocks propagate.
- Deferred intervention: violations detected async are applied as prompt injections on the next turn.
- Session ID extraction from headers, query params, or request body.

### Known Limitations

- Session state is in-memory only. Not persistent across restarts.
- No dashboard UI.
- No pre-built rules library. *(Resolved in 0.4.0)*
- No rate limiting.
