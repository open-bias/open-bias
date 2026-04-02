"""
LLM-as-a-Judge Policy Engine.

Evaluates agent responses and conversation trajectories against
configurable rules using LLM judges. Integrates with the Open Bias
policy engine infrastructure via PolicyEngine ABC.
"""

import logging
from pathlib import Path
from typing import Any

from openbias.core.session import SessionStore
from openbias.policy.compiler.protocol import CompilationResult, PolicyCompiler
from openbias.policy.protocols import (
    PolicyEngine,
    EvaluationResult,
    EvaluationStatus,
    ViolationRecord,
    require_initialized,
)
from openbias.policy.registry import register_engine
from openbias.policy.engines.judge.models import (
    JudgeVerdict,
    JudgeSessionContext,
    VerdictAction,
)
from openbias.core.utils import extract_response_content, extract_tool_calls
from openbias.policy.engines.judge.client import JudgeClient
from openbias.policy.engines.judge.evaluator import JudgeEvaluator

logger = logging.getLogger(__name__)


class JudgeRuntimeCompiler(PolicyCompiler):
    """Runtime compiler used by serve-time rule normalization."""

    @property
    def engine_type(self) -> str:
        return "judge"

    async def compile(
        self,
        rules_text: str,
        context: dict[str, Any] | None = None,
    ) -> CompilationResult:
        del context
        compiled_rules = [line.strip() for line in rules_text.splitlines() if line.strip()]
        if not compiled_rules:
            return CompilationResult.failure(
                errors=["Judge runtime compilation produced no rules."],
            )
        return CompilationResult(
            success=True,
            config={
                "_compiled_rules": compiled_rules,
                "_rules_source": "rules.md",
            },
        )

    def export(self, result: CompilationResult, output_path: Path) -> None:
        del result, output_path
        return None

@register_engine("judge")
class JudgePolicyEngine(PolicyEngine):
    """Policy engine that uses LLM judges to evaluate agent behavior.

    Supports turn-level evaluation against configurable rules.
    Works with single or multiple judge models.
    """

    # Defaults for session memory management
    DEFAULT_SESSION_TTL = 3600  # 1 hour
    DEFAULT_MAX_SESSIONS = 10_000

    def __init__(self) -> None:
        self._initialized = False
        self._client: JudgeClient | None = None
        self._evaluator: JudgeEvaluator | None = None
        self._sessions: SessionStore[JudgeSessionContext] = SessionStore(
            ttl=self.DEFAULT_SESSION_TTL,
            max_sessions=self.DEFAULT_MAX_SESSIONS,
        )
        self._tracer: Any | None = None
        self._rules: list[str] = []
        self._rules_source: str = "compiled_rules"

    @property
    def name(self) -> str:
        return "judge"

    @property
    def engine_type(self) -> str:
        return "judge"

    @property
    def timeout(self) -> float:
        """Worst-case wall-clock time for a judge evaluation, including retries.

        Used by Callback to ensure the hook timeout never races
        the judge's own LLM timeout.  Returns 0.0 when no models are
        configured (e.g. before initialize() is called).
        """
        return self._client.timeout if self._client is not None else 0.0

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the judge engine with configuration.

        Args:
            config: Configuration dict with:
                - models: List of judge model configs [{name, model, temperature, ...}]
                - _compiled_rules: Precompiled canonical rules list (required)
                - session_ttl: Session TTL in seconds
                - max_sessions: Maximum concurrent sessions
        """
        # Build client with judge models
        self._client = JudgeClient()
        models = config.get("models", [])

        if not models:
            raise ValueError(
                "Judge engine requires a model. "
                "Set 'model' in openbias.yaml or configure judge.model."
            )

        for model_config in models:
            self._client.add_model(
                name=model_config.get("name", "primary"),
                model=model_config["model"],
                temperature=model_config.get("temperature", 0.0),
                max_tokens=model_config.get("max_tokens", 2048),
                timeout=model_config.get("timeout", 15.0),
            )

        # Build evaluator
        self._evaluator = JudgeEvaluator(
            client=self._client,
            verbose=False,
        )

        # Session memory management
        self._sessions.configure(
            ttl=int(config["session_ttl"]) if "session_ttl" in config else None,
            max_sessions=int(config["max_sessions"]) if "max_sessions" in config else None,
        )

        if "rules_file" in config:
            raise ValueError(
                "Judge engine no longer accepts `rules_file`; use runtime-compiled "
                "`_compiled_rules` from project rules.md."
            )
        compiled_rules = config.get("_compiled_rules")
        if not isinstance(compiled_rules, list) or not all(
            isinstance(rule, str) and rule.strip() for rule in compiled_rules
        ):
            raise ValueError(
                "Judge engine `_compiled_rules` must be a non-empty list of strings."
            )
        self._rules = [rule.strip() for rule in compiled_rules if rule.strip()]
        self._rules_source = str(config.get("_rules_source") or "compiled_rules")
        if not self._rules:
            raise ValueError("No rules found for judge evaluator.")

        self._initialized = True
        logger.info("JudgePolicyEngine initialized with %d rules", len(self._rules))

    def set_tracer(self, tracer: Any) -> None:
        """Set an optional Tracer for OTEL tracing."""
        self._tracer = tracer

    @require_initialized
    async def evaluate_request(
        self,
        session_id: str,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate an incoming request (PRE_CALL).

        Runs the configured default ruleset against the latest user message.
        The interceptor only calls this method when the evaluator is assigned
        to the pre_call phase, so no phase guard is needed here.
        """
        messages = request_data.get("messages", [])
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return EvaluationResult(status=EvaluationStatus.ALLOW)

        primary_model = self._client.primary_model
        if not primary_model:
            return EvaluationResult(status=EvaluationStatus.ALLOW)

        try:
            verdict = await self._evaluator.evaluate_turn(
                model_name=primary_model,
                rules=self._rules,
                response_content=user_message,
                conversation=messages,
                metadata=(context or {}).get("metadata", {}),
                session_id=session_id,
                fail_action=VerdictAction.INTERVENE,
            )
            parent_span = (context or {}).get("_parent_span")
            self._trace_verdict(session_id, verdict, parent_span=parent_span)
            return self._build_result([verdict], self._get_or_create_session(session_id))
        except Exception as e:
            logger.error(f"Pre-call evaluation failed: {e}")
            return EvaluationResult(status=EvaluationStatus.ALLOW)

    @require_initialized
    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate an LLM response (POST_CALL).

        Runs the configured default ruleset against the latest response.
        Each evaluator instance has one ruleset; conversation-scope evaluation
        is handled by configuring a separate evaluator instance.
        """
        session = self._get_or_create_session(session_id)
        response_content = extract_response_content(response_data)
        tool_calls = extract_tool_calls(response_data)
        tool_definitions = self._extract_tool_definitions(request_data)
        conversation = self._extract_conversation(request_data)
        metadata = (context or {}).get("metadata", {})
        parent_span = (context or {}).get("_parent_span")

        primary_model = self._client.primary_model
        if not primary_model:
            logger.error("No judge models configured")
            return EvaluationResult(status=EvaluationStatus.ALLOW)

        try:
            verdict = await self._evaluator.evaluate_turn(
                model_name=primary_model,
                rules=self._rules,
                response_content=response_content,
                conversation=conversation,
                metadata=metadata,
                session_id=session_id,
                tool_calls=tool_calls,
                session_context=session,
                tool_definitions=tool_definitions,
                fail_action=VerdictAction.INTERVENE,
            )
            self._trace_verdict(session_id, verdict, parent_span=parent_span)
        except Exception as e:
            logger.error(
                f"Turn evaluation failed for session {session_id} "
                f"({type(e).__name__}): {e}"
            )
            return EvaluationResult(status=EvaluationStatus.ALLOW)

        # Build result before recording verdicts so escalation checks
        # compare against prior session state, not the current turn's own data
        result = self._build_result([verdict], session)

        # Record verdicts to session after result is built
        session.turn_count += 1
        session.record_verdict(verdict)

        return result

    async def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        """Get current session state for debugging/tracing."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        return session.to_dict()

    async def reset_session(self, session_id: str) -> None:
        """Reset session state."""
        self._sessions.remove(session_id)

    def get_compiler(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> PolicyCompiler | None:
        """Return the judge runtime compiler used by serve-time compilation."""
        del model, api_key, base_url
        return JudgeRuntimeCompiler()

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> list[str]:
        """Validate judge engine configuration without needing an LLM connection.

        Runs the same checks as initialize() but collects errors into a list
        instead of raising on the first one.

        Returns:
            List of error strings. Empty list means config is valid.
        """
        errors: list[str] = []

        # Check models
        models = config.get("models", [])
        if not models:
            errors.append(
                "No model configured. Set 'model' in openbias.yaml or configure judge.model."
            )
        else:
            for i, m in enumerate(models):
                if not isinstance(m, dict) or not m.get("model"):
                    errors.append(f"models[{i}]: missing 'model' field.")

        if "rules_file" in config:
            errors.append(
                "Judge engine no longer accepts `rules_file`; use runtime-compiled "
                "`_compiled_rules` from project rules.md."
            )
        compiled_rules = config.get("_compiled_rules")
        if not isinstance(compiled_rules, list) or not all(
            isinstance(rule, str) and rule.strip() for rule in compiled_rules
        ):
            errors.append(
                "Judge engine `_compiled_rules` must be a non-empty list of strings."
            )

        return errors

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._sessions.clear()
        self._initialized = False
        logger.info("JudgePolicyEngine shut down")

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _trace_verdict(
        self,
        session_id: str,
        verdict: JudgeVerdict,
        parent_span: Any | None = None,
    ) -> None:
        """Log a verdict to the OTEL tracer if available."""
        if not self._tracer:
            return
        try:
            self._tracer.log_judge_evaluation(
                session_id=session_id,
                rubric_name=self._rules_source,
                scope=verdict.scope.value,
                composite_score=verdict.composite_score,
                action=verdict.action.value,
                judge_model=verdict.judge_model,
                scores=[
                    {
                        "criterion": s.criterion,
                        "score": s.score,
                        "max_score": s.max_score,
                        "normalized": s.normalized,
                        "confidence": s.confidence,
                        "reasoning": s.reasoning,
                        "corrective_actions": s.corrective_actions,
                    }
                    for s in verdict.scores
                ],
                latency_ms=verdict.latency_ms,
                token_usage=verdict.token_usage,
                metadata=verdict.metadata,
                parent_span=parent_span,
                evaluator_name=self.name,
            )
        except Exception as e:
            logger.debug(f"Failed to trace verdict: {e}")

    def _get_or_create_session(self, session_id: str) -> JudgeSessionContext:
        session = self._sessions.get(session_id)
        if session is None:
            session = JudgeSessionContext(session_id=session_id)
            self._sessions.put(session_id, session)
        else:
            self._sessions.touch(session_id)
        return session

    def _build_result(
        self,
        verdicts: list[JudgeVerdict],
        session: JudgeSessionContext,
    ) -> EvaluationResult:
        """Build EvaluationResult from judge verdicts.

        Takes the most restrictive action across all verdicts.
        Engines are pure evaluators: PASS → ALLOW, anything else → VIOLATION.
        """
        action_priority = {
            VerdictAction.PASS: 0,
            VerdictAction.INTERVENE: 1,
            VerdictAction.BLOCK: 2,
        }

        worst_verdict = max(verdicts, key=lambda v: action_priority.get(v.action, 0))

        # Build violation records for non-PASS verdicts
        violation_records: list[ViolationRecord] = []
        for v in verdicts:
            if v.action == VerdictAction.PASS:
                continue
            violation_records.append(ViolationRecord(
                reason=self._build_violation_message(v),
                severity=v.action.value,
                scope=v.scope.value,
                engine=self.name,
                confidence=v.composite_score,
                extra={
                    "composite_score": v.composite_score,
                    "judge_model": v.judge_model,
                    "summary": v.summary,
                },
            ))

        status = EvaluationStatus.VIOLATION if violation_records else EvaluationStatus.ALLOW

        metadata: dict[str, Any] = {
            "judge": {
                "verdicts": [
                    {**v.to_dict(), "rubric_name": self._rules_source}
                    for v in verdicts
                ],
                "session_turn": session.turn_count + 1,
            },
        }

        return EvaluationResult(
            status=status,
            violations=violation_records,
            metadata=metadata,
        )

    def _extract_conversation(self, request_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract conversation messages from request data."""
        return request_data.get("messages", [])

    def _extract_tool_definitions(
        self, request_data: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Extract tool schemas from request data (OpenAI API format).

        Returns a dict mapping tool name to its definition summary:
        {
            "delete_user": {
                "description": "Permanently removes a user account",
                "parameters": {"id": "integer — the user ID to delete"},
            }
        }
        """
        tools = request_data.get("tools", [])
        definitions: dict[str, dict[str, Any]] = {}

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function", {})
            name = func.get("name")
            if not name:
                continue

            definition: dict[str, Any] = {}
            if func.get("description"):
                definition["description"] = func["description"]

            params = func.get("parameters", {})
            if isinstance(params, dict) and "properties" in params:
                param_summaries: dict[str, str] = {}
                for pname, pdef in params["properties"].items():
                    ptype = pdef.get("type", "any")
                    pdesc = pdef.get("description", "")
                    param_summaries[pname] = (
                        f"{ptype} — {pdesc}" if pdesc else ptype
                    )
                definition["parameters"] = param_summaries

            definitions[name] = definition

        return definitions

    def _build_violation_message(self, verdict: JudgeVerdict) -> str:
        """Build a diagnostic reason string from failed criteria.

        The reason field is diagnostic: it explains *what* went wrong and
        *why*.  User-facing prose and enforcement decisions belong in the
        interceptor layer.

        Prioritizes corrective_actions (judge-generated guidance) over raw
        criterion data.  Falls back to the verdict summary when no
        per-criterion failure data is available.
        """
        failed_criteria: list[str] = verdict.metadata.get("criterion_failures", [])
        if not failed_criteria:
            return verdict.summary or "Rule violation detected."

        score_map = {s.criterion: s for s in verdict.scores}

        paragraphs: list[str] = []
        for criterion_name in failed_criteria:
            score = score_map.get(criterion_name)
            if not score:
                paragraphs.append(f"Criterion not met: {criterion_name}.")
                continue

            if score.corrective_actions:
                parts: list[str] = [score.corrective_actions]
                if score.evidence:
                    quotes = ", ".join(f'"{e}"' for e in score.evidence)
                    parts.append(f"(Evidence: {quotes})")
                paragraphs.append(" ".join(parts))
            else:
                parts = []
                if score.reasoning:
                    parts.append(score.reasoning.rstrip(".") + ".")
                if score.evidence:
                    quotes = ", ".join(f'"{e}"' for e in score.evidence)
                    parts.append(f"(Evidence: {quotes})")
                paragraphs.append(" ".join(parts) if parts else
                                  f"Criterion not met: {criterion_name}.")

        return "\n\n".join(paragraphs)

