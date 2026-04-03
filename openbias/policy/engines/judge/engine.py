"""
LLM-as-a-Judge Policy Engine.

Evaluates user messages or assistant responses against runtime-compiled rules
using LLM judges.
Integrates with the Open Bias policy engine infrastructure via PolicyEngine ABC.
"""

import asyncio
import logging
from typing import Any

from openbias.core.session import SessionStore
from openbias.policy.compiler.protocol import PolicyCompiler
from openbias.policy.protocols import (
    PolicyEngine,
    EvaluationResult,
    EvaluationStatus,
    ViolationRecord,
    require_initialized,
)
from openbias.policy.registry import register_engine
from openbias.policy.engines.judge.models import (
    AggregatedRuleResult,
    JudgeRuleResult,
    JudgeVerdict,
    JudgeSessionContext,
    VerdictAction,
)
from openbias.core.utils import extract_response_content, extract_tool_calls
from openbias.policy.engines.judge.client import JudgeClient
from openbias.policy.engines.judge.evaluator import JudgeEvaluator

logger = logging.getLogger(__name__)

@register_engine("judge")
class JudgePolicyEngine(PolicyEngine):
    """Policy engine that uses LLM judges to evaluate agent behavior.

    Supports turn-level evaluation against runtime-compiled rules.
    Works with single or multiple judge models.
    """

    # Defaults for session memory management
    DEFAULT_SESSION_TTL = 3600  # 1 hour
    DEFAULT_MAX_SESSIONS = 10_000
    VALID_AGGREGATION_MODES = frozenset({"majority", "all", "any"})

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
        self._aggregation_mode: str = "majority"

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

        compiled_rules = config.get("_compiled_rules")
        if not isinstance(compiled_rules, list) or not all(
            isinstance(rule, str) and rule.strip() for rule in compiled_rules
        ):
            raise ValueError(
                "Judge engine `_compiled_rules` must be a non-empty list of strings."
            )
        self._rules = [rule.strip() for rule in compiled_rules if rule.strip()]
        self._rules_source = str(config.get("_rules_source") or "compiled_rules")
        self._aggregation_mode = str(config.get("aggregation_mode") or "majority")
        if self._aggregation_mode not in self.VALID_AGGREGATION_MODES:
            raise ValueError(
                "Judge engine `aggregation_mode` must be one of: "
                f"{', '.join(sorted(self.VALID_AGGREGATION_MODES))}."
            )
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

        Runs the evaluator's compiled rules against the latest user message.
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

        session = self._get_or_create_session(session_id)
        if not self._client or not self._client.model_names:
            return EvaluationResult(status=EvaluationStatus.ALLOW)

        try:
            verdict = await self._evaluate_turn(
                content_to_evaluate=user_message,
                conversation=messages,
                metadata=(context or {}).get("metadata", {}),
                session_id=session_id,
                session_context=session,
                target_role="user",
            )
            parent_span = (context or {}).get("_parent_span")
            self._trace_verdict(session_id, verdict, parent_span=parent_span)
            return self._build_result(verdict, session)
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

        Runs the evaluator's compiled rules against the latest response.
        """
        session = self._get_or_create_session(session_id)
        response_content = extract_response_content(response_data)
        tool_calls = extract_tool_calls(response_data)
        tool_definitions = self._extract_tool_definitions(request_data)
        conversation = self._extract_conversation(request_data)
        metadata = (context or {}).get("metadata", {})
        parent_span = (context or {}).get("_parent_span")

        if not self._client or not self._client.model_names:
            logger.error("No judge models configured")
            return EvaluationResult(status=EvaluationStatus.ALLOW)

        try:
            verdict = await self._evaluate_turn(
                content_to_evaluate=response_content,
                conversation=conversation,
                metadata=metadata,
                session_id=session_id,
                tool_calls=tool_calls,
                session_context=session,
                tool_definitions=tool_definitions,
                target_role="assistant",
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
        result = self._build_result(verdict, session)

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
        from openbias.policy.engines.judge.compiler import JudgeRuntimeCompiler

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

        compiled_rules = config.get("_compiled_rules")
        if (
            not isinstance(compiled_rules, list)
            or not compiled_rules
            or not all(isinstance(rule, str) and rule.strip() for rule in compiled_rules)
        ):
            errors.append(
                "Judge engine `_compiled_rules` must be a non-empty list of strings."
            )

        aggregation_mode = config.get("aggregation_mode")
        if aggregation_mode is not None and aggregation_mode not in cls.VALID_AGGREGATION_MODES:
            errors.append(
                "Judge engine `aggregation_mode` must be one of: "
                f"{', '.join(sorted(cls.VALID_AGGREGATION_MODES))}."
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
                rules_source=self._rules_source,
                scope=verdict.scope.value,
                action=verdict.action.value,
                participating_judges=self._collect_participating_judges(
                    verdict.aggregated_results
                ),
                failed_rules=verdict.failed_rules,
                rule_results=[
                    {
                        "rule": result.rule,
                        "passed": result.passed,
                        "action": result.action.value,
                        "summary": result.summary,
                        "aggregation_mode": result.aggregation_mode,
                        "judge_results": [
                            judge_result.to_dict() for judge_result in result.judge_results
                        ],
                    }
                    for result in verdict.aggregated_results
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
        verdict: JudgeVerdict,
        session: JudgeSessionContext,
    ) -> EvaluationResult:
        """Build EvaluationResult from judge verdicts.

        Emits one normalized ViolationRecord per failed aggregated rule.
        """
        violation_records: list[ViolationRecord] = []
        for rule_result in verdict.aggregated_results:
            if rule_result.passed:
                continue
            violation_records.append(ViolationRecord(
                reason=self._build_violation_message(rule_result),
                severity=rule_result.action.value,
                scope=verdict.scope.value,
                engine=self.name,
                confidence=self._rule_confidence(rule_result),
                extra={
                    "rule": rule_result.rule,
                    "aggregation_mode": rule_result.aggregation_mode,
                    "judge_results": [
                        judge_result.to_dict()
                        for judge_result in rule_result.judge_results
                    ],
                    "summary": rule_result.summary,
                    **rule_result.metadata,
                },
            ))

        status = EvaluationStatus.VIOLATION if violation_records else EvaluationStatus.ALLOW

        metadata: dict[str, Any] = {
            "judge": {
                "verdict": {
                    **{
                        key: value
                        for key, value in verdict.to_dict().items()
                        if key != "judge_models"
                    },
                    "rules_source": self._rules_source,
                    "participating_judges": self._collect_participating_judges(
                        verdict.aggregated_results
                    ),
                },
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

    async def _evaluate_turn(
        self,
        content_to_evaluate: str,
        conversation: list[dict[str, Any]],
        target_role: str,
        metadata: dict[str, Any] | None,
        session_id: str,
        tool_calls: list[dict[str, Any]] | None = None,
        session_context: JudgeSessionContext | None = None,
        tool_definitions: dict[str, dict[str, Any]] | None = None,
        fail_action: VerdictAction = VerdictAction.INTERVENE,
    ) -> JudgeVerdict:
        aggregated_results: list[AggregatedRuleResult] = []
        for rule in self._rules:
            aggregated_results.append(
                await self._evaluate_rule_with_judges(
                    rule=rule,
                    content_to_evaluate=content_to_evaluate,
                    conversation=conversation,
                    target_role=target_role,
                    metadata=metadata,
                    session_id=session_id,
                    tool_calls=tool_calls,
                    session_context=session_context,
                    tool_definitions=tool_definitions,
                    fail_action=fail_action,
                )
            )

        failed_rules = [result.rule for result in aggregated_results if not result.passed]
        summary = (
            "All compiled rules passed."
            if not failed_rules
            else "Failed compiled rules: " + ", ".join(failed_rules)
        )
        return JudgeVerdict(
            aggregated_results=aggregated_results,
            failed_rules=failed_rules,
            action=VerdictAction.PASS if not failed_rules else fail_action,
            summary=summary,
            judge_models=self._collect_judge_models(aggregated_results),
            latency_ms=sum(
                max((result.latency_ms for result in aggregated.judge_results), default=0.0)
                for aggregated in aggregated_results
            ),
            token_usage=sum(
                result.token_usage
                for aggregated in aggregated_results
                for result in aggregated.judge_results
            ),
            metadata={"aggregation_mode": self._aggregation_mode},
        )

    async def _evaluate_rule_with_judges(
        self,
        rule: str,
        content_to_evaluate: str,
        conversation: list[dict[str, Any]],
        target_role: str,
        metadata: dict[str, Any] | None,
        session_id: str,
        tool_calls: list[dict[str, Any]] | None = None,
        session_context: JudgeSessionContext | None = None,
        tool_definitions: dict[str, dict[str, Any]] | None = None,
        fail_action: VerdictAction = VerdictAction.INTERVENE,
    ) -> AggregatedRuleResult:
        model_names = self._client.model_names if self._client else []
        evaluations = await asyncio.gather(
            *[
                self._evaluator.evaluate_rule(
                    model_name=model_name,
                    rule=rule,
                    content_to_evaluate=content_to_evaluate,
                    conversation=conversation,
                    target_role=target_role,
                    metadata=metadata,
                    session_id=session_id,
                    tool_calls=tool_calls,
                    session_context=session_context,
                    tool_definitions=tool_definitions,
                )
                for model_name in model_names
            ],
            return_exceptions=True,
        )

        judge_results: list[JudgeRuleResult] = []
        judge_errors: list[str] = []
        for model_name, evaluation in zip(model_names, evaluations):
            if isinstance(evaluation, Exception):
                logger.error("Judge model '%s' failed for rule '%s': %s", model_name, rule, evaluation)
                judge_errors.append(f"{model_name}: {evaluation}")
                continue
            judge_results.append(evaluation)

        if not judge_results:
            return AggregatedRuleResult(
                rule=rule,
                passed=False,
                action=fail_action,
                judge_results=[],
                summary=f"All judge models failed while evaluating rule: {rule}",
                aggregation_mode=self._aggregation_mode,
                metadata={
                    "passed_count": 0,
                    "failed_count": 0,
                    "participating_judges": [],
                    "failing_judges": [],
                    "evaluation_error": "all_judges_failed",
                    "judge_errors": judge_errors,
                },
            )

        passed = self._aggregate_rule_pass(judge_results)
        passed_count = sum(1 for result in judge_results if result.passed)
        failed_count = len(judge_results) - passed_count
        return AggregatedRuleResult(
            rule=rule,
            passed=passed,
            action=VerdictAction.PASS if passed else fail_action,
            judge_results=judge_results,
            summary=self._summarize_rule_result(rule, judge_results, passed),
            aggregation_mode=self._aggregation_mode,
            metadata={
                "passed_count": passed_count,
                "failed_count": failed_count,
                "participating_judges": [result.judge_name for result in judge_results],
                "failing_judges": [
                    result.judge_name for result in judge_results if not result.passed
                ],
            },
        )

    def _aggregate_rule_pass(self, judge_results: list[JudgeRuleResult]) -> bool:
        passed_count = sum(1 for result in judge_results if result.passed)
        failed_count = len(judge_results) - passed_count
        if self._aggregation_mode == "all":
            return failed_count == 0
        if self._aggregation_mode == "any":
            return passed_count > 0
        return passed_count > failed_count

    def _collect_participating_judges(
        self, aggregated_results: list[AggregatedRuleResult]
    ) -> list[str]:
        seen: list[str] = []
        for aggregated in aggregated_results:
            for result in aggregated.judge_results:
                if result.judge_name and result.judge_name not in seen:
                    seen.append(result.judge_name)
        return seen

    def _collect_judge_models(
        self, aggregated_results: list[AggregatedRuleResult]
    ) -> list[str]:
        seen: list[str] = []
        for aggregated in aggregated_results:
            for result in aggregated.judge_results:
                if result.judge_model and result.judge_model not in seen:
                    seen.append(result.judge_model)
        return seen

    def _rule_confidence(self, rule_result: AggregatedRuleResult) -> float:
        if not rule_result.judge_results:
            return 0.0
        return sum(result.confidence for result in rule_result.judge_results) / len(
            rule_result.judge_results
        )

    def _summarize_rule_result(
        self,
        rule: str,
        judge_results: list[JudgeRuleResult],
        passed: bool,
    ) -> str:
        if passed:
            return f"Rule passed after {len(judge_results)} judge evaluation(s): {rule}"
        failing_judges = ", ".join(
            result.judge_name for result in judge_results if not result.passed
        )
        if failing_judges:
            return f"Rule failed: {rule} (failing judges: {failing_judges})"
        return f"Rule failed: {rule}"

    def _build_violation_message(self, rule_result: AggregatedRuleResult) -> str:
        """Build a diagnostic reason string from a failed aggregated rule.

        The reason field is diagnostic: it explains *what* went wrong and
        *why*.  User-facing prose and enforcement decisions belong in the
        interceptor layer.

        Prioritizes corrective_actions from failing judges. Falls back to the
        aggregated rule summary when no per-judge guidance is available.
        """
        failing_results = [result for result in rule_result.judge_results if not result.passed]
        if not failing_results:
            return rule_result.summary or f"Rule violation detected: {rule_result.rule}"
        paragraphs: list[str] = []
        for result in failing_results:
            if result.corrective_actions:
                parts: list[str] = [result.corrective_actions]
                if result.evidence:
                    quotes = ", ".join(f'"{e}"' for e in result.evidence)
                    parts.append(f"(Evidence: {quotes})")
                paragraphs.append(" ".join(parts))
            else:
                parts = []
                if result.reasoning:
                    parts.append(result.reasoning.rstrip(".") + ".")
                if result.evidence:
                    quotes = ", ".join(f'"{e}"' for e in result.evidence)
                    parts.append(f"(Evidence: {quotes})")
                paragraphs.append(
                    " ".join(parts) if parts else f"Rule not met: {rule_result.rule}."
                )

        return "\n\n".join(paragraphs) if paragraphs else (
            rule_result.summary or f"Rule violation detected: {rule_result.rule}"
        )

