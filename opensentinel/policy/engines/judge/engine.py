"""
LLM-as-a-Judge Policy Engine.

Evaluates agent responses and conversation trajectories against
configurable rubrics using LLM judges. Integrates with the Open Sentinel
policy engine infrastructure via PolicyEngine ABC.
"""

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from opensentinel.policy.compiler.protocol import PolicyCompiler

from opensentinel.core.session import SessionStore
from opensentinel.policy.protocols import (
    PolicyEngine,
    Decision,
    EngineResult,
    require_initialized,
)
from opensentinel.policy.registry import register_engine
from opensentinel.policy.engines.judge.models import (
    JudgeVerdict,
    JudgeSessionContext,
    VerdictAction,
    EvaluationScope,
)
from opensentinel.core.utils import extract_response_content, extract_tool_calls
from opensentinel.policy.engines.judge.client import JudgeClient
from opensentinel.policy.engines.judge.evaluator import JudgeEvaluator
from opensentinel.policy.engines.judge.rubrics import (
    RubricRegistry,
    create_rules_rubric,
    _parse_rubric_dict,
)

logger = logging.getLogger(__name__)

# Mapping from VerdictAction to Decision
_VERDICT_MAP: dict[VerdictAction, Decision] = {
    VerdictAction.PASS: Decision.ALLOW,
    VerdictAction.INTERVENE: Decision.INTERVENE,
    VerdictAction.BLOCK: Decision.BLOCK,
}


@register_engine("judge")
class JudgePolicyEngine(PolicyEngine):
    """Policy engine that uses LLM judges to evaluate agent behavior.

    Supports turn-level and conversation-level evaluation against
    configurable rubrics. Works with single or multiple judge models.
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
        self._registry = RubricRegistry()

        # Config
        self._default_rubric: str = "agent_behavior"
        self._conversation_rubric: str | None = "conversation_policy"
        self._pre_call_enabled: bool = False
        self._pre_call_rubric: str = "safety"
        self._conversation_eval_interval: int = 5

    @property
    def name(self) -> str:
        return f"judge:{self._default_rubric}"

    @property
    def engine_type(self) -> str:
        return "judge"

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the judge engine with configuration.

        Args:
            config: Configuration dict with:
                - models: List of judge model configs [{name, model, temperature, ...}]
                - default_rubric: Name of default turn-scope rubric
                - conversation_rubric: Name of conversation-scope rubric (or null to disable)
                - pre_call_enabled: Whether to evaluate requests (default: false)
                - pre_call_rubric: Rubric for pre-call evaluation
                - conversation_eval_interval: Run conversation eval every N turns (default: 5)
                - custom_rubrics_path: Path to custom rubric YAML files
                - checker_mode: "async" or "sync" (used by interceptor, not engine)
        """
        # Build client with judge models
        self._client = JudgeClient()
        models = config.get("models", [])

        if not models:
            raise ValueError(
                "Judge engine requires a model. "
                "Set 'model' in osentinel.yaml or configure judge.model."
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
            verbose=config.get("verbose", False),
        )

        # Config
        self._default_rubric = config.get("default_rubric", "agent_behavior")
        self._conversation_rubric = config.get("conversation_rubric", "conversation_policy")
        self._pre_call_enabled = config.get("pre_call_enabled", False)
        self._pre_call_rubric = config.get("pre_call_rubric", "safety")
        self._conversation_eval_interval = config.get("conversation_eval_interval", 5)

        # Session memory management
        self._sessions.configure(
            ttl=int(config["session_ttl"]) if "session_ttl" in config else None,
            max_sessions=int(config["max_sessions"]) if "max_sessions" in config else None,
        )

        # Load custom rubrics if configured
        custom_path = config.get("custom_rubrics_path")
        if custom_path:
            self._registry.load_from_yaml(custom_path)

        # Load inline policy if provided
        inline_policy = config.get("inline_policy")
        if inline_policy is not None:
            self._load_inline_policy(inline_policy)

        # Fail-loud: verify default rubric exists
        if not self._registry.get(self._default_rubric):
            available = self._registry.list_rubrics()
            raise ValueError(
                f"Default rubric '{self._default_rubric}' not found. "
                f"Available: {available}"
            )

        # Verify conversation rubric exists if configured
        if self._conversation_rubric and not self._registry.get(self._conversation_rubric):
            available = self._registry.list_rubrics()
            raise ValueError(
                f"Conversation rubric '{self._conversation_rubric}' not found. "
                f"Available: {available}"
            )

        self._initialized = True
        logger.info(f"JudgePolicyEngine initialized: {self.name}")

    def set_tracer(self, tracer: Any) -> None:
        """Set an optional SentinelTracer for OTEL tracing."""
        self._tracer = tracer

    @require_initialized
    async def evaluate_request(
        self,
        session_id: str,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        """Evaluate an incoming request (PRE_CALL).

        Default: ALLOW (most judgment happens post-call).
        If pre_call_enabled, run safety screening.
        """
        # Optional pre-call screening
        if self._pre_call_enabled:
            return await self._evaluate_pre_call(session_id, request_data, context)

        return EngineResult(decision=Decision.ALLOW)

    @require_initialized
    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        """Evaluate an LLM response (POST_CALL).

        Main evaluation path:
        1. Always run turn-scope rubric on latest response
        2. Run conversation-scope rubric on interval or when triggered
        3. Merge verdicts (most restrictive action wins)
        4. Map to EngineResult
        """
        session = self._get_or_create_session(session_id)
        response_content = extract_response_content(response_data)
        tool_calls = extract_tool_calls(response_data)
        tool_definitions = self._extract_tool_definitions(request_data)
        conversation = self._extract_conversation(request_data)
        metadata = (context or {}).get("metadata", {})

        primary_model = self._client.primary_model
        if not primary_model:
            logger.error("No judge models configured")
            return EngineResult(decision=Decision.ALLOW)

        verdicts: list[JudgeVerdict] = []

        # 1. Turn-scope evaluation (always runs)
        turn_rubric = self._registry.get(self._default_rubric)
        if turn_rubric:
            try:
                turn_verdict = await self._evaluator.evaluate_turn(
                    model_name=primary_model,
                    rubric=turn_rubric,
                    response_content=response_content,
                    conversation=conversation,
                    metadata=metadata,
                    session_id=session_id,
                    tool_calls=tool_calls,
                    session_context=session,
                    tool_definitions=tool_definitions,
                    fail_action=VerdictAction.INTERVENE,
                )
                self._trace_verdict(session_id, turn_verdict, turn_rubric.name)
                verdicts.append(turn_verdict)
            except Exception as e:
                logger.error(
                    f"Turn evaluation failed for session {session_id} "
                    f"({type(e).__name__}): {e}"
                )
        else:
            logger.warning(f"Default rubric not found: {self._default_rubric}")

        # 2. Conversation-scope evaluation (on interval or trigger)
        if self._should_run_conversation_eval(session, verdicts):
            conv_rubric = self._registry.get(self._conversation_rubric)
            if conv_rubric:
                try:
                    conv_verdict = await self._evaluator.evaluate_conversation(
                        model_name=primary_model,
                        rubric=conv_rubric,
                        full_conversation=conversation,
                        metadata=metadata,
                        session_id=session_id,
                        session_context=session,
                        fail_action=VerdictAction.INTERVENE,
                    )
                    self._trace_verdict(session_id, conv_verdict, conv_rubric.name)
                    verdicts.append(conv_verdict)
                except Exception as e:
                    logger.error(
                        f"Conversation evaluation failed for session {session_id} "
                        f"({type(e).__name__}): {e}"
                    )

        # 3. Merge verdicts and build result
        if not verdicts:
            return EngineResult(decision=Decision.ALLOW)

        # Build result before recording verdicts so escalation checks
        # compare against prior session state, not the current turn's own data
        result = self._build_result(verdicts, session)

        # 4. Record verdicts to session after result is built
        session.turn_count += 1
        session.last_intervention_criteria = []
        for v in verdicts:
            session.record_verdict(v)
        if result.decision == Decision.INTERVENE:
            session.intervention_count += 1

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
    ) -> "PolicyCompiler | None":
        """Return a JudgeCompiler instance."""
        from opensentinel.policy.engines.judge.compiler import JudgeCompiler
        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return JudgeCompiler(**kwargs)

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
                "No model configured. Set 'model' in osentinel.yaml or configure judge.model."
            )
        else:
            for i, m in enumerate(models):
                if not isinstance(m, dict) or not m.get("model"):
                    errors.append(f"models[{i}]: missing 'model' field.")

        # Build a temporary registry and load inline policy to check rubrics
        registry = RubricRegistry()

        custom_path = config.get("custom_rubrics_path")
        if custom_path:
            from pathlib import Path as _Path

            p = _Path(custom_path)
            if not p.exists():
                errors.append(f"custom_rubrics_path '{custom_path}' does not exist.")
            else:
                registry.load_from_yaml(custom_path)

        inline_policy = config.get("inline_policy")
        if inline_policy is not None:
            try:
                # Validate inline policy by attempting to parse it
                temp_engine = cls()
                temp_engine._registry = registry
                temp_engine._load_inline_policy(inline_policy)
                registry = temp_engine._registry
            except (ValueError, TypeError) as e:
                errors.append(f"Invalid inline policy: {e}")

        # Check default rubric exists
        default_rubric = config.get("default_rubric", "agent_behavior")
        if not registry.get(default_rubric):
            available = registry.list_rubrics()
            errors.append(
                f"Default rubric '{default_rubric}' not found. Available: {available}"
            )

        # Check conversation rubric if set
        conv_rubric = config.get("conversation_rubric", "conversation_policy")
        if conv_rubric and not registry.get(conv_rubric):
            available = registry.list_rubrics()
            errors.append(
                f"Conversation rubric '{conv_rubric}' not found. Available: {available}"
            )

        # Check pre_call_rubric if pre_call is enabled
        if config.get("pre_call_enabled", False):
            pre_rubric = config.get("pre_call_rubric", "safety")
            if not registry.get(pre_rubric):
                available = registry.list_rubrics()
                errors.append(
                    f"Pre-call rubric '{pre_rubric}' not found. Available: {available}"
                )

        return errors

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._sessions.clear()
        logger.info("JudgePolicyEngine shut down")

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _load_inline_policy(self, policy_data: Any) -> None:
        """Load inline policy definitions from config.

        Supported shapes:
        - str: multiline string split into rules
        - list[str]: plain-text rules → auto-generated binary rubric
        - list[dict]: formal rubric definitions

        Raises ValueError for dict input or unrecognized formats.
        """
        if isinstance(policy_data, str):
            # Multiline string → split into rules
            rules = [line.strip() for line in policy_data.strip().splitlines() if line.strip()]
            if rules:
                rubric = create_rules_rubric(rules)
                self._registry.register(rubric)
                self._default_rubric = rubric.name
                logger.info(f"Loaded {len(rules)} inline policy rules as '{rubric.name}'")
            return

        if isinstance(policy_data, list):
            if not policy_data:
                return
            # List of strings → plain-text rules
            if all(isinstance(item, str) for item in policy_data):
                rubric = create_rules_rubric(policy_data)
                self._registry.register(rubric)
                self._default_rubric = rubric.name
                logger.info(f"Loaded {len(policy_data)} inline policy rules as '{rubric.name}'")
                return
            # List of dicts → formal rubric definitions
            for idx, item in enumerate(policy_data):
                if isinstance(item, dict):
                    try:
                        rubric = _parse_rubric_dict(item)
                        self._registry.register(rubric)
                        # First rubric becomes default
                        if idx == 0:
                            self._default_rubric = rubric.name
                        logger.info(f"Loaded inline rubric '{rubric.name}'")
                    except Exception as e:
                        logger.error(f"Failed to parse inline rubric: {e}")
            return

        if isinstance(policy_data, dict):
            raise ValueError(
                "Dict-format inline policy is no longer supported. "
                "Use a list of rule strings or a list of rubric dicts instead."
            )

        raise ValueError(f"Unrecognized inline_policy format: {type(policy_data)}")

    def _trace_verdict(
        self,
        session_id: str,
        verdict: JudgeVerdict,
        rubric_name: str,
    ) -> None:
        """Log a verdict to the OTEL tracer if available."""
        if not self._tracer:
            return
        try:
            self._tracer.log_judge_evaluation(
                session_id=session_id,
                rubric_name=rubric_name,
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

    def _should_run_conversation_eval(
        self,
        session: JudgeSessionContext,
        turn_verdicts: list[JudgeVerdict],
    ) -> bool:
        """Determine if conversation-scope evaluation should run."""
        if not self._conversation_rubric:
            return False

        # Run on interval (check +1 because record_verdict hasn't incremented yet)
        effective_turn = session.turn_count + 1
        if (
            effective_turn > 0
            and effective_turn % self._conversation_eval_interval == 0
        ):
            return True

        # Run when a turn verdict is intervene or worse
        for v in turn_verdicts:
            if v.action != VerdictAction.PASS:
                return True

        return False

    async def _evaluate_pre_call(
        self,
        session_id: str,
        request_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EngineResult:
        """Run pre-call safety screening on user message."""
        rubric = self._registry.get(self._pre_call_rubric)
        if not rubric:
            return EngineResult(decision=Decision.ALLOW)

        messages = request_data.get("messages", [])
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return EngineResult(decision=Decision.ALLOW)

        primary_model = self._client.primary_model
        if not primary_model:
            return EngineResult(decision=Decision.ALLOW)

        try:
            verdict = await self._evaluator.evaluate_turn(
                model_name=primary_model,
                rubric=rubric,
                response_content=user_message,
                conversation=messages,
                metadata=(context or {}).get("metadata", {}),
                session_id=session_id,
                fail_action=VerdictAction.INTERVENE,
            )
            return self._build_result([verdict], self._get_or_create_session(session_id))
        except Exception as e:
            logger.error(f"Pre-call evaluation failed: {e}")
            return EngineResult(decision=Decision.ALLOW)

    def _build_result(
        self,
        verdicts: list[JudgeVerdict],
        session: JudgeSessionContext,
    ) -> EngineResult:
        """Build EngineResult from judge verdicts.

        Takes the most restrictive action across all verdicts.
        Applies escalation when repeat violations are detected:
        if the escalation cap is reached and the current decision is
        INTERVENE, it is upgraded to BLOCK.
        """
        action_priority = {
            VerdictAction.PASS: 0,
            VerdictAction.INTERVENE: 1,
            VerdictAction.BLOCK: 2,
        }

        worst_verdict = max(verdicts, key=lambda v: action_priority.get(v.action, 0))
        decision = _VERDICT_MAP[worst_verdict.action]

        # Check escalation across all non-PASS verdicts, not just the worst.
        # A turn verdict with repeat criterion violations must trigger escalation
        # even when a conversation verdict is the worst overall.
        escalation_info: dict[str, Any] = {
            "should_escalate": False,
            "reason": "",
            "escalation_prefix": "",
        }
        for v in verdicts:
            if v.action == VerdictAction.PASS:
                continue
            info = self._check_escalation(v, session)
            if info["should_escalate"]:
                escalation_info = info
                break

        was_escalated = False
        if escalation_info["should_escalate"] and decision == Decision.INTERVENE:
            decision = Decision.BLOCK
            was_escalated = True

        # message = guidance for INTERVENE, reason for BLOCK
        message: str | None = None
        if decision in (Decision.INTERVENE, Decision.BLOCK):
            message = self._build_violation_message(worst_verdict)
            if was_escalated:
                message = escalation_info["escalation_prefix"] + "\n" + message

        metadata: dict[str, Any] = {
            "judge": {
                "verdicts": [v.to_dict() for v in verdicts],
                "session_turn": session.turn_count + 1,
            },
            "violations": [
                {
                    "name": f"judge_{v.scope.value}_{v.action.value}",
                    "message": v.summary,
                    "severity": v.action.value,
                    "composite_score": v.composite_score,
                    "judge_model": v.judge_model,
                    "scope": v.scope.value,
                }
                for v in verdicts
                if v.action != VerdictAction.PASS
            ],
        }

        if was_escalated:
            metadata["escalated"] = True
            metadata["escalation_reason"] = escalation_info["reason"]

        return EngineResult(
            decision=decision,
            message=message,
            metadata=metadata,
        )

    def _check_escalation(
        self,
        verdict: JudgeVerdict,
        session: JudgeSessionContext,
    ) -> dict[str, Any]:
        """Check if the current violation should be escalated.

        Escalation triggers:
        1. Same criterion fails after a prior intervention was applied
        2. Total intervention count exceeds cap (3)

        Returns dict with should_escalate, reason, and escalation_prefix.
        """
        result: dict[str, Any] = {
            "should_escalate": False,
            "reason": "",
            "escalation_prefix": "",
        }

        if verdict.action == VerdictAction.PASS:
            return result

        failed_criteria = verdict.metadata.get("criterion_failures", [])

        # Check 1: repeat criterion violation after prior intervention
        repeat_criteria = [
            c for c in failed_criteria
            if c in session.last_intervention_criteria
        ]
        if repeat_criteria:
            counts = {
                c: session.criterion_intervention_counts.get(c, 0) + 1
                for c in repeat_criteria
            }
            detail = ", ".join(
                f"{c} (violation #{counts[c]})" for c in repeat_criteria
            )
            result["should_escalate"] = True
            result["reason"] = f"repeat_criterion_violation: {detail}"
            result["escalation_prefix"] = (
                f"ESCALATED — repeat violation after prior intervention: {detail}."
            )
            return result

        # Check 2: total intervention count cap (3)
        # +1 because session hasn't recorded this verdict yet
        pending_count = session.intervention_count + 1
        if pending_count > 3:
            result["should_escalate"] = True
            result["reason"] = (
                f"intervention_count_exceeded: {pending_count} interventions in session"
            )
            result["escalation_prefix"] = (
                f"ESCALATED — intervention limit exceeded "
                f"({pending_count} violations in this session)."
            )
            return result

        return result

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
        """Build a natural language intervention message from failed criteria.

        Prioritizes corrective_actions (judge-generated guidance) over raw
        criterion data. Produces prose that LLMs can follow, not machine
        labels.

        Falls back to the verdict summary when no per-criterion failure
        data is available.
        """
        failed_criteria: list[str] = verdict.metadata.get("criterion_failures", [])
        if not failed_criteria:
            summary = verdict.summary or "Policy violation detected."
            # If the summary lacks directive language, append actionable guidance
            directive_markers = ("must", "should", "please", "stop", "avoid", "do not", "don't")
            if not any(marker in summary.lower() for marker in directive_markers):
                summary += (
                    " Please review and adjust your response to comply with the policy."
                )
            return summary

        score_map = {s.criterion: s for s in verdict.scores}

        paragraphs: list[str] = []
        for criterion_name in failed_criteria:
            score = score_map.get(criterion_name)
            if not score:
                paragraphs.append(f"A policy criterion was not met ({criterion_name}).")
                continue

            if score.corrective_actions:
                # Lead with the corrective action — this is the most useful signal
                parts: list[str] = [score.corrective_actions]
                if score.evidence:
                    quotes = ", ".join(f'"{e}"' for e in score.evidence)
                    parts.append(f"(Evidence from your response: {quotes})")
                paragraphs.append(" ".join(parts))
            else:
                # Fall back to reasoning + criterion description
                parts = []
                if score.reasoning:
                    parts.append(score.reasoning.rstrip(".") + ".")
                if score.evidence:
                    quotes = ", ".join(f'"{e}"' for e in score.evidence)
                    parts.append(f"(Evidence: {quotes})")
                paragraphs.append(" ".join(parts) if parts else
                                  f"A policy criterion was not met ({criterion_name}).")

        message = "\n\n".join(paragraphs)
        message += "\n\nPlease adjust your response accordingly."
        return message

