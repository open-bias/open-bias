"""
LLM-as-a-Judge Policy Engine.

Evaluates agent responses and conversation trajectories against
configurable rubrics using LLM judges. Integrates with the Open Sentinel
policy engine infrastructure via PolicyEngine ABC.
"""

import logging
import time
from collections import OrderedDict
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from opensentinel.policy.engines.judge.ensemble import JudgeEnsemble
    from opensentinel.policy.compiler.protocol import PolicyCompiler

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
from opensentinel.policy.engines.judge.client import JudgeClient
from opensentinel.policy.engines.judge.evaluator import JudgeEvaluator
from opensentinel.policy.engines.judge.rubrics import (
    RubricRegistry,
    create_rules_rubric,
    _parse_rubric_dict,
)

logger = logging.getLogger(__name__)

# Mapping from VerdictAction to Decision
_VERDICT_MAP: Dict[VerdictAction, Decision] = {
    VerdictAction.PASS: Decision.ALLOW,
    VerdictAction.WARN: Decision.ALLOW,
    VerdictAction.INTERVENE: Decision.INTERVENE,
    VerdictAction.BLOCK: Decision.BLOCK,
    VerdictAction.ESCALATE: Decision.INTERVENE,
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
        self._client: Optional[JudgeClient] = None
        self._evaluator: Optional[JudgeEvaluator] = None
        self._ensemble: Optional["JudgeEnsemble"] = None
        self._sessions: OrderedDict[str, JudgeSessionContext] = OrderedDict()
        self._session_timestamps: OrderedDict[str, float] = OrderedDict()
        self._tracer: Optional[Any] = None

        # Session memory config (can be overridden in initialize())
        self._session_ttl = self.DEFAULT_SESSION_TTL
        self._max_sessions = self.DEFAULT_MAX_SESSIONS

        # Config
        self._default_rubric: str = "agent_behavior"
        self._conversation_rubric: Optional[str] = "conversation_policy"
        self._pre_call_enabled: bool = False
        self._pre_call_rubric: str = "safety"
        self._conversation_eval_interval: int = 5
        self._ensemble_enabled: bool = False

    @property
    def name(self) -> str:
        return f"judge:{self._default_rubric}"

    @property
    def engine_type(self) -> str:
        return "judge"

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the judge engine with configuration.

        Args:
            config: Configuration dict with:
                - models: List of judge model configs [{name, model, temperature, ...}]
                - default_rubric: Name of default turn-scope rubric
                - conversation_rubric: Name of conversation-scope rubric (or null to disable)
                - pre_call_enabled: Whether to evaluate requests (default: false)
                - pre_call_rubric: Rubric for pre-call evaluation
                - pass_threshold: Score threshold for PASS (default: 0.6)
                - warn_threshold: Score threshold for WARN (default: 0.4)
                - block_threshold: Score threshold for BLOCK (default: 0.2)
                - conversation_eval_interval: Run conversation eval every N turns (default: 5)
                - custom_rubrics_path: Path to custom rubric YAML files
                - checker_mode: "async" or "sync" (used by interceptor, not engine)
        """
        # Build client with judge models
        self._client = JudgeClient()
        models = config.get("models", [])
        
        if not models:
            # No explicit models — create a primary model entry.
            # model comes from config (injected by SentinelSettings.get_policy_config).
            models = [{
                "name": "primary",
                "model": config.get("default_model") or config.get("llm_model"),
                "temperature": 0.0,
            }]
            logger.info("No judge models explicitly configured; using default_model from config")

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
            pass_threshold=config.get("pass_threshold", 0.6),
            warn_threshold=config.get("warn_threshold", 0.4),
            block_threshold=config.get("block_threshold", 0.2),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            verbose=config.get("verbose", False),
        )

        # Config
        self._default_rubric = config.get("default_rubric", "agent_behavior")
        self._conversation_rubric = config.get("conversation_rubric", "conversation_policy")
        self._pre_call_enabled = config.get("pre_call_enabled", False)
        self._pre_call_rubric = config.get("pre_call_rubric", "safety")
        self._conversation_eval_interval = config.get("conversation_eval_interval", 5)

        # Ensemble configuration
        self._ensemble_enabled = config.get("ensemble_enabled", False)
        if self._ensemble_enabled and len(self._client.model_names) > 1:
            from opensentinel.policy.engines.judge.ensemble import JudgeEnsemble, AggregationStrategy
            strategy = config.get("aggregation_strategy", AggregationStrategy.MEAN_SCORE)
            min_agreement = config.get("min_agreement", 0.6)
            self._ensemble = JudgeEnsemble(
                evaluator=self._evaluator,
                strategy=strategy,
                min_agreement=min_agreement,
            )
            logger.info(f"Ensemble enabled: strategy={strategy}, min_agreement={min_agreement}")

        # Session memory management
        if "session_ttl" in config:
            self._session_ttl = int(config["session_ttl"])
        if "max_sessions" in config:
            self._max_sessions = int(config["max_sessions"])

        # Load custom rubrics if configured
        custom_path = config.get("custom_rubrics_path")
        if custom_path:
            RubricRegistry.load_from_yaml(custom_path)

        # Load inline policy if provided
        inline_policy = config.get("inline_policy")
        if inline_policy is not None:
            self._load_inline_policy(inline_policy)

        self._initialized = True
        logger.info(f"JudgePolicyEngine initialized: {self.name}")

    def set_tracer(self, tracer: Any) -> None:
        """Set an optional SentinelTracer for OTEL tracing."""
        self._tracer = tracer

    async def evaluate_request(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EngineResult:
        """Evaluate an incoming request (PRE_CALL).

        Default: ALLOW (most judgment happens post-call).
        If pre_call_enabled, run safety screening.
        """
        if not self._initialized:
            return EngineResult(decision=Decision.ALLOW)

        # Optional pre-call screening
        if self._pre_call_enabled:
            return await self._evaluate_pre_call(session_id, request_data, context)

        return EngineResult(decision=Decision.ALLOW)

    async def evaluate_response(
        self,
        session_id: str,
        response_data: Any,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EngineResult:
        """Evaluate an LLM response (POST_CALL).

        Main evaluation path:
        1. Always run turn-scope rubric on latest response
        2. Run conversation-scope rubric on interval or when triggered
        3. Merge verdicts (most restrictive action wins)
        4. Map to EngineResult
        """
        if not self._initialized:
            return EngineResult(decision=Decision.ALLOW)

        session = self._get_or_create_session(session_id)
        response_content = self._extract_response_content(response_data)
        tool_calls = self._extract_tool_calls(response_data)
        conversation = self._extract_conversation(request_data)
        metadata = (context or {}).get("metadata", {})

        primary_model = self._client.primary_model
        if not primary_model:
            logger.error("No judge models configured")
            return EngineResult(decision=Decision.ALLOW)

        verdicts: List[JudgeVerdict] = []

        # 1. Turn-scope evaluation (always runs)
        turn_rubric = RubricRegistry.get(self._default_rubric)
        if turn_rubric:
            try:
                if self._ensemble_enabled and self._ensemble:
                    ensemble_verdict = await self._ensemble.evaluate_turn(
                        model_names=self._client.model_names,
                        rubric=turn_rubric,
                        response_content=response_content,
                        conversation=conversation,
                        metadata=metadata,
                        session_id=session_id,
                        tool_calls=tool_calls,
                    )
                    # Use ensemble's final verdict as a JudgeVerdict
                    turn_verdict = JudgeVerdict(
                        scores=ensemble_verdict.final_scores,
                        composite_score=ensemble_verdict.final_composite,
                        action=ensemble_verdict.final_action,
                        summary=f"Ensemble ({ensemble_verdict.aggregation_strategy}), "
                                f"agreement={ensemble_verdict.agreement_rate:.2f}",
                        judge_model="ensemble",
                        scope=EvaluationScope.TURN,
                        metadata={
                            "ensemble": True,
                            "agreement_rate": ensemble_verdict.agreement_rate,
                            "criterion_agreement": ensemble_verdict.criterion_agreement,
                            "individual_count": len(ensemble_verdict.individual_verdicts),
                        },
                    )
                    self._trace_verdict(session_id, turn_verdict, turn_rubric.name,
                                        ensemble=True,
                                        agreement_rate=ensemble_verdict.agreement_rate)
                else:
                    turn_verdict = await self._evaluator.evaluate_turn(
                        model_name=primary_model,
                        rubric=turn_rubric,
                        response_content=response_content,
                        conversation=conversation,
                        metadata=metadata,
                        session_id=session_id,
                        tool_calls=tool_calls,
                    )
                    self._trace_verdict(session_id, turn_verdict, turn_rubric.name)
                verdicts.append(turn_verdict)
            except Exception as e:
                logger.error(f"Turn evaluation failed: {e}")
        else:
            logger.warning(f"Default rubric not found: {self._default_rubric}")

        # 2. Conversation-scope evaluation (on interval or trigger)
        if self._should_run_conversation_eval(session, verdicts):
            conv_rubric = RubricRegistry.get(self._conversation_rubric)
            if conv_rubric:
                try:
                    if self._ensemble_enabled and self._ensemble:
                        ensemble_verdict = await self._ensemble.evaluate_conversation(
                            model_names=self._client.model_names,
                            rubric=conv_rubric,
                            full_conversation=conversation,
                            metadata=metadata,
                            session_id=session_id,
                        )
                        conv_verdict = JudgeVerdict(
                            scores=ensemble_verdict.final_scores,
                            composite_score=ensemble_verdict.final_composite,
                            action=ensemble_verdict.final_action,
                            summary=f"Conversation ensemble ({ensemble_verdict.aggregation_strategy}), "
                                    f"agreement={ensemble_verdict.agreement_rate:.2f}",
                            judge_model="ensemble",
                            scope=EvaluationScope.CONVERSATION,
                            metadata={
                                "ensemble": True,
                                "agreement_rate": ensemble_verdict.agreement_rate,
                            },
                        )
                        self._trace_verdict(session_id, conv_verdict, conv_rubric.name,
                                            ensemble=True,
                                            agreement_rate=ensemble_verdict.agreement_rate)
                    else:
                        conv_verdict = await self._evaluator.evaluate_conversation(
                            model_name=primary_model,
                            rubric=conv_rubric,
                            full_conversation=conversation,
                            metadata=metadata,
                            session_id=session_id,
                        )
                        self._trace_verdict(session_id, conv_verdict, conv_rubric.name)
                    verdicts.append(conv_verdict)
                except Exception as e:
                    logger.error(f"Conversation evaluation failed: {e}")

        # 3. Merge verdicts and build result
        if not verdicts:
            return EngineResult(decision=Decision.ALLOW)

        # Build result before recording verdicts so escalation checks
        # compare against prior session state, not the current turn's own data
        result = self._build_result(verdicts, session)

        # 4. Record verdicts to session after result is built
        for v in verdicts:
            session.record_verdict(v)

        return result

    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session state for debugging/tracing."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        return session.to_dict()

    async def reset_session(self, session_id: str) -> None:
        """Reset session state."""
        self._sessions.pop(session_id, None)
        self._session_timestamps.pop(session_id, None)

    def get_compiler(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Optional["PolicyCompiler"]:
        """Return a JudgeCompiler instance."""
        from opensentinel.policy.engines.judge.compiler import JudgeCompiler
        kwargs: Dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return JudgeCompiler(**kwargs)

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._sessions.clear()
        self._session_timestamps.clear()
        logger.info("JudgePolicyEngine shut down")

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _load_inline_policy(self, policy_data: Any) -> None:
        """Load inline policy definitions from config.

        Handles several shapes:
        - List of strings: plain-text rules → auto-generated binary rubric
        - Multiline string: split into rules
        - Dict with 'rules' key: extract rules list
        - Dict with 'rubrics' key: parse each as formal rubric
        - List of dicts: parse each as formal rubric
        """
        if isinstance(policy_data, str):
            # Multiline string → split into rules
            rules = [line.strip() for line in policy_data.strip().splitlines() if line.strip()]
            if rules:
                rubric = create_rules_rubric(rules)
                RubricRegistry.register(rubric)
                self._default_rubric = rubric.name
                logger.info(f"Loaded {len(rules)} inline policy rules as '{rubric.name}'")
            return

        if isinstance(policy_data, list):
            if not policy_data:
                return
            # List of strings → plain-text rules
            if all(isinstance(item, str) for item in policy_data):
                rubric = create_rules_rubric(policy_data)
                RubricRegistry.register(rubric)
                self._default_rubric = rubric.name
                logger.info(f"Loaded {len(policy_data)} inline policy rules as '{rubric.name}'")
                return
            # List of dicts → formal rubric definitions
            for item in policy_data:
                if isinstance(item, dict):
                    try:
                        rubric = _parse_rubric_dict(item)
                        RubricRegistry.register(rubric)
                        # First rubric becomes default
                        if policy_data.index(item) == 0:
                            self._default_rubric = rubric.name
                        logger.info(f"Loaded inline rubric '{rubric.name}'")
                    except Exception as e:
                        logger.error(f"Failed to parse inline rubric: {e}")
            return

        if isinstance(policy_data, dict):
            if "rules" in policy_data:
                rules = policy_data["rules"]
                if isinstance(rules, list):
                    rubric = create_rules_rubric(rules)
                    RubricRegistry.register(rubric)
                    self._default_rubric = rubric.name
                    logger.info(f"Loaded {len(rules)} inline policy rules as '{rubric.name}'")
                return
            if "rubrics" in policy_data:
                for rubric_def in policy_data["rubrics"]:
                    try:
                        rubric = _parse_rubric_dict(rubric_def)
                        RubricRegistry.register(rubric)
                        logger.info(f"Loaded inline rubric '{rubric.name}'")
                    except Exception as e:
                        logger.error(f"Failed to parse inline rubric: {e}")
                # Set first rubric as default
                if policy_data["rubrics"]:
                    first = policy_data["rubrics"][0]
                    if isinstance(first, dict) and "name" in first:
                        self._default_rubric = first["name"]
                return

        logger.warning(f"Unrecognized inline_policy format: {type(policy_data)}")

    def _trace_verdict(
        self,
        session_id: str,
        verdict: JudgeVerdict,
        rubric_name: str,
        ensemble: bool = False,
        agreement_rate: Optional[float] = None,
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
                ensemble=ensemble,
                agreement_rate=agreement_rate,
                metadata=verdict.metadata,
            )
        except Exception as e:
            logger.debug(f"Failed to trace verdict: {e}")

    def _get_or_create_session(self, session_id: str) -> JudgeSessionContext:
        if session_id not in self._sessions:
            self._evict_stale_sessions()
            self._sessions[session_id] = JudgeSessionContext(session_id=session_id)
        # Touch for LRU tracking
        self._session_timestamps[session_id] = time.monotonic()
        self._session_timestamps.move_to_end(session_id)
        self._sessions.move_to_end(session_id)
        return self._sessions[session_id]

    def _evict_stale_sessions(self) -> None:
        """Remove sessions that have exceeded their TTL or breach the max cap."""
        now = time.monotonic()

        # TTL eviction (oldest-first)
        stale_ids: list[str] = []
        for sid, ts in self._session_timestamps.items():
            if now - ts > self._session_ttl:
                stale_ids.append(sid)
            else:
                break

        for sid in stale_ids:
            self._sessions.pop(sid, None)
            self._session_timestamps.pop(sid, None)

        if stale_ids:
            logger.debug("Evicted %d stale judge sessions (TTL=%ds)", len(stale_ids), self._session_ttl)

        # Hard cap eviction
        overflow = len(self._sessions) - self._max_sessions
        if overflow > 0:
            oldest = list(self._sessions.keys())[:overflow]
            for sid in oldest:
                self._sessions.pop(sid, None)
                self._session_timestamps.pop(sid, None)
            logger.debug("Evicted %d judge sessions (max=%d)", overflow, self._max_sessions)

    def _should_run_conversation_eval(
        self,
        session: JudgeSessionContext,
        turn_verdicts: List[JudgeVerdict],
    ) -> bool:
        """Determine if conversation-scope evaluation should run."""
        if not self._conversation_rubric:
            return False

        # Run on interval
        if (
            session.turn_count > 0
            and session.turn_count % self._conversation_eval_interval == 0
        ):
            return True

        # Run when a turn verdict is warn or worse
        for v in turn_verdicts:
            if v.action in (VerdictAction.WARN, VerdictAction.INTERVENE, VerdictAction.BLOCK):
                return True

        return False

    async def _evaluate_pre_call(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EngineResult:
        """Run pre-call safety screening on user message."""
        rubric = RubricRegistry.get(self._pre_call_rubric)
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
            )
            return self._build_result([verdict], self._get_or_create_session(session_id))
        except Exception as e:
            logger.error(f"Pre-call evaluation failed: {e}")
            return EngineResult(decision=Decision.ALLOW)

    def _build_result(
        self,
        verdicts: List[JudgeVerdict],
        session: JudgeSessionContext,
    ) -> EngineResult:
        """Build EngineResult from judge verdicts.

        Takes the most restrictive action across all verdicts.
        Applies escalation when repeat violations are detected:
        - Same criterion fails after prior intervention → escalate to BLOCK
        - Total intervention count exceeds cap (3) → auto-block
        """
        action_priority = {
            VerdictAction.PASS: 0,
            VerdictAction.WARN: 1,
            VerdictAction.ESCALATE: 2,
            VerdictAction.INTERVENE: 3,
            VerdictAction.BLOCK: 4,
        }

        worst_verdict = max(verdicts, key=lambda v: action_priority.get(v.action, 0))
        decision = _VERDICT_MAP[worst_verdict.action]

        # Check for escalation conditions
        escalation_info = self._check_escalation(worst_verdict, session)

        if escalation_info["should_escalate"] and decision != Decision.BLOCK:
            decision = Decision.BLOCK

        # message = guidance for INTERVENE, reason for BLOCK
        message: Optional[str] = None
        if decision in (Decision.INTERVENE, Decision.BLOCK):
            message = self._build_violation_message(worst_verdict)
            if escalation_info["should_escalate"]:
                message = escalation_info["escalation_prefix"] + "\n" + message

        any_low_confidence = any(v.low_confidence for v in verdicts)

        metadata: Dict[str, Any] = {
            "judge": {
                "verdicts": [v.to_dict() for v in verdicts],
                "session_turn": session.turn_count,
                "low_confidence": any_low_confidence,
            },
            "violations": [
                {
                    "name": f"judge_{v.scope.value}_{v.action.value}",
                    "composite_score": v.composite_score,
                    "judge_model": v.judge_model,
                    "scope": v.scope.value,
                    "summary": v.summary,
                }
                for v in verdicts
                if v.action != VerdictAction.PASS
            ],
        }

        if worst_verdict.action == VerdictAction.ESCALATE:
            metadata["escalate"] = True

        if escalation_info["should_escalate"]:
            metadata["escalated"] = True
            metadata["escalation_reason"] = escalation_info["reason"]

        if any_low_confidence:
            metadata["judge"]["confidence_warning"] = (
                "One or more judge evaluations had low confidence. "
                "Results may be unreliable."
            )

        return EngineResult(
            decision=decision,
            message=message,
            metadata=metadata,
        )

    def _check_escalation(
        self,
        verdict: JudgeVerdict,
        session: JudgeSessionContext,
    ) -> Dict[str, Any]:
        """Check if the current violation should be escalated.

        Escalation triggers:
        1. Same criterion fails after a prior intervention was applied
        2. Total intervention count exceeds cap (3)

        Returns dict with should_escalate, reason, and escalation_prefix.
        """
        result: Dict[str, Any] = {
            "should_escalate": False,
            "reason": "",
            "escalation_prefix": "",
        }

        if verdict.action in (VerdictAction.PASS, VerdictAction.WARN):
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

    def _extract_response_content(self, response_data: Any) -> str:
        """Extract text content from response data."""
        if isinstance(response_data, str):
            return response_data
        if isinstance(response_data, dict):
            # OpenAI-style response
            choices = response_data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "")
            return response_data.get("content", "")
        return str(response_data)

    def _extract_tool_calls(self, response_data: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from response data.

        Handles OpenAI dict format and object format responses.
        Returns a list of tool call dicts with 'id', 'function_name', and 'arguments'.
        """
        tool_calls: List[Dict[str, Any]] = []

        if isinstance(response_data, dict):
            # OpenAI dict format: choices[0].message.tool_calls
            choices = response_data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                raw_calls = message.get("tool_calls", [])
            else:
                raw_calls = response_data.get("tool_calls", [])

            for tc in raw_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    tool_calls.append({
                        "id": tc.get("id", ""),
                        "function_name": func.get("name", ""),
                        "arguments": func.get("arguments", ""),
                    })

        elif hasattr(response_data, "choices") and response_data.choices:
            choice = response_data.choices[0]
            if hasattr(choice, "message") and choice.message:
                raw_calls = getattr(choice.message, "tool_calls", None) or []
                for tc in raw_calls:
                    if hasattr(tc, "function") and tc.function:
                        tool_calls.append({
                            "id": getattr(tc, "id", ""),
                            "function_name": getattr(tc.function, "name", ""),
                            "arguments": getattr(tc.function, "arguments", ""),
                        })

        return tool_calls

    def _extract_conversation(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract conversation messages from request data."""
        return request_data.get("messages", [])

    def _build_violation_message(self, verdict: JudgeVerdict) -> str:
        """Build an intervention message citing specific failed criteria.

        Constructs a targeted message from per-criterion data including:
        - Which criteria failed
        - The judge's reasoning and evidence for each failure
        - Corrective actions (what the agent should do instead)

        Falls back to the verdict summary when no per-criterion failure
        data is available.
        """
        failed_criteria: List[str] = verdict.metadata.get("criterion_failures", [])
        if not failed_criteria:
            return verdict.summary or "Policy violation detected."

        # Build a map from criterion name → JudgeScore for lookup
        score_map = {s.criterion: s for s in verdict.scores}

        parts: List[str] = ["POLICY VIOLATION:"]
        for criterion_name in failed_criteria:
            score = score_map.get(criterion_name)
            if not score:
                parts.append(f"- {criterion_name}")
                continue

            # Criterion name + reasoning
            line = f"- {criterion_name} FAILED."
            if score.reasoning:
                line += f" {score.reasoning}"
            parts.append(line)

            # Evidence citations
            if score.evidence:
                for e in score.evidence:
                    parts.append(f"  Evidence: {e}")

            # Corrective actions
            if score.corrective_actions:
                parts.append(f"  REQUIRED: {score.corrective_actions}")

        return "\n".join(parts)

