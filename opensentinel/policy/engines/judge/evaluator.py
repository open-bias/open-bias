"""
Core single-judge evaluator for the Judge Policy Engine.

Handles pointwise, pairwise, and conversation-level evaluation
by building prompts, calling the judge LLM, and parsing results
into structured verdicts.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from opensentinel.policy.engines.judge.models import (
    Rubric,
    RubricCriterion,
    JudgeScore,
    JudgeVerdict,
    JudgeSessionContext,
    VerdictAction,
    EvaluationScope,
    EvaluationType,
)
from opensentinel.policy.engines.judge.client import JudgeClient
from opensentinel.policy.engines.judge.bias import (
    randomize_positions,
    demap_pairwise_scores,
)
from opensentinel.policy.engines.judge.prompts import (
    TURN_POINTWISE_SYSTEM,
    TURN_POINTWISE_USER,
    TURN_PAIRWISE_SYSTEM,
    TURN_PAIRWISE_USER,
    TURN_REFERENCE_SYSTEM,
    TURN_REFERENCE_USER,
    CONVERSATION_SYSTEM,
    CONVERSATION_USER,
    format_criteria_block,
    format_conversation_block,
    format_metadata_block,
    format_tool_calls_block,
    format_session_context_block,
)

logger = logging.getLogger(__name__)


class JudgeEvaluator:
    """Core evaluation logic for a single judge model.

    Builds prompts from rubrics, calls the judge via JudgeClient,
    parses JSON responses into JudgeScore/JudgeVerdict objects,
    and maps composite scores to verdict actions.
    """

    def __init__(
        self,
        client: JudgeClient,
        verbose: bool = False,
    ) -> None:
        self._client = client
        self.verbose = verbose

    async def evaluate_turn(
        self,
        model_name: str,
        rubric: Rubric,
        response_content: str,
        conversation: List[Dict[str, Any]],
        reference: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        session_context: Optional[JudgeSessionContext] = None,
        tool_definitions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> JudgeVerdict:
        """Evaluate a single turn (latest assistant response).

        The conversation is provided as context, but scoring focuses
        on the latest response only.

        Args:
            model_name: Which judge model to use.
            rubric: Rubric with criteria to evaluate against.
            response_content: The assistant response to evaluate.
            conversation: Full conversation history for context.
            reference: Optional reference/ideal answer.
            metadata: Optional metadata (platform, session info, etc.).
            tool_calls: Optional tool calls from the response.
            tool_definitions: Optional tool schemas from the request.

        Returns:
            JudgeVerdict with per-criterion scores and composite.
        """
        if reference and rubric.evaluation_type in (
            EvaluationType.REFERENCE,
            EvaluationType.POINTWISE,
        ):
            return await self._evaluate_with_reference(
                model_name, rubric, response_content, conversation,
                reference, metadata, session_id=session_id,
                session_context=session_context,
            )

        criteria_block = format_criteria_block(rubric.criteria)
        conversation_block = format_conversation_block(conversation)
        metadata_block = format_metadata_block(metadata or {})
        tool_calls_block = format_tool_calls_block(tool_calls or [], tool_definitions)
        session_block = format_session_context_block(session_context)

        system_prompt = (
            rubric.prompt_overrides.get("system")
            or TURN_POINTWISE_SYSTEM.format(
                criteria_block=criteria_block,
                additional_instructions=rubric.prompt_overrides.get("additional_instructions", ""),
                session_context_block=session_block,
            )
        )
        user_prompt = (
            rubric.prompt_overrides.get("user")
            or TURN_POINTWISE_USER.format(
                conversation_block=conversation_block,
                response_content=response_content,
                tool_calls_block=tool_calls_block,
                metadata_block=metadata_block,
            )
        )


        if self.verbose:
            logger.info("=== JUDGE PROMPT (TURN) ===")
            logger.info(f"System: {self._truncate_log(system_prompt)}")
            logger.info(f"User: {self._truncate_log(user_prompt)}")
            logger.info("===========================")

        start = time.monotonic()
        raw = await self._client.call_judge(model_name, system_prompt, user_prompt, session_id=session_id)
        latency_ms = (time.monotonic() - start) * 1000

        if self.verbose:
            logger.info(f"=== JUDGE RESPONSE (TURN - {latency_ms:.2f}ms) ===")
            logger.info(raw)
            logger.info("=============================================")

        self._validate_judge_response(raw, rubric.criteria)
        scores = self._parse_pointwise_scores(raw, rubric.criteria)
        
        # Check for critical failures immediately
        failed_criteria = self._check_criterion_failures(scores, rubric.criteria)
        composite = self._compute_composite(scores, rubric.criteria)
        action = self._map_action(composite, rubric)
        model_id = self._client.get_model_id(model_name)

        # If any criterion failed, override action
        if failed_criteria and action != VerdictAction.BLOCK:
            action = rubric.fail_action

        return JudgeVerdict(
            scores=scores,
            composite_score=composite,
            action=action,
            summary=raw.get("summary", ""),
            judge_model=model_id,
            latency_ms=latency_ms,
            token_usage=self._client.get_tokens_for_model(model_name),
            scope=EvaluationScope.TURN,
            metadata={"criterion_failures": failed_criteria} if failed_criteria else {},
        )

    async def evaluate_conversation(
        self,
        model_name: str,
        rubric: Rubric,
        full_conversation: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        session_context: Optional[JudgeSessionContext] = None,
    ) -> JudgeVerdict:
        """Evaluate the entire conversation trajectory.

        The full message history IS the evaluation target. The judge
        scores cross-turn patterns, cumulative behavior, and trajectory.

        Args:
            model_name: Which judge model to use.
            rubric: Conversation-scope rubric.
            full_conversation: Complete message history.
            metadata: Optional metadata.
            session_context: Optional session state for evaluation history.

        Returns:
            JudgeVerdict with conversation-level scores.
        """
        criteria_block = format_criteria_block(rubric.criteria)
        conversation_block = format_conversation_block(full_conversation)
        metadata_block = format_metadata_block(metadata or {})
        session_block = format_session_context_block(session_context)

        # Count non-system turns
        turn_count = sum(
            1 for m in full_conversation if m.get("role") != "system"
        )

        system_prompt = (
            rubric.prompt_overrides.get("system")
            or CONVERSATION_SYSTEM.format(
                criteria_block=criteria_block,
                additional_instructions=rubric.prompt_overrides.get("additional_instructions", ""),
                session_context_block=session_block,
            )
        )
        user_prompt = (
            rubric.prompt_overrides.get("user")
            or CONVERSATION_USER.format(
                turn_count=turn_count,
                conversation_block=conversation_block,
                metadata_block=metadata_block,
            )
        )

        if self.verbose:
            logger.info("=== JUDGE PROMPT (CONVERSATION) ===")
            logger.info(f"System: {self._truncate_log(system_prompt)}")
            logger.info(f"User: {self._truncate_log(user_prompt)}")
            logger.info("===================================")

        start = time.monotonic()
        raw = await self._client.call_judge(model_name, system_prompt, user_prompt, session_id=session_id)
        latency_ms = (time.monotonic() - start) * 1000

        if self.verbose:
            logger.info(f"=== JUDGE RESPONSE (CONVERSATION - {latency_ms:.2f}ms) ===")
            logger.info(raw)
            logger.info("=====================================================")

        self._validate_judge_response(raw, rubric.criteria)
        scores = self._parse_pointwise_scores(raw, rubric.criteria)
        
        failed_criteria = self._check_criterion_failures(scores, rubric.criteria)
        composite = self._compute_composite(scores, rubric.criteria)
        action = self._map_action(composite, rubric)
        model_id = self._client.get_model_id(model_name)

        if failed_criteria and action != VerdictAction.BLOCK:
            action = rubric.fail_action

        return JudgeVerdict(
            scores=scores,
            composite_score=composite,
            action=action,
            summary=raw.get("summary", ""),
            judge_model=model_id,
            latency_ms=latency_ms,
            token_usage=self._client.get_tokens_for_model(model_name),
            scope=EvaluationScope.CONVERSATION,
            metadata={"criterion_failures": failed_criteria} if failed_criteria else {},
        )

    async def evaluate_pairwise(
        self,
        model_name: str,
        rubric: Rubric,
        response_a: str,
        response_b: str,
        conversation: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> JudgeVerdict:
        """Compare two responses using pairwise evaluation.

        Positions are randomized to mitigate position bias,
        then de-mapped after evaluation.

        Args:
            model_name: Which judge model to use.
            rubric: Pairwise rubric.
            response_a: First candidate response.
            response_b: Second candidate response.
            conversation: Conversation context.
            metadata: Optional metadata.

        Returns:
            JudgeVerdict with comparison scores (de-mapped).
        """
        # Randomize positions to mitigate bias
        first, second, mapping = randomize_positions(response_a, response_b)

        criteria_block = format_criteria_block(rubric.criteria)
        conversation_block = format_conversation_block(conversation)
        metadata_block = format_metadata_block(metadata or {})

        system_prompt = (
            rubric.prompt_overrides.get("system")
            or TURN_PAIRWISE_SYSTEM.format(criteria_block=criteria_block)
        )
        user_prompt = (
            rubric.prompt_overrides.get("user")
            or TURN_PAIRWISE_USER.format(
                conversation_block=conversation_block,
                response_a=first,
                response_b=second,
                metadata_block=metadata_block,
            )
        )

        start = time.monotonic()
        raw = await self._client.call_judge(model_name, system_prompt, user_prompt, session_id=session_id)
        latency_ms = (time.monotonic() - start) * 1000

        # De-map positions back to original a/b
        raw_scores = raw.get("scores", [])
        demapped_scores = demap_pairwise_scores(raw_scores, mapping)



        # Build JudgeScores from the "a" side scores
        scores = self._parse_pairwise_scores(demapped_scores, rubric.criteria)
        
        # Check for critical failures
        failed_criteria = self._check_criterion_failures(scores, rubric.criteria)
        composite = self._compute_composite(scores, rubric.criteria)
        action = self._map_action(composite, rubric)
        model_id = self._client.get_model_id(model_name)

        if failed_criteria and action != VerdictAction.BLOCK:
            action = rubric.fail_action

        return JudgeVerdict(
            scores=scores,
            composite_score=composite,
            action=action,
            summary=raw.get("summary", ""),
            judge_model=model_id,
            latency_ms=latency_ms,
            token_usage=self._client.get_tokens_for_model(model_name),
            scope=EvaluationScope.TURN,
            metadata={
                "pairwise": True,
                "overall_winner": raw.get("overall_winner", "tie"),
                "position_mapping": mapping,
                "criterion_failures": failed_criteria,
            },
        )

    async def _evaluate_with_reference(
        self,
        model_name: str,
        rubric: Rubric,
        response_content: str,
        conversation: List[Dict[str, Any]],
        reference: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        session_context: Optional[JudgeSessionContext] = None,
    ) -> JudgeVerdict:
        """Evaluate a response against a reference answer."""
        criteria_block = format_criteria_block(rubric.criteria)
        conversation_block = format_conversation_block(conversation)
        metadata_block = format_metadata_block(metadata or {})
        session_block = format_session_context_block(session_context)

        system_prompt = (
            rubric.prompt_overrides.get("system")
            or TURN_REFERENCE_SYSTEM.format(
                criteria_block=criteria_block,
                ref_scale="1-5",
                additional_instructions=rubric.prompt_overrides.get("additional_instructions", ""),
                session_context_block=session_block,
            )
        )
        user_prompt = (
            rubric.prompt_overrides.get("user")
            or TURN_REFERENCE_USER.format(
                conversation_block=conversation_block,
                response_content=response_content,
                reference_answer=reference,
                metadata_block=metadata_block,
            )
        )

        start = time.monotonic()
        raw = await self._client.call_judge(model_name, system_prompt, user_prompt, session_id=session_id)
        latency_ms = (time.monotonic() - start) * 1000

        self._validate_judge_response(raw, rubric.criteria)
        scores = self._parse_pointwise_scores(raw, rubric.criteria)
        
        failed_criteria = self._check_criterion_failures(scores, rubric.criteria)
        composite = self._compute_composite(scores, rubric.criteria)
        action = self._map_action(composite, rubric)
        model_id = self._client.get_model_id(model_name)

        if failed_criteria and action != VerdictAction.BLOCK:
            action = rubric.fail_action

        return JudgeVerdict(
            scores=scores,
            composite_score=composite,
            action=action,
            summary=raw.get("summary", ""),
            judge_model=model_id,
            latency_ms=latency_ms,
            token_usage=self._client.get_tokens_for_model(model_name),
            scope=EvaluationScope.TURN,
            metadata={
                "reference_based": True,
                "criterion_failures": failed_criteria,
            },
        )

    # =========================================================================
    # Parsing & Scoring
    # =========================================================================

    def _parse_pointwise_scores(
        self,
        raw: Dict[str, Any],
        criteria: List[RubricCriterion],
    ) -> List[JudgeScore]:
        """Parse pointwise scores from raw LLM JSON response."""
        raw_scores = raw.get("scores", [])
        criteria_map = {c.name: c for c in criteria}
        scores = []

        for raw_score in raw_scores:
            criterion_name = raw_score.get("criterion", "")
            criterion = criteria_map.get(criterion_name)
            if not criterion:
                logger.warning(f"Unknown criterion in judge response: {criterion_name}")
                continue

            score_val = int(raw_score.get("score", 0))
            # Clamp score to validity range
            score_val = max(criterion.scale.min_score, min(criterion.scale.max_score, score_val))

            scores.append(JudgeScore(
                criterion=criterion_name,
                score=score_val,
                max_score=criterion.scale.max_score,
                reasoning=raw_score.get("reasoning", ""),
                evidence=raw_score.get("evidence", []),
                confidence=float(raw_score.get("confidence", 1.0)),
                corrective_actions=raw_score.get("corrective_actions"),
            ))

        # Fill in missing criteria with minimum scores
        scored_names = {s.criterion for s in scores}
        for criterion in criteria:
            if criterion.name not in scored_names:
                logger.warning(f"Judge did not score criterion: {criterion.name}")
                scores.append(JudgeScore(
                    criterion=criterion.name,
                    score=criterion.scale.min_score,
                    max_score=criterion.scale.max_score,
                    reasoning="Not evaluated by judge",
                    confidence=0.0,
                ))

        return scores

    def _parse_pairwise_scores(
        self,
        demapped_scores: List[Dict[str, Any]],
        criteria: List[RubricCriterion],
    ) -> List[JudgeScore]:
        """Parse pairwise scores into JudgeScore objects.

        Uses score_a as the primary score (evaluating response A).
        """
        criteria_map = {c.name: c for c in criteria}
        scores = []

        for raw_score in demapped_scores:
            criterion_name = raw_score.get("criterion", "")
            criterion = criteria_map.get(criterion_name)
            if not criterion:
                continue

            score_val = int(raw_score.get("score_a", 0))
            # Clamp score
            score_val = max(criterion.scale.min_score, min(criterion.scale.max_score, score_val))

            scores.append(JudgeScore(
                criterion=criterion_name,
                score=score_val,
                max_score=criterion.scale.max_score,
                reasoning=raw_score.get("reasoning", ""),
                evidence=raw_score.get("evidence", []),
                confidence=float(raw_score.get("confidence", 1.0)),
                corrective_actions=raw_score.get("corrective_actions"),
            ))

        return scores

    def _compute_composite(
        self,
        scores: List[JudgeScore],
        criteria: List[RubricCriterion],
    ) -> float:
        """Compute weighted normalized composite score (0-1).

        Each score is normalized to 0-1 using its scale, then
        weighted by the criterion weight.
        """
        if not scores:
            return 0.0

        criteria_map = {c.name: c for c in criteria}
        total_weight = 0.0
        weighted_sum = 0.0

        for score in scores:
            criterion = criteria_map.get(score.criterion)
            weight = criterion.weight if criterion else 1.0
            weighted_sum += score.normalized * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0

        return weighted_sum / total_weight

    def _map_action(self, composite: float, rubric: Rubric) -> VerdictAction:
        """Map composite score to a verdict action.

        Binary: pass if above rubric threshold, otherwise rubric's fail_action.
        """
        if composite >= rubric.pass_threshold:
            return VerdictAction.PASS
        return rubric.fail_action

    def _check_criterion_failures(
        self,
        scores: List[JudgeScore],
        criteria: List[RubricCriterion],
    ) -> List[str]:
        """Check if any individual criteria fall below their fail thresholds.

        Returns list of criterion names that failed.
        """
        criteria_map = {c.name: c for c in criteria}
        failures = []

        for score in scores:
            # Skip synthetic fills (confidence=0.0) — these are placeholders
            # for criteria the judge LLM omitted, not real failures
            if score.confidence == 0.0:
                continue
            criterion = criteria_map.get(score.criterion)
            if criterion and criterion.fail_threshold is not None:
                if score.normalized < criterion.fail_threshold:
                    failures.append(score.criterion)

        return failures


    def _validate_judge_response(
        self,
        raw: Dict[str, Any],
        criteria: List[RubricCriterion],
    ) -> None:
        """Validate the structure of the judge LLM response."""
        if "scores" not in raw:
            raise ValueError("Judge response missing 'scores' key")
        
        if not isinstance(raw["scores"], list):
            raise ValueError("Judge response 'scores' must be a list")

        # Basic schema check for each score item
        for item in raw["scores"]:
            if not isinstance(item, dict):
                raise ValueError("Score item must be a dictionary")
            if "criterion" not in item:
                raise ValueError("Score item missing 'criterion'")
            if "score" not in item:
                raise ValueError(f"Score item for '{item.get('criterion')}' missing 'score'")

    def _truncate_log(self, text: str, max_len: int = 2000) -> str:
        """Truncate text for logging if it exceeds max_len."""
        if len(text) <= max_len:
            return text
        return text[:max_len] + f"... (truncated {len(text) - max_len} chars)"
