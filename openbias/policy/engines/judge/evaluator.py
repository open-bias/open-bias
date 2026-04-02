"""Turn-level binary evaluator for the simplified judge engine."""

import json
import logging
import time
from typing import Any

from openbias.policy.engines.judge.client import JudgeClient
from openbias.policy.engines.judge.models import (
    EvaluationScope,
    JudgeScore,
    JudgeSessionContext,
    JudgeVerdict,
    VerdictAction,
)
from openbias.policy.engines.judge.prompts import (
    RULES_TURN_SYSTEM,
    RULES_TURN_USER,
    format_conversation_block,
    format_metadata_block,
    format_session_context_block,
    format_tool_calls_block,
)

logger = logging.getLogger(__name__)


class JudgeEvaluator:
    """Core evaluation logic for per-rule binary checks."""

    def __init__(self, client: JudgeClient, verbose: bool = False) -> None:
        self._client = client
        self.verbose = verbose

    async def evaluate_turn(
        self,
        model_name: str,
        rules: list[str],
        response_content: str,
        conversation: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        session_context: JudgeSessionContext | None = None,
        tool_definitions: dict[str, dict[str, Any]] | None = None,
        fail_action: VerdictAction = VerdictAction.INTERVENE,
    ) -> JudgeVerdict:
        if not rules:
            raise ValueError("Judge evaluator requires at least one rule.")

        rules_block = "\n".join(f"- Rule {i}: {rule}" for i, rule in enumerate(rules, 1))
        conversation_block = format_conversation_block(conversation)
        metadata_block = format_metadata_block(metadata or {})
        tool_calls_block = format_tool_calls_block(tool_calls or [], tool_definitions)
        session_block = format_session_context_block(session_context)

        system_prompt = RULES_TURN_SYSTEM.format(
            rules_block=rules_block,
            session_context_block=session_block,
        )
        user_prompt = RULES_TURN_USER.format(
            conversation_block=conversation_block,
            response_content=response_content,
            tool_calls_block=tool_calls_block,
            metadata_block=metadata_block,
        )

        if self.verbose:
            logger.info("=== JUDGE PROMPT (TURN BINARY) ===")
            logger.info("System: %s", self._truncate_log(system_prompt))
            logger.info("User: %s", self._truncate_log(user_prompt))
            logger.info("==================================")

        start = time.monotonic()
        raw = await self._client.call_judge(model_name, system_prompt, user_prompt, session_id=session_id)
        latency_ms = (time.monotonic() - start) * 1000
        raw = self._coerce_raw_payload(raw)

        if self.verbose:
            logger.info("=== JUDGE RESPONSE (TURN BINARY - %.2fms) ===", latency_ms)
            logger.info("%s", raw)
            logger.info("=============================================")

        scores = self._parse_rule_results(raw, rules)
        failed_rules = [score.criterion for score in scores if score.score == 0]
        action = fail_action if failed_rules else VerdictAction.PASS
        composite = 0.0 if failed_rules else 1.0

        return JudgeVerdict(
            scores=scores,
            composite_score=composite,
            action=action,
            summary=str(raw.get("summary", "")),
            judge_model=self._client.get_model_id(model_name),
            latency_ms=latency_ms,
            token_usage=self._client.get_tokens_for_model(model_name),
            scope=EvaluationScope.TURN,
            metadata={"criterion_failures": failed_rules} if failed_rules else {},
        )

    def _parse_rule_results(self, raw: dict[str, Any], rules: list[str]) -> list[JudgeScore]:
        items = raw.get("results")
        if not isinstance(items, list):
            raise ValueError("Judge response missing 'results' list.")

        by_rule: dict[str, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            rule_text = str(item.get("rule", "")).strip()
            if rule_text:
                by_rule[rule_text] = item

        scores: list[JudgeScore] = []
        for rule_text in rules:
            payload = by_rule.get(rule_text)
            if payload is None:
                scores.append(
                    JudgeScore(
                        criterion=rule_text,
                        score=0,
                        reasoning="Rule not evaluated by judge.",
                        confidence=0.0,
                    )
                )
                continue

            raw_passed = payload.get("passed")
            passed = bool(raw_passed) if isinstance(raw_passed, bool) else False
            evidence = payload.get("evidence", [])
            scores.append(
                JudgeScore(
                    criterion=rule_text,
                    score=1 if passed else 0,
                    reasoning=str(payload.get("reasoning", "")),
                    evidence=evidence if isinstance(evidence, list) else [],
                    confidence=float(payload.get("confidence", 1.0)),
                    corrective_actions=payload.get("corrective_actions"),
                )
            )
        return scores

    def _coerce_raw_payload(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        raise ValueError("Judge response must be a JSON object.")

    def _truncate_log(self, text: str, max_len: int = 2000) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + f"... (truncated {len(text) - max_len} chars)"
