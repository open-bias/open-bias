"""Turn-level binary evaluator for the simplified judge engine."""

import json
import logging
import time
from typing import Any

from openbias.policy.engines.judge.client import JudgeClient
from openbias.policy.engines.judge.models import (
    JudgeRuleResult,
    JudgeSessionContext,
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

    async def evaluate_rule(
        self,
        model_name: str,
        rule: str,
        response_content: str,
        conversation: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        session_context: JudgeSessionContext | None = None,
        tool_definitions: dict[str, dict[str, Any]] | None = None,
    ) -> JudgeRuleResult:
        if not rule.strip():
            raise ValueError("Judge evaluator requires a non-empty rule.")

        rules_block = f"- Rule 1: {rule}"
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

        result = self._parse_rule_result(raw, rule)
        result.judge_name = model_name
        result.judge_model = self._client.get_model_id(model_name)
        result.latency_ms = latency_ms
        result.token_usage = self._client.get_tokens_for_model(model_name)
        result.metadata["judge_summary"] = str(raw.get("summary", ""))
        return result

    def _parse_rule_result(self, raw: dict[str, Any], rule: str) -> JudgeRuleResult:
        items = raw.get("results")
        if not isinstance(items, list):
            raise ValueError("Judge response missing 'results' list.")

        payload: dict[str, Any] | None = None
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("rule", "")).strip() == rule:
                payload = item
                break

        if payload is None:
            return JudgeRuleResult(
                rule=rule,
                passed=False,
                reasoning="Rule not evaluated by judge.",
                confidence=0.0,
            )

        raw_passed = payload.get("passed")
        passed = bool(raw_passed) if isinstance(raw_passed, bool) else False
        evidence = payload.get("evidence", [])
        return JudgeRuleResult(
            rule=rule,
            passed=passed,
            reasoning=str(payload.get("reasoning", "")),
            evidence=evidence if isinstance(evidence, list) else [],
            confidence=float(payload.get("confidence", 1.0)),
            corrective_actions=payload.get("corrective_actions"),
        )

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
