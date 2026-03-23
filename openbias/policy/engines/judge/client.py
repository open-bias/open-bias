"""
Judge LLM client for the Judge Policy Engine.

Composes LLMClient instances to manage one or more judge models.
Supports single-model and parallel multi-model calls.
"""

import asyncio
import logging
from typing import Any

from openbias.policy.engines.llm.llm_client import LLMClient, LLMClientError

logger = logging.getLogger(__name__)

class JudgeClient:
    """Manages LLMClient instances for judge model calls.

    Each configured judge model gets its own LLMClient. Supports
    calling a single judge or all judges in parallel.
    """

    def __init__(self) -> None:
        self._clients: dict[str, LLMClient] = {}

    def add_model(
        self,
        name: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 15.0,
        max_retries: int = 2,
    ) -> None:
        """Register a judge model.

        Args:
            name: Logical name for this judge (e.g., "primary").
            model: Model identifier (e.g., "gpt-4o-mini").
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts.
        """
        self._clients[name] = LLMClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        logger.debug(f"Judge model registered: {name} ({model})")

    @property
    def timeout(self) -> float:
        """Maximum wall-clock time a single evaluation could take, including retries."""
        worst_cases = [
            client.timeout * (client.max_retries + 1)
            for client in self._clients.values()
        ]
        return max(worst_cases, default=0.0)

    @property
    def model_names(self) -> list[str]:
        """List of registered model names."""
        return list(self._clients.keys())

    @property
    def primary_model(self) -> str | None:
        """First registered model name, or None."""
        return self.model_names[0] if self._clients else None

    def get_model_id(self, name: str) -> str:
        """Get the underlying model identifier for a named judge."""
        if name not in self._clients:
            raise ValueError(f"Unknown judge model: {name}")
        return self._clients[name].model

    async def call_judge(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Call a single judge model.

        Args:
            model_name: Name of the registered judge model.
            system_prompt: System prompt for the judge.
            user_prompt: User prompt with content to evaluate.
            session_id: Optional session ID for tracing/logging.

        Returns:
            Parsed JSON response from the judge.

        Raises:
            ValueError: If model_name is not registered.
            LLMClientError: If the LLM call fails.
        """
        if model_name not in self._clients:
            raise ValueError(f"Unknown judge model: {model_name}")

        client = self._clients[model_name]
        return await client.complete_json(
            system_prompt, user_prompt, session_id=session_id
        )

    async def call_all_judges(
        self,
        system_prompt: str,
        user_prompt: str,
        session_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Call all registered judge models in parallel.

        Failed models are logged and excluded from results (fail-open).

        Args:
            system_prompt: System prompt for all judges.
            user_prompt: User prompt with content to evaluate.
            session_id: Optional session ID for tracing/logging.

        Returns:
            Dict mapping model name to parsed JSON response.
            Only includes models that succeeded.
        """
        if not self._clients:
            return {}

        tasks = {
            name: client.complete_json(
                system_prompt, user_prompt, session_id=session_id
            )
            for name, client in self._clients.items()
        }

        results = await asyncio.gather(
            *tasks.values(), return_exceptions=True
        )

        responses: dict[str, dict[str, Any]] = {}
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Judge model '{name}' failed: {result}")
            else:
                responses[name] = result

        return responses

    def get_total_tokens(self) -> int:
        """Get total tokens used across all judge models."""
        return sum(c.total_tokens_used for c in self._clients.values())

    def get_tokens_for_model(self, name: str) -> int:
        """Get tokens used by a specific judge model."""
        if name not in self._clients:
            return 0
        return self._clients[name].total_tokens_used
