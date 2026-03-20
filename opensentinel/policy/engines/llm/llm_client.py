"""
LLM client wrapper for the LLM Policy Engine.

Provides a thin async wrapper around litellm.acompletion for
structured JSON responses used in state classification and
constraint evaluation.
"""

import json
import logging
import re
from typing import Any

from opensentinel.core.utils import extract_response_content

logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """Error from LLM client operations."""
    pass

class LLMClient:
    """Async LLM wrapper for JSON-structured responses.
    
    Wraps litellm.acompletion with:
    - JSON response parsing with markdown fence stripping
    - Token usage tracking
    - Configurable model, temperature, and timeout
    
    Example:
        client = LLMClient(model="gpt-4o-mini")
        result = await client.complete_json(
            system_prompt="Return JSON array of states",
            user_prompt="Classify: Hello world"
        )
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 10.0,
        max_retries: int = 2,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._total_tokens_used = 0

    @property
    def total_tokens_used(self) -> int:
        """Total tokens used across all calls."""
        return self._total_tokens_used

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        additional_messages: list | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Make an LLM call expecting a JSON response.
        
        Args:
            system_prompt: System message for the LLM
            user_prompt: User message/query
            additional_messages: Optional additional messages to include
            session_id: Optional session ID for tracing/logging
            
        Returns:
            Parsed JSON response as dict
            
        Raises:
            LLMClientError: If call fails or JSON parsing fails
        """
        # Lazy import to avoid import-time side effects
        import litellm
        
        if not self.model:
            raise LLMClientError("No LLM model configured. Please set a model in configuration or environment.")

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if additional_messages:
            messages.extend(additional_messages)
            
        messages.append({"role": "user", "content": user_prompt})
        
        # Prepare headers
        extra_headers = {}
        if session_id:
            extra_headers["x-sentinel-session-id"] = session_id
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    extra_headers=extra_headers,
                    metadata={"session_id": session_id} if session_id else {},
                )
                
                # Track token usage
                if hasattr(response, "usage") and response.usage:
                    self._total_tokens_used += getattr(
                        response.usage, "total_tokens", 0
                    )
                
                # Extract content
                content = extract_response_content(response)
                if not content:
                    raise LLMClientError("Empty response from LLM")
                
                # Parse JSON
                return self._parse_json(content)
                
            except LLMClientError:
                raise
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call attempt {attempt + 1}/{self.max_retries + 1} "
                    f"failed: {e}",
                    exc_info=True,
                )
                if attempt == self.max_retries:
                    break
        
        raise LLMClientError(
            f"LLM call failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def _parse_json(self, content: str) -> dict[str, Any]:
        """Parse JSON from content, handling markdown fences.
        
        Handles common patterns:
        - Plain JSON
        - ```json ... ```
        - ``` ... ```
        """
        # Strip leading/trailing whitespace
        content = content.strip()
        
        # Remove markdown code fences if present
        if content.startswith("```"):
            # Handle ```json or just ```
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nContent: {content[:500]}")
            raise LLMClientError(f"Invalid JSON response: {e}")

    def reset_token_count(self) -> None:
        """Reset the token usage counter."""
        self._total_tokens_used = 0
