"""
Base compiler class with shared LLM interaction logic.

Provides common functionality for rules compilers:
- LLM client management
- Prompt construction
- Response parsing utilities
"""

import json
import logging
from abc import abstractmethod
from typing import Any

from openbias.policy.compiler.protocol import PolicyCompiler, CompilationResult

logger = logging.getLogger(__name__)

# Default system prompt for rules compilation
DEFAULT_COMPILER_SYSTEM_PROMPT = """You are a rules compiler that converts normalized rules text into structured configuration.

Your task is to:
1. Extract states/phases mentioned in the rules
2. Identify temporal constraints (must happen before, never, eventually, etc.)
3. Determine classification hints (tool calls, patterns, exemplars)
4. Generate intervention messages for violations

Respond ONLY with valid JSON matching the requested schema. Do not include explanations or markdown formatting."""

class LLMPolicyCompiler(PolicyCompiler):
    """
    Base class for compilers that use an LLM for parsing.

    Provides common LLM interaction utilities. Subclasses implement
    engine-specific prompt construction and response parsing.

    Attributes:
        model: LLM model identifier (default: gpt-4o-mini)
        temperature: Generation temperature (default: 0.2 for determinism)
        system_prompt: Base system prompt for compilation
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        system_prompt: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize the LLM compiler.

        Args:
            model: LLM model to use for compilation
            temperature: Generation temperature (lower = more deterministic)
            system_prompt: Custom system prompt (uses default if not provided)
            api_key: API key for LLM provider (uses env var if not provided)
            base_url: Base URL for LLM API (for custom endpoints)
        """
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt or DEFAULT_COMPILER_SYSTEM_PROMPT
        self._api_key = api_key
        self._base_url = base_url
    async def _call_llm(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """
        Call the LLM with a prompt using litellm.

        Args:
            user_prompt: User message content
            system_prompt: Override system prompt (uses default if None)
            response_format: Optional response format spec (e.g., {"type": "json_object"})

        Returns:
            LLM response content

        Raises:
            Exception: If LLM call fails
        """
        import litellm

        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if response_format:
            kwargs["response_format"] = response_format

        logger.debug(f"Calling LLM with model={self.model}, prompt length={len(user_prompt)}")

        response = await litellm.acompletion(**kwargs)

        content = response.choices[0].message.content or ""
        logger.debug(f"LLM response length={len(content)}")

        return content

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """
        Parse JSON from LLM response.

        Handles common issues like markdown code blocks.

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON as dict

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        # Strip markdown code blocks if present
        content = response.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        return json.loads(content.strip())

    @abstractmethod
    def _build_compilation_prompt(
        self,
        rules_text: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Build the prompt for rules compilation.

        Subclasses implement this to create engine-specific prompts
        with appropriate schema examples.

        Args:
            rules_text: User's normalized rules text
            context: Optional compilation context

        Returns:
            Complete user prompt for LLM
        """
        ...

    @abstractmethod
    def _parse_compilation_response(
        self,
        response: dict[str, Any],
        rules_text: str,
    ) -> CompilationResult:
        """
        Parse LLM response into CompilationResult.

        Subclasses implement this to convert the JSON response
        into engine-specific configuration.

        Args:
            response: Parsed JSON from LLM
            rules_text: Original normalized rules text (for metadata)

        Returns:
            CompilationResult with engine config
        """
        ...

    async def compile(
        self,
        rules_text: str,
        context: dict[str, Any] | None = None,
    ) -> CompilationResult:
        """
        Compile normalized rules text to engine config.

        Uses LLM to parse the rules and convert to engine-specific
        configuration.

        Args:
            rules_text: Normalized rules text
            context: Optional compilation hints

        Returns:
            CompilationResult with engine-specific config
        """
        try:
            # Build prompt
            prompt = self._build_compilation_prompt(rules_text, context)

            # Call LLM
            response = await self._call_llm(
                user_prompt=prompt,
                response_format={"type": "json_object"},
            )

            # Parse JSON
            try:
                parsed = self._parse_json_response(response)
            except json.JSONDecodeError as e:
                return CompilationResult.failure(
                    errors=[f"Failed to parse LLM response as JSON: {e}"],
                )

            # Convert to engine config
            result = self._parse_compilation_response(parsed, rules_text)

            # Validate
            validation_errors = self.validate_result(result)
            if validation_errors:
                result.errors.extend(validation_errors)
                result.success = False

            return result

        except ImportError as e:
            return CompilationResult.failure(errors=[str(e)])
        except Exception as e:
            logger.exception("Compilation failed")
            return CompilationResult.failure(
                errors=[f"Compilation failed: {type(e).__name__}: {e}"]
            )
