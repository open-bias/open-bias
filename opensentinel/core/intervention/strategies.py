"""
Intervention strategies for workflow correction.

Strategies define HOW to modify LLM requests or responses when deviation is detected:

1. SYSTEM_PROMPT_APPEND: Add correction to system message (request)
   - Least disruptive, preserves conversation flow
   - Best for gentle guidance

2. USER_MESSAGE_INJECT: Add user message with guidance (request)
   - More visible to the model
   - Good for important corrections

3. RESPONSE_MODIFICATION: Modify the current LLM response (response)
   - Strips tool calls, replaces content, or appends warnings
   - Used by sync POST_CALL checkers for real-time enforcement

4. HARD_BLOCK: Raise WorkflowViolationError to halt execution (request)
   - Most aggressive — LLM call never happens
   - Used when policy requires immediate rejection
"""

import logging
from typing import Any
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of intervention strategies."""

    SYSTEM_PROMPT_APPEND = "system_prompt_append"
    USER_MESSAGE_INJECT = "user_message_inject"
    RESPONSE_MODIFICATION = "response_modification"
    HARD_BLOCK = "hard_block"


def format_message(template: str, context: dict[str, Any]) -> str:
    """Format message template with context."""
    try:
        return template.format(**context)
    except KeyError as e:
        logger.warning(f"Missing context key in template: {e}")
        # Return template with unfilled placeholders rather than failing
        return template


class SystemPromptAppendStrategy:
    """
    Append correction guidance to system message.

    This is the least disruptive strategy - it adds guidance
    to the system message without altering the conversation flow.
    """

    @staticmethod
    def merge(messages: list[dict[str, Any]], value: str) -> list[dict[str, Any]]:
        messages = [dict(m) for m in messages]

        system_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_idx = i
                break

        guidance = f"\n\n[WORKFLOW GUIDANCE]: {value}"

        if system_idx is not None:
            messages[system_idx]["content"] = (
                messages[system_idx].get("content", "") + guidance
            )
        else:
            messages.insert(0, {
                "role": "system",
                "content": f"[WORKFLOW GUIDANCE]: {value}",
            })

        return messages


class UserMessageInjectStrategy:
    """
    Inject a user message with guidance.

    More visible than system prompt, appears as if the user
    is providing additional instructions.
    """

    @staticmethod
    def merge(messages: list[dict[str, Any]], value: str) -> list[dict[str, Any]]:
        messages = list(messages)
        guidance = {"role": "user", "content": f"[System Note]: {value}"}

        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is not None:
            messages.insert(last_user_idx + 1, guidance)
        else:
            messages.append(guidance)

        return messages


class ResponseModificationStrategy:
    """
    Modify an LLM response after POST_CALL evaluation.

    Unlike request-modifying strategies (SystemPromptAppend, UserMessageInject),
    this operates on the *current response* — stripping tool calls, replacing
    content, or appending warnings.

    Works with LiteLLM response objects (ModelResponse / dict).
    """

    @staticmethod
    def apply_to_response(
        response: Any,
        message: str | None = None,
        modified_messages: list[dict[str, Any]] | None = None,
    ) -> Any:
        """
        Apply intervention to an LLM response.

        Args:
            response: The LLM response object (LiteLLM ModelResponse or dict).
            message: Warning/guidance text to append to the response content.
            modified_messages: If provided, replace the response content entirely.

        Returns:
            The modified response object.
        """
        if modified_messages is not None:
            # Full replacement — use the first modified message's content
            replacement_content = ""
            for msg in modified_messages:
                if msg.get("role") == "assistant":
                    replacement_content = msg.get("content", "")
                    break
            if not replacement_content and modified_messages:
                replacement_content = modified_messages[0].get("content", "")
            ResponseModificationStrategy._set_response_content(
                response, replacement_content
            )
            # Strip tool calls when replacing content
            ResponseModificationStrategy._strip_tool_calls(response)
            logger.debug("Applied response modification: full replacement")
        elif message:
            # Append warning to existing content
            current = ResponseModificationStrategy._get_response_content(response)
            warning = f"\n\n[POLICY WARNING]: {message}"
            ResponseModificationStrategy._set_response_content(
                response, (current or "") + warning
            )
            logger.debug("Applied response modification: appended warning")

        return response

    @staticmethod
    def _get_response_content(response: Any) -> str | None:
        """Extract text content from a response object."""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                content: str | None = getattr(choice.message, "content", None)
                return content
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                result: str | None = choices[0].get("message", {}).get("content")
                return result
        return None

    @staticmethod
    def _set_response_content(response: Any, content: str) -> bool:
        """Set text content on a response object. Returns True if content was set."""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                choice.message.content = content
                return True
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices and "message" in choices[0]:
                choices[0]["message"]["content"] = content
                return True
        logger.warning(
            "Response has no choices — intervention message could not be applied"
        )
        return False

    @staticmethod
    def _strip_tool_calls(response: Any) -> None:
        """Remove tool calls from a response."""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                if hasattr(choice.message, "tool_calls"):
                    choice.message.tool_calls = None
                return
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices and "message" in choices[0]:
                choices[0]["message"].pop("tool_calls", None)


class WorkflowViolationError(Exception):
    """Exception raised when a workflow violation blocks a request."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}


class HardBlockStrategy:
    """
    Immediately block the request by raising WorkflowViolationError.

    Unlike other strategies that modify requests or responses,
    this strategy halts execution entirely — the LLM call never happens.
    """

    @staticmethod
    def apply(message: str) -> None:
        """Raise WorkflowViolationError to block the request."""
        raise WorkflowViolationError(message)
