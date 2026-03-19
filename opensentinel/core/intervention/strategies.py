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
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of intervention strategies."""

    SYSTEM_PROMPT_APPEND = "system_prompt_append"
    USER_MESSAGE_INJECT = "user_message_inject"
    RESPONSE_MODIFICATION = "response_modification"


@dataclass
class InterventionConfig:
    """Configuration for an intervention."""

    strategy_type: StrategyType
    message_template: str
    priority: int = 0  # Higher = more important
    max_applications: int = 3  # Informational; used by engine handlers for config display


class InterventionStrategy(ABC):
    """
    Base class for intervention strategies.

    Strategies modify LLM request data to guide the agent
    back to the expected workflow path.
    """

    @abstractmethod
    def apply(
        self,
        data: dict,
        config: InterventionConfig,
        context: Dict[str, Any],
    ) -> dict:
        """
        Apply intervention to request data.

        Args:
            data: LLM request data (messages, model, etc.)
            config: Intervention configuration
            context: Additional context (states, violations, etc.)

        Returns:
            Modified request data
        """
        pass

    @staticmethod
    @abstractmethod
    def merge(messages: List[Dict[str, Any]], value: str) -> List[Dict[str, Any]]:
        """
        Merge an intervention value into a messages list.

        Low-level operation used when applying deferred interventions
        from async checkers. Unlike apply(), takes a pre-formatted string
        and operates directly on messages.

        Args:
            messages: The messages list (will be copied, not mutated).
            value: The pre-formatted intervention text.

        Returns:
            New messages list with the intervention applied.
        """
        pass

    @staticmethod
    def format_message(template: str, context: Dict[str, Any]) -> str:
        """Format message template with context."""
        try:
            return template.format(**context)
        except KeyError as e:
            logger.warning(f"Missing context key in template: {e}")
            # Return template with unfilled placeholders rather than failing
            return template


class SystemPromptAppendStrategy(InterventionStrategy):
    """
    Append correction guidance to system message.

    This is the least disruptive strategy - it adds guidance
    to the system message without altering the conversation flow.

    Example output:
    ```
    System: You are a helpful assistant.

    [WORKFLOW GUIDANCE]: You must verify the customer's identity
    before performing any account actions.
    ```
    """

    @staticmethod
    def merge(messages: List[Dict[str, Any]], value: str) -> List[Dict[str, Any]]:
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

    def apply(
        self,
        data: dict,
        config: InterventionConfig,
        context: Dict[str, Any],
    ) -> dict:
        data = dict(data)
        correction = self.format_message(config.message_template, context)
        data["messages"] = self.merge(data.get("messages", []), correction)
        logger.debug("Applied system_prompt_append intervention")
        return data


class UserMessageInjectStrategy(InterventionStrategy):
    """
    Inject a user message with guidance.

    More visible than system prompt, appears as if the user
    is providing additional instructions.

    Example output:
    ```
    User: [System Note] Before proceeding, please verify
    the customer's identity as required by the workflow.
    ```
    """

    @staticmethod
    def merge(messages: List[Dict[str, Any]], value: str) -> List[Dict[str, Any]]:
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

    def apply(
        self,
        data: dict,
        config: InterventionConfig,
        context: Dict[str, Any],
    ) -> dict:
        data = dict(data)
        correction = self.format_message(config.message_template, context)
        data["messages"] = self.merge(data.get("messages", []), correction)
        logger.debug("Applied user_message_inject intervention")
        return data


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
        message: Optional[str] = None,
        modified_messages: Optional[List[Dict[str, Any]]] = None,
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
    def _get_response_content(response: Any) -> Optional[str]:
        """Extract text content from a response object."""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                content: Optional[str] = getattr(choice.message, "content", None)
                return content
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                result: Optional[str] = choices[0].get("message", {}).get("content")
                return result
        return None

    @staticmethod
    def _set_response_content(response: Any, content: str) -> None:
        """Set text content on a response object."""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                choice.message.content = content
                return
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices and "message" in choices[0]:
                choices[0]["message"]["content"] = content

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

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
