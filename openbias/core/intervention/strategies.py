"""
Intervention strategies for workflow correction.

Strategies define HOW to modify LLM requests when deviation is detected:

1. SYSTEM_PROMPT_APPEND: Add correction to system message (request)
   - Least disruptive, preserves conversation flow
   - Best for gentle guidance

2. USER_MESSAGE_INJECT: Add user message with guidance (request)
   - More visible to the model
   - Good for important corrections

"""

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

RESPONSE_CLEANUP_METADATA_KEY = "_openbias_response_cleanup"


class StrategyType(Enum):
    """Types of intervention strategies."""

    SYSTEM_PROMPT_APPEND = "system_prompt_append"
    USER_MESSAGE_INJECT = "user_message_inject"


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

        guidance = (
            "\n\n<system-reminder>\n"
            "Please address this message and continue with your tasks.\n\n"
            f"{value}\n"
            "</system-reminder>"
        )

        if system_idx is not None:
            existing = messages[system_idx].get("content", "")
            if isinstance(existing, list):
                messages[system_idx]["content"] = existing + [
                    {"type": "text", "text": guidance}
                ]
            else:
                messages[system_idx]["content"] = (existing or "") + guidance
        else:
            messages.insert(0, {
                "role": "system",
                "content": (
                    "<system-reminder>\n"
                    "Please address this message and continue with your tasks.\n\n"
                    f"{value}\n"
                    "</system-reminder>"
                ),
            })

        return messages

    @staticmethod
    def cleanup_rules() -> list[str]:
        """System-appended reminders are not scrubbed from model output."""
        return []


class UserMessageInjectStrategy:
    """
    Inject a user message with guidance.

    More visible than system prompt, appears as if the user
    is providing additional instructions.
    """

    @staticmethod
    def merge(messages: list[dict[str, Any]], value: str) -> list[dict[str, Any]]:
        messages = [dict(m) for m in messages]
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

    @staticmethod
    def cleanup_rules() -> list[str]:
        """Markers that may leak when guidance is injected as a user note."""
        return [
            "[REPAIR-INSTRUCTION]",
            "[END-REPAIR-INSTRUCTION]",
            "[System Note]",
        ]


class WorkflowViolationError(Exception):
    """Exception raised when a workflow violation blocks a request."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}
