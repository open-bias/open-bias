"""
Generic intervention strategies for policy engines.

This module provides the intervention infrastructure that can be used
by any policy engine (FSM, NeMo, custom engines, etc.).

Strategies define HOW to modify LLM requests when policy violations
are detected:

1. SYSTEM_PROMPT_APPEND: Add correction to system message
2. USER_MESSAGE_INJECT: Add user message with guidance
"""

from openbias.core.intervention.strategies import (
    ResponseModificationStrategy,
    StrategyType,
    SystemPromptAppendStrategy,
    UserMessageInjectStrategy,
    WorkflowViolationError,
    format_message,
)

__all__ = [
    "StrategyType",
    "format_message",
    "SystemPromptAppendStrategy",
    "UserMessageInjectStrategy",
    "ResponseModificationStrategy",
    "WorkflowViolationError",
]
