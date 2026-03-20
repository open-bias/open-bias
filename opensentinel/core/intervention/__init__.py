"""
Generic intervention strategies for policy engines.

This module provides the intervention infrastructure that can be used
by any policy engine (FSM, NeMo, custom engines, etc.).

Strategies define HOW to modify LLM requests when policy violations
are detected:

1. SYSTEM_PROMPT_APPEND: Add correction to system message
2. USER_MESSAGE_INJECT: Add user message with guidance
"""

from opensentinel.core.intervention.strategies import (
    StrategyType,
    InterventionConfig,
    InterventionStrategy,
    SystemPromptAppendStrategy,
    UserMessageInjectStrategy,
    ResponseModificationStrategy,
    WorkflowViolationError,
)

__all__ = [
    "StrategyType",
    "InterventionConfig",
    "InterventionStrategy",
    "SystemPromptAppendStrategy",
    "UserMessageInjectStrategy",
    "ResponseModificationStrategy",
    "WorkflowViolationError",
]
