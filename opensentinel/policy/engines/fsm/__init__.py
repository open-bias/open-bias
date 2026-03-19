"""
Finite State Machine policy engine.

Wraps Open Sentinel's existing workflow/state machine implementation
as a PolicyEngine for use with the pluggable policy infrastructure.

This module now contains all FSM-specific components:
- Workflow: Schema, Parser, StateMachine, Constraints
- Monitor: StateClassifier, WorkflowTracker
- Intervention: InterventionHandler
"""

from opensentinel.policy.engines.fsm.engine import FSMPolicyEngine

# Workflow components
from opensentinel.policy.engines.fsm.workflow import (
    # Schema
    ClassificationHint,
    State,
    TransitionGuard,
    Transition,
    ConstraintType,
    Constraint,
    WorkflowDefinition,
    # Parser
    WorkflowParser,
    WorkflowRegistry,
    # State machine
    TransitionResult,
    StateHistoryEntry,
    SessionState,
    WorkflowStateMachine,
    # Constraints
    EvaluationResult,
    ConstraintViolation,
    ConstraintEvaluator,
)

# Monitor components
from opensentinel.policy.engines.fsm.classifier import (

    StateClassifier,
)


# Intervention components
from opensentinel.policy.engines.fsm.intervention import (
    InterventionHandler,
)

# Compiler
from opensentinel.policy.engines.fsm.compiler import FSMCompiler

__all__ = [
    # Engine
    "FSMPolicyEngine",
    # Workflow - Schema
    "ClassificationHint",
    "State",
    "TransitionGuard",
    "Transition",
    "ConstraintType",
    "Constraint",
    "WorkflowDefinition",
    # Workflow - Parser
    "WorkflowParser",
    "WorkflowRegistry",
    # Workflow - State machine
    "TransitionResult",
    "StateHistoryEntry",
    "SessionState",
    "WorkflowStateMachine",
    # Workflow - Constraints
    "EvaluationResult",
    "ConstraintViolation",
    "ConstraintEvaluator",
    # Monitor
    "StateClassifier",

    # Intervention
    "InterventionHandler",
    # Compiler
    "FSMCompiler",
]

