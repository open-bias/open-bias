"""
Finite State Machine policy engine.

Wraps Open Sentinel's existing workflow/state machine implementation
as a PolicyEngine for use with the pluggable policy infrastructure.

This module now contains all FSM-specific components:
- Workflow: Schema, Parser, StateMachine, Constraints
- Monitor: StateClassifier, WorkflowTracker
- Intervention: InterventionHandler
"""

# Monitor components
from opensentinel.policy.engines.fsm.classifier import (
    StateClassifier,
)

# Compiler
from opensentinel.policy.engines.fsm.compiler import FSMCompiler, compile_workflow
from opensentinel.policy.engines.fsm.engine import FSMPolicyEngine

# Intervention components
from opensentinel.policy.engines.fsm.intervention import (
    InterventionHandler,
)

# Workflow components
from opensentinel.policy.engines.fsm.workflow import (
    ClassificationHint,
    Constraint,
    ConstraintEvaluator,
    ConstraintType,
    ConstraintViolation,
    # Constraints
    EvaluationResult,
    SessionState,
    # Schema
    SimpleWorkflowConfig,
    State,
    StateHistoryEntry,
    Transition,
    # State machine
    TransitionResult,
    WorkflowDefinition,
    # Parser
    WorkflowParser,
    WorkflowRegistry,
    WorkflowStateMachine,
)

__all__ = [
    # Engine
    "FSMPolicyEngine",
    # Workflow - Schema
    "SimpleWorkflowConfig",
    "ClassificationHint",
    "State",
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
    "compile_workflow",
]

