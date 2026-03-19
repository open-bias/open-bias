"""
FSM workflow module.

Contains all FSM-specific workflow components:
- Schema: WorkflowDefinition, State, Transition, Constraint, etc.
- Parser: WorkflowParser for YAML/JSON loading
- StateMachine: WorkflowStateMachine for state tracking
- Constraints: ConstraintEvaluator for LTL-lite verification
"""

from opensentinel.policy.engines.fsm.workflow.schema import (
    SimpleWorkflowConfig,
    ClassificationHint,
    State,
    Transition,
    ConstraintType,
    Constraint,
    WorkflowDefinition,
)
from opensentinel.policy.engines.fsm.workflow.parser import (
    WorkflowParser,
    WorkflowRegistry,
)
from opensentinel.policy.engines.fsm.workflow.state_machine import (
    TransitionResult,
    StateHistoryEntry,
    SessionState,
    WorkflowStateMachine,
)
from opensentinel.policy.engines.fsm.workflow.constraints import (
    EvaluationResult,
    ConstraintViolation,
    ConstraintEvaluator,
)

__all__ = [
    # Schema
    "SimpleWorkflowConfig",
    "ClassificationHint",
    "State",
    "Transition",
    "ConstraintType",
    "Constraint",
    "WorkflowDefinition",
    # Parser
    "WorkflowParser",
    "WorkflowRegistry",
    # State Machine
    "TransitionResult",
    "StateHistoryEntry",
    "SessionState",
    "WorkflowStateMachine",
    # Constraints
    "EvaluationResult",
    "ConstraintViolation",
    "ConstraintEvaluator",
]
