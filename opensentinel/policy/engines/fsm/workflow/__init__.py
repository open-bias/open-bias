"""
FSM workflow module.

Contains all FSM-specific workflow components:
- Schema: WorkflowDefinition, State, Transition, Constraint, etc.
- Parser: WorkflowParser for YAML/JSON loading
- StateMachine: WorkflowStateMachine for state tracking
- Constraints: ConstraintEvaluator for LTL-lite verification
"""

from opensentinel.policy.engines.fsm.workflow.constraints import (
    ConstraintEvaluator,
    ConstraintViolation,
    EvaluationResult,
    get_decision,
)
from opensentinel.policy.engines.fsm.workflow.parser import (
    WorkflowParser,
    WorkflowRegistry,
)
from opensentinel.policy.engines.fsm.workflow.schema import (
    ClassificationHint,
    Constraint,
    ConstraintType,
    SimpleWorkflowConfig,
    State,
    Transition,
    WorkflowDefinition,
)
from opensentinel.policy.engines.fsm.workflow.state_machine import (
    SessionState,
    StateHistoryEntry,
    TransitionResult,
    WorkflowStateMachine,
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
    "get_decision",
]
