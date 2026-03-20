"""
Workflow definition schema using Pydantic models.

Supports two config formats:

Simple (human-authored):
```yaml
name: customer-support
mode: guide

steps:
  - greet the customer
  - understand their issue
  - resolve and close

rules:
  - verify identity before any account action
  - never share internal system information

tools:
  verify identity: [verify_identity]
```

Internal (compiler output):
```yaml
name: customer-support
mode: guide

states:
  - name: greeting
    is_initial: true
    classification:
      patterns: ["hello", "hi", "welcome"]

transitions:
  - from_state: greeting
    to_state: identify_issue

constraints:
  - name: must_verify_identity
    type: precedence
    trigger: account_action
    target: identity_verified
    message: You must verify the customer's identity before account actions.
```
"""

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

class SimpleWorkflowConfig(BaseModel):
    """Human-authored workflow config with plain English steps and rules."""

    name: str
    mode: Literal["guide", "enforce"]
    steps: list[str]
    rules: list[str]
    tools: dict[str, list[str]] | None = None

class ClassificationHint(BaseModel):
    """
    Hints for classifying LLM output to a workflow state.

    Classification uses a priority cascade:
    1. tool_calls: Exact match on function names (highest confidence)
    2. patterns: Regex patterns to match in response
    3. exemplars: Example phrases for semantic similarity

    At least one classification method should be specified for non-initial states.
    """

    # Tool-based classification (highest priority, exact match)
    tool_calls: list[str] | None = None

    # Pattern-based classification (regex patterns)
    patterns: list[str] | None = None

    # Semantic similarity classification (embedding-based)
    exemplars: list[str] | None = None

    # Minimum confidence for semantic match (0.0 to 1.0)
    min_similarity: float = Field(default=0.7, ge=0.0, le=1.0)

class State(BaseModel):
    """
    A workflow state definition.

    States represent distinct phases in the agent's workflow.
    Each state can have classification hints to identify when
    the agent is in that state based on LLM output.
    """

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None

    # Classification configuration
    classification: ClassificationHint = Field(default_factory=ClassificationHint)

    # State metadata
    is_initial: bool = Field(default=False)
    is_terminal: bool = Field(default=False)
    is_error: bool = Field(default=False)

    # Allowed dwell time (for temporal constraints)
    max_duration_seconds: float | None = Field(default=None, ge=0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate state name is a valid identifier."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"State name must be alphanumeric (with _ or -): {v}")
        return v

class Transition(BaseModel):
    """
    A transition between workflow states.

    Transitions define valid state progressions.
    If no transitions are defined FROM a state, any state can follow.
    """

    from_state: str
    to_state: str

    # Priority for disambiguation (higher = preferred)
    priority: int = Field(default=0, ge=0)

    # Optional description
    description: str | None = None

class ConstraintType(str, Enum):
    """
    Temporal constraint operators.

    - PRECEDENCE: Target must occur before trigger
    - NEVER: Target state must never occur
    - EVENTUALLY: Must eventually reach target state
    - RESPONSE: If trigger occurs, target must eventually follow
    """

    PRECEDENCE = "precedence"
    NEVER = "never"
    EVENTUALLY = "eventually"
    RESPONSE = "response"

class Constraint(BaseModel):
    """
    Temporal constraint on workflow execution.

    Examples:
        Constraint(
            name="verify_first",
            type=ConstraintType.PRECEDENCE,
            trigger="account_action",
            target="identity_verified",
            message="You must verify identity before account actions.",
        )
    """

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    type: ConstraintType

    # Constraint parameters (interpretation depends on type)
    trigger: str | None = None  # For response/precedence
    target: str | None = None  # Target state

    # Intervention message when violated
    message: str = ""

    @model_validator(mode="after")
    def validate_constraint_params(self) -> "Constraint":
        """Validate that required parameters are present for each constraint type."""
        t = self.type

        if t == ConstraintType.EVENTUALLY and not self.target:
            raise ValueError("EVENTUALLY constraint requires 'target'")

        if t == ConstraintType.NEVER and not self.target:
            raise ValueError("NEVER constraint requires 'target'")

        if t == ConstraintType.RESPONSE and (not self.trigger or not self.target):
            raise ValueError("RESPONSE constraint requires 'trigger' and 'target'")

        if t == ConstraintType.PRECEDENCE and (not self.trigger or not self.target):
            raise ValueError("PRECEDENCE constraint requires 'trigger' and 'target'")

        return self

class WorkflowDefinition(BaseModel):
    """
    Complete workflow definition (internal compiler output format).

    A workflow defines:
    - States: The valid phases of the agent's operation
    - Transitions: Valid state progressions
    - Constraints: Temporal invariants that must hold
    """

    name: str = Field(..., min_length=1, max_length=100)
    mode: Literal["guide", "enforce"] = "guide"
    description: str | None = None

    # Workflow components
    states: list[State]
    transitions: list[Transition] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("states")
    @classmethod
    def validate_has_initial_state(cls, v: list[State]) -> list[State]:
        """Validate that at least one initial state exists."""
        if not any(s.is_initial for s in v):
            raise ValueError("Workflow must have at least one initial state")
        return v

    @model_validator(mode="after")
    def validate_references(self) -> "WorkflowDefinition":
        """Validate that all state references are valid."""
        state_names = {s.name for s in self.states}

        # Check transitions
        for t in self.transitions:
            if t.from_state not in state_names:
                raise ValueError(f"Transition references unknown state: {t.from_state}")
            if t.to_state not in state_names:
                raise ValueError(f"Transition references unknown state: {t.to_state}")

        # Check constraints
        for c in self.constraints:
            if c.trigger and c.trigger not in state_names:
                raise ValueError(
                    f"Constraint '{c.name}' references unknown trigger state: {c.trigger}"
                )
            # NEVER constraints can reference conceptual forbidden states
            # that don't exist in the workflow (they represent states that should never occur)
            if (
                c.target
                and c.target not in state_names
                and c.type != ConstraintType.NEVER
            ):
                raise ValueError(
                    f"Constraint '{c.name}' references unknown target state: {c.target}"
                )

        return self

    def get_state(self, name: str) -> State | None:
        """Get state by name."""
        for state in self.states:
            if state.name == name:
                return state
        return None

    def get_initial_states(self) -> list[State]:
        """Get all initial states."""
        return [s for s in self.states if s.is_initial]

    def get_terminal_states(self) -> list[State]:
        """Get all terminal states."""
        return [s for s in self.states if s.is_terminal]

    def get_transitions_from(self, state_name: str) -> list[Transition]:
        """Get all transitions from a state."""
        return [t for t in self.transitions if t.from_state == state_name]
