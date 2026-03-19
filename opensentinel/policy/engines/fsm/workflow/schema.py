"""
Workflow definition schema using Pydantic models.

Supports:
- State definitions with classification hints
- Transitions with guards
- LTL-lite constraints (eventually, always, never, precedence, response)
- Intervention strategies

Example YAML:
```yaml
name: customer-support
version: "1.0"

states:
  - name: greeting
    is_initial: true
    classification:
      patterns: ["hello", "hi", "welcome"]

  - name: identify_issue
    classification:
      tool_calls: ["search_kb"]
      exemplars:
        - "Let me look that up"
        - "I'll search our documentation"

  - name: resolution
    is_terminal: true

transitions:
  - from_state: greeting
    to_state: identify_issue

constraints:
  - name: must_verify_identity
    type: precedence
    trigger: account_action
    target: identity_verified
    intervention: prompt_identity_verification

interventions:
  prompt_identity_verification: |
    You must verify the customer's identity before account actions.
```
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


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
    tool_calls: Optional[List[str]] = None

    # Pattern-based classification (regex patterns)
    patterns: Optional[List[str]] = None

    # Semantic similarity classification (embedding-based)
    exemplars: Optional[List[str]] = None

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
    description: Optional[str] = None

    # Classification configuration
    classification: ClassificationHint = Field(default_factory=ClassificationHint)

    # State metadata
    is_initial: bool = Field(default=False)
    is_terminal: bool = Field(default=False)
    is_error: bool = Field(default=False)

    # Allowed dwell time (for temporal constraints)
    max_duration_seconds: Optional[float] = Field(default=None, ge=0)

    # Support inline transitions
    transitions: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate state name is a valid identifier."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"State name must be alphanumeric (with _ or -): {v}")
        return v


class TransitionGuard(BaseModel):
    """
    Conditions that must be true for a transition to occur.

    Guards can specify:
    - A Python expression to evaluate
    - Required metadata fields
    """

    # Python expression evaluated with context
    # Example: "confidence > 0.8"
    expression: Optional[str] = None

    # Required metadata fields and values
    # Example: {"user_verified": True}
    required_metadata: Optional[Dict[str, Any]] = None


class Transition(BaseModel):
    """
    A transition between workflow states.

    Transitions define valid state progressions.
    If no transitions are defined FROM a state, any state can follow.
    """

    from_state: str
    to_state: str

    # Optional guard conditions
    guard: Optional[TransitionGuard] = None

    # Priority for disambiguation (higher = preferred)
    priority: int = Field(default=0, ge=0)

    # Optional description
    description: Optional[str] = None


class ConstraintType(str, Enum):
    """
    LTL-lite constraint operators.

    Simplified temporal logic for practical workflow constraints:

    - EVENTUALLY (F): Must eventually reach target state
    - ALWAYS (G): Condition must always hold
    - NEVER (G!): Target state must never occur
    - UNTIL (U): Stay in trigger until target reached
    - NEXT (X): Immediate next state requirement
    - RESPONSE: If trigger occurs, target must eventually follow
    - PRECEDENCE: Target cannot occur before trigger
    """

    EVENTUALLY = "eventually"
    ALWAYS = "always"
    NEVER = "never"
    UNTIL = "until"
    NEXT = "next"
    RESPONSE = "response"
    PRECEDENCE = "precedence"


class Constraint(BaseModel):
    """
    LTL-lite temporal constraint.

    Constraints define invariants that must hold during workflow execution.
    When violated, the specified intervention is triggered.

    Examples:
        # Must verify identity before account actions
        Constraint(
            name="verify_first",
            type=ConstraintType.PRECEDENCE,
            trigger="account_action",
            target="identity_verified",
            intervention="prompt_verify"
        )

        # Must eventually reach resolution
        Constraint(
            name="must_resolve",
            type=ConstraintType.EVENTUALLY,
            target="resolution"
        )

        # Never share credentials
        Constraint(
            name="no_credentials",
            type=ConstraintType.NEVER,
            target="share_credentials",
            severity="critical"
        )
    """

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    type: ConstraintType

    # Constraint parameters (interpretation depends on type)
    trigger: Optional[str] = None  # For response/precedence/until
    target: Optional[str] = None  # Target state
    condition: Optional[str] = None  # Boolean expression for ALWAYS

    # Violation handling
    severity: Literal["warning", "error", "critical"] = "error"
    intervention: Optional[str] = None  # Intervention strategy name

    @model_validator(mode="after")
    def validate_constraint_params(self):
        """Validate that required parameters are present for each constraint type."""
        t = self.type

        if t == ConstraintType.EVENTUALLY and not self.target:
            raise ValueError("EVENTUALLY constraint requires 'target'")

        if t == ConstraintType.ALWAYS and not self.condition:
            raise ValueError("ALWAYS constraint requires 'condition'")

        if t == ConstraintType.NEVER and not self.target:
            raise ValueError("NEVER constraint requires 'target'")

        if t == ConstraintType.UNTIL and (not self.trigger or not self.target):
            raise ValueError("UNTIL constraint requires 'trigger' and 'target'")

        if t == ConstraintType.NEXT and not self.target:
            raise ValueError("NEXT constraint requires 'target'")

        if t == ConstraintType.RESPONSE and (not self.trigger or not self.target):
            raise ValueError("RESPONSE constraint requires 'trigger' and 'target'")

        if t == ConstraintType.PRECEDENCE and (not self.trigger or not self.target):
            raise ValueError("PRECEDENCE constraint requires 'trigger' and 'target'")

        return self


class WorkflowDefinition(BaseModel):
    """
    Complete workflow definition.

    A workflow defines:
    - States: The valid phases of the agent's operation
    - Transitions: Valid state progressions
    - Constraints: Temporal invariants that must hold
    - Interventions: Correction strategies when constraints violated
    """

    name: str = Field(..., min_length=1, max_length=100)
    version: str = "1.0"
    description: Optional[str] = None

    # Workflow components
    states: List[State]
    transitions: List[Transition] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)

    # Intervention templates
    # Maps intervention name -> prompt template
    interventions: Dict[str, str] = Field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def collect_state_transitions(cls, data: Any) -> Any:
        """Collect transitions defined inside states into the top-level transitions list."""
        if not isinstance(data, dict):
            return data

        states = data.get("states", [])
        if not isinstance(states, list):
            return data

        transitions = data.get("transitions", [])
        if not isinstance(transitions, list):
            transitions = []

        for state in states:
            if not isinstance(state, dict):
                continue
            
            state_name = state.get("name")
            state_transitions = state.get("transitions", [])
            
            if not state_name or not isinstance(state_transitions, list):
                continue
                
            for trans in state_transitions:
                if not isinstance(trans, dict):
                    continue
                
                # If from_state is not specified, use the current state
                if "from_state" not in trans:
                    trans["from_state"] = state_name
                    
                transitions.append(trans)
            
            # Remove inline transitions after collecting them
            # state.pop("transitions", None) # Keep them for now to avoid issues with State model validation

        data["transitions"] = transitions
        return data

    @field_validator("states")
    @classmethod
    def validate_has_initial_state(cls, v: List[State]) -> List[State]:
        """Validate that at least one initial state exists."""
        if not any(s.is_initial for s in v):
            raise ValueError("Workflow must have at least one initial state")
        return v

    @model_validator(mode="after")
    def validate_references(self):
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

            # Check intervention reference
            if c.intervention and c.intervention not in self.interventions:
                raise ValueError(
                    f"Constraint '{c.name}' references unknown intervention: {c.intervention}"
                )

        return self

    def get_state(self, name: str) -> Optional[State]:
        """Get state by name."""
        for state in self.states:
            if state.name == name:
                return state
        return None

    def get_initial_states(self) -> List[State]:
        """Get all initial states."""
        return [s for s in self.states if s.is_initial]

    def get_terminal_states(self) -> List[State]:
        """Get all terminal states."""
        return [s for s in self.states if s.is_terminal]

    def get_transitions_from(self, state_name: str) -> List[Transition]:
        """Get all transitions from a state."""
        return [t for t in self.transitions if t.from_state == state_name]
