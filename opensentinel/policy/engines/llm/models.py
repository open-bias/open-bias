"""
Type definitions for the LLM Policy Engine.

Contains all enums and dataclasses used by the LLM-based policy engine
for state classification, drift detection, constraint evaluation, and intervention.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

class ConfidenceTier(Enum):
    """Classification confidence tiers.
    
    - CONFIDENT: High confidence (>0.8), safe to proceed
    - UNCERTAIN: Medium confidence (0.5-0.8), may need attention
    - LOST: Low confidence (<0.5), agent may be off-track
    """
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    LOST = "lost"

class DriftLevel(Enum):
    """Drift severity levels.
    
    - NOMINAL: Normal operation (<0.3)
    - WARNING: Minor drift detected (0.3-0.6)
    - INTERVENTION: Significant drift (0.6-0.85)
    - CRITICAL: Severe drift requiring hard block (>0.85)
    """
    NOMINAL = "nominal"
    WARNING = "warning"
    INTERVENTION = "intervention"
    CRITICAL = "critical"

@dataclass
class LLMStateCandidate:
    """A candidate state from LLM classification."""
    state_id: str
    confidence: float
    reasoning: str

@dataclass
class LLMClassificationResult:
    """Result of LLM-based state classification."""
    candidates: list[LLMStateCandidate]
    best_state: str
    best_confidence: float
    tier: ConfidenceTier
    transition_legal: bool
    skip_violations: list[str] = field(default_factory=list)
    raw_llm_response: str | None = None

    @property
    def is_confident(self) -> bool:
        """Check if classification is confident."""
        return self.tier == ConfidenceTier.CONFIDENT

    @property
    def is_lost(self) -> bool:
        """Check if agent appears lost."""
        return self.tier == ConfidenceTier.LOST

@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: str
    to_state: str
    confidence: float
    tier: ConfidenceTier
    drift_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class DriftScores:
    """Composite drift scores from drift detection."""
    temporal: float  # 0.0-1.0, based on state sequence divergence
    semantic: float  # 0.0-1.0, based on message content drift
    composite: float  # Weighted combination
    level: DriftLevel
    anomaly_flags: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_scores(
        cls,
        temporal: float,
        semantic: float,
        temporal_weight: float = 0.55,
    ) -> "DriftScores":
        """Create DriftScores with computed composite and level."""
        semantic_weight = 1.0 - temporal_weight
        composite = temporal * temporal_weight + semantic * semantic_weight

        # Determine level based on composite score
        if composite < 0.3:
            level = DriftLevel.NOMINAL
        elif composite < 0.6:
            level = DriftLevel.WARNING
        elif composite < 0.85:
            level = DriftLevel.INTERVENTION
        else:
            level = DriftLevel.CRITICAL

        return cls(
            temporal=temporal,
            semantic=semantic,
            composite=composite,
            level=level,
        )

@dataclass
class ConstraintEvaluation:
    """Result of evaluating a single constraint."""
    constraint_id: str
    violated: bool
    confidence: float
    evidence: str
    severity: str  # "warning", "error", "critical"

@dataclass
class SessionContext:
    """Session state maintained across turns.

    Tracks the agent's progress through the workflow, including
    state history, drift scores, and constraint violations.
    """
    session_id: str
    workflow_name: str
    current_state: str
    state_history: list[StateTransition] = field(default_factory=list)
    drift_score: float = 0.0
    violation_buffer: list[ConstraintEvaluation] = field(default_factory=list)
    turn_count: int = 0
    
    # Ring buffer for recent confidences (max 3)
    recent_confidences: list[float] = field(default_factory=list)
    
    # Sliding window of recent turns (max default 10)
    turn_window: list[dict[str, Any]] = field(default_factory=list)
    
    # Evidence memory for constraints
    constraint_memory: dict[str, list[str]] = field(default_factory=dict)
    
    # Turn of last intervention (for cooldown)
    last_intervention_turn: int = -1

    # Drift score at last intervention (for self-correction detection)
    drift_at_last_intervention: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_state_sequence(self) -> list[str]:
        """Get chronological list of states visited."""
        states = [t.to_state for t in self.state_history]
        if not states and self.current_state:
            return [self.current_state]
        return states

    def add_confidence(self, confidence: float, max_size: int = 3) -> None:
        """Add a confidence score to the ring buffer."""
        self.recent_confidences.append(confidence)
        if len(self.recent_confidences) > max_size:
            self.recent_confidences.pop(0)

    def is_structurally_drifting(self) -> bool:
        """Check if all recent confidences are below threshold.
        
        Returns True if we have 3 consecutive uncertain classifications,
        indicating the agent may be structurally drifting from the workflow.
        """
        if len(self.recent_confidences) < 3:
            return False
        return all(c < 0.8 for c in self.recent_confidences)

    def add_turn(self, data: dict[str, Any], max_window: int = 10) -> None:
        """Add turn data to sliding window."""
        self.turn_window.append(data)
        if len(self.turn_window) > max_window:
            self.turn_window.pop(0)
        self.turn_count += 1
        self.last_updated_at = datetime.now(timezone.utc)

    def record_transition(
        self,
        from_state: str,
        to_state: str,
        confidence: float,
        tier: ConfidenceTier,
        drift_score: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a state transition."""
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            confidence=confidence,
            tier=tier,
            drift_score=drift_score,
            metadata=metadata or {},
        )
        self.state_history.append(transition)
        self.current_state = to_state
        self.drift_score = drift_score
        self.last_updated_at = datetime.now(timezone.utc)

    def clear_violation_buffer(self) -> None:
        """Clear the violation buffer after interventions."""
        self.violation_buffer.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "workflow_name": self.workflow_name,
            "current_state": self.current_state,
            "state_history": [
                {
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "confidence": t.confidence,
                    "tier": t.tier.value,
                    "drift_score": t.drift_score,
                    "timestamp": t.timestamp.isoformat(),
                }
                for t in self.state_history
            ],
            "drift_score": self.drift_score,
            "turn_count": self.turn_count,
            "is_structurally_drifting": self.is_structurally_drifting(),
            "created_at": self.created_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
        }
