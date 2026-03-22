"""
Drift detector for the LLM Policy Engine.

Detects behavioral drift by computing:
1. Temporal drift: Divergence from expected state sequence
2. Semantic drift: Content drift from on-policy exemplars
3. Composite score: Weighted combination with configurable weights
"""

import logging

from opensentinel.policy.engines.fsm.workflow.schema import WorkflowDefinition
from opensentinel.policy.engines.llm.models import (
    DriftLevel,
    DriftScores,
    SessionContext,
)

logger = logging.getLogger(__name__)

class DriftDetector:
    """Detects behavioral drift in agent conversations.
    
    Combines temporal drift (state sequence divergence) and semantic
    drift (message content similarity) to produce a composite drift score.
    
    Example:
        detector = DriftDetector(workflow)
        drift = detector.compute_drift(
            session,
            "Let me help you with that",
            tool_calls=["search_kb"],
            expected_tool_calls=["verify_identity"]
        )
        if drift.level == DriftLevel.INTERVENTION:
            # Apply intervention
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        temporal_weight: float = 0.55,
        decay_alpha: float = 0.3,
    ):
        self.workflow = workflow
        self.temporal_weight = temporal_weight
        self.decay_alpha = decay_alpha
        
        # Expected state sequence (computed from workflow)
        self._expected_sequence: list[str] | None = None
        
        # On-policy centroid embedding (lazy loaded)
        self._centroid = None
        self._centroid_computed: bool = False
        self._model = None

    @property
    def expected_sequence(self) -> list[str]:
        """Compute expected state sequence from workflow transitions.
        
        Uses greedy traversal from initial state following highest-priority
        transitions.
        """
        if self._expected_sequence is not None:
            return self._expected_sequence
        
        # Find initial state
        initial_states = self.workflow.get_initial_states()
        if not initial_states:
            self._expected_sequence = []
            return self._expected_sequence
        
        sequence = [initial_states[0].name]
        visited: set[str] = {initial_states[0].name}
        current = initial_states[0].name
        
        # Greedy traversal by priority
        while True:
            transitions = self.workflow.get_transitions_from(current)
            if not transitions:
                break
            
            # Sort by priority (descending)
            transitions = sorted(transitions, key=lambda t: t.priority, reverse=True)
            
            # Find first unvisited target
            next_state = None
            for t in transitions:
                if t.to_state not in visited:
                    next_state = t.to_state
                    break
            
            if not next_state:
                break
            
            sequence.append(next_state)
            visited.add(next_state)
            current = next_state
        
        self._expected_sequence = sequence
        logger.debug(f"Expected sequence: {sequence}")
        return self._expected_sequence

    def _get_centroid(self):
        """Lazy-load on-policy centroid embedding.
        
        Computes average embedding of all state descriptions and exemplars.
        """
        if self._centroid_computed:
            return self._centroid

        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            
            if self._model is None:
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Collect all on-policy text
            texts = []
            for state in self.workflow.states:
                if state.description:
                    texts.append(state.description)
                if state.classification.exemplars:
                    texts.extend(state.classification.exemplars)
            
            if not texts:
                self._centroid = None
                return None
            
            # Compute centroid
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            self._centroid = np.mean(embeddings, axis=0)
            logger.debug(f"Computed on-policy centroid from {len(texts)} texts")
            
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, semantic drift disabled"
            )
            self._centroid = None
        except Exception as e:
            logger.error(f"Failed to compute centroid: {e}")
            self._centroid = None

        self._centroid_computed = True
        return self._centroid

    def compute_temporal_drift(
        self,
        expected: list[str],
        actual: list[str],
    ) -> float:
        """Compute temporal drift using weighted Levenshtein distance.
        
        Uses exponential decay to weight recent states more heavily.
        
        Args:
            expected: Expected state sequence
            actual: Actual state sequence
            
        Returns:
            Normalized drift score (0.0-1.0)
        """
        if not expected and not actual:
            return 0.0
        if not actual:
            return 0.0
        if not expected:
            return 0.5
        
        # Pad shorter sequence
        max_len = max(len(expected), len(actual))
        expected = list(expected) + [""] * (max_len - len(expected))
        actual = list(actual) + [""] * (max_len - len(actual))
        
        # Compute weighted distance with exponential decay
        total_weight = 0.0
        weighted_mismatches = 0.0
        
        for i in range(max_len):
            # More recent positions have higher weight
            weight = (1.0 - self.decay_alpha) ** (max_len - 1 - i)
            total_weight += weight
            
            if expected[i] != actual[i]:
                weighted_mismatches += weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalize to 0.0-1.0
        return min(1.0, weighted_mismatches / total_weight)

    def compute_semantic_drift(
        self,
        recent_messages: list[str],
    ) -> float:
        """Compute semantic drift from on-policy centroid.
        
        Uses 5-turn rolling average of cosine similarity to the
        on-policy centroid.
        
        Args:
            recent_messages: Recent assistant messages
            
        Returns:
            Drift score (0.0-1.0), or 0.0 if embeddings unavailable
        """
        centroid = self._get_centroid()
        if centroid is None or not recent_messages:
            return 0.0
        
        try:
            import numpy as np
            
            # Take last 5 messages
            messages = recent_messages[-5:]
            
            # Encode messages
            embeddings = self._model.encode(messages, convert_to_numpy=True)
            
            # Average embedding
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Cosine similarity to centroid
            similarity = np.dot(avg_embedding, centroid) / (
                np.linalg.norm(avg_embedding) * np.linalg.norm(centroid)
            )
            
            # Drift is inverse of similarity
            return float(max(0.0, 1.0 - similarity))
            
        except Exception as e:
            logger.error(f"Semantic drift computation failed: {e}")
            return 0.0

    def compute_drift(
        self,
        session: SessionContext,
        latest_message: str,
        tool_calls: list[str] | None = None,
        expected_tool_calls: list[str] | None = None,
    ) -> DriftScores:
        """Compute composite drift scores.
        
        Args:
            session: Current session context
            latest_message: Latest assistant message
            tool_calls: Tool calls made in this turn
            expected_tool_calls: Tool calls expected for this state
            
        Returns:
            DriftScores with temporal, semantic, and composite scores
        """
        # Get actual state sequence from session
        actual_sequence = session.get_state_sequence()
        
        # Compute temporal drift
        temporal = self.compute_temporal_drift(
            self.expected_sequence[:len(actual_sequence)],
            actual_sequence,
        )
        
        # Collect recent messages from turn window
        recent_messages = [
            turn.get("message", "")
            for turn in session.turn_window
            if turn.get("message")
        ]
        if latest_message:
            recent_messages.append(latest_message)
        
        # Compute semantic drift
        semantic = self.compute_semantic_drift(recent_messages)
        
        # Build drift scores
        drift = DriftScores.from_scores(
            temporal=temporal,
            semantic=semantic,
            temporal_weight=self.temporal_weight,
        )
        
        # Check for anomaly flags
        tool_calls = tool_calls or []
        expected_tool_calls = expected_tool_calls or []
        
        # Check for unexpected tool calls (agent called tools not in expected list)
        if tool_calls:
            unexpected = set(tool_calls) - set(expected_tool_calls)
            if unexpected:
                drift.anomaly_flags["unexpected_tool_call"] = True

        # Check for missing expected tool calls
        if expected_tool_calls:
            missing = set(expected_tool_calls) - set(tool_calls)
            if missing:
                drift.anomaly_flags["missing_expected_tool_call"] = True
        
        logger.debug(
            f"Drift computed: temporal={temporal:.3f}, semantic={semantic:.3f}, "
            f"composite={drift.composite:.3f}, level={drift.level.value}"
        )
        
        return drift
