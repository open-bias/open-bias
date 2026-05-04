"""
LLM-based state classifier for the LLM Policy Engine.

Uses an LLM to classify agent responses to workflow states with
confidence scoring, transition legality checking, and skip violation
detection.
"""

import logging
from collections import deque
from typing import Any

from openbias.policy.engines.fsm.workflow.schema import WorkflowDefinition
from openbias.policy.engines.llm.llm_client import LLMClient, LLMClientError
from openbias.policy.engines.llm.models import (
    ConfidenceTier,
    LLMClassificationResult,
    LLMStateCandidate,
    SessionContext,
)
from openbias.policy.engines.llm.prompts import (
    STATE_CLASSIFICATION_SYSTEM,
    STATE_CLASSIFICATION_USER,
)

logger = logging.getLogger(__name__)

class LLMStateClassifier:
    """LLM-based workflow state classifier.
    
    Uses an LLM to analyze agent responses and classify them to
    workflow states. Provides confidence tiers, transition legality
    checking, and detection of skipped intermediate states.
    
    Example:
        classifier = LLMStateClassifier(llm_client, workflow)
        result = await classifier.classify(
            session,
            assistant_message="Let me search the knowledge base",
            tool_calls=["search_kb"]
        )
        if result.tier == ConfidenceTier.LOST:
            # Agent may be off-track
    """

    # Confidence thresholds
    CONFIDENT_THRESHOLD = 0.8
    UNCERTAIN_THRESHOLD = 0.5

    def __init__(
        self,
        llm_client: LLMClient,
        workflow: WorkflowDefinition,
        confident_threshold: float = 0.8,
        uncertain_threshold: float = 0.5,
    ):
        self.llm_client = llm_client
        self.workflow = workflow
        self.confident_threshold = confident_threshold
        self.uncertain_threshold = uncertain_threshold
        
        # Build transition adjacency map
        self._transition_map: dict[str, set[str]] = {}
        for transition in workflow.transitions:
            if transition.from_state not in self._transition_map:
                self._transition_map[transition.from_state] = set()
            self._transition_map[transition.from_state].add(transition.to_state)
        
        # State lookup
        self._states = {s.name: s for s in workflow.states}

    async def classify(
        self,
        session: SessionContext,
        assistant_message: str,
        tool_calls: list[str] | None = None,
    ) -> LLMClassificationResult:
        """Classify an assistant response to a workflow state.
        
        Args:
            session: Current session context
            assistant_message: The assistant's response text
            tool_calls: List of tool names called
            
        Returns:
            LLMClassificationResult with candidates, tier, and legality
        """
        tool_calls = tool_calls or []
        current_state = session.current_state
        
        try:
            # Build prompts
            system_prompt = STATE_CLASSIFICATION_SYSTEM.format(
                workflow_name=self.workflow.name,
                states_block=self._build_states_block(),
                current_state=current_state,
            )
            
            user_prompt = STATE_CLASSIFICATION_USER.format(
                assistant_message=assistant_message or "(no text content)",
                tool_calls_block=self._format_tool_calls(tool_calls),
                window_size=len(session.turn_window),
                conversation_block=self._format_conversation(session.turn_window),
            )
            
            # Call LLM
            response = await self.llm_client.complete_json(
                system_prompt, 
                user_prompt,
                session_id=session.session_id
            )
            
            # Parse candidates
            candidates = self._parse_candidates(response)
            
            if not candidates:
                # Fallback to current state if no candidates
                return self._fallback_result(current_state)
            
            # Get best candidate
            best = candidates[0]
            tier = self._determine_tier(best.confidence)
            
            # Check transition legality
            transition_legal = self._check_transition_legality(
                current_state, best.state_id
            )
            
            # Detect skip violations
            skip_violations = []
            if not transition_legal:
                skip_violations = self._detect_skip_violations(
                    current_state, best.state_id
                )
            
            return LLMClassificationResult(
                candidates=candidates,
                best_state=best.state_id,
                best_confidence=best.confidence,
                tier=tier,
                transition_legal=transition_legal,
                skip_violations=skip_violations,
                raw_llm_response=str(response),
            )
            
        except LLMClientError as e:
            logger.error(f"LLM classification failed: {e}")
            return self._fallback_result(current_state)
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self._fallback_result(current_state)

    def _build_states_block(self) -> str:
        """Build formatted state definitions for the prompt."""
        lines = []
        for state in self.workflow.states:
            lines.append(f"State: {state.name}")
            if state.description:
                lines.append(f"  Description: {state.description}")
            if state.classification.tool_calls:
                lines.append(
                    f"  Expected tool calls: {', '.join(state.classification.tool_calls)}"
                )
            if state.classification.exemplars:
                lines.append(f"  Example phrases:")
                for ex in state.classification.exemplars[:3]:
                    lines.append(f"    - \"{ex}\"")
            lines.append("")
        return "\n".join(lines)

    def _format_tool_calls(self, tool_calls: list[str]) -> str:
        """Format tool calls for the prompt."""
        if not tool_calls:
            return "(none)"
        return ", ".join(tool_calls)

    def _format_conversation(self, turn_window: list[dict[str, Any]]) -> str:
        """Format recent conversation for the prompt."""
        if not turn_window:
            return "(no previous turns)"
        
        lines = []
        for turn in turn_window[-5:]:  # Last 5 turns
            role = turn.get("role", "unknown")
            message = turn.get("message", "")[:200]  # Truncate
            lines.append(f"{role}: {message}")
        return "\n".join(lines)

    def _parse_candidates(self, response: Any) -> list[LLMStateCandidate]:
        """Parse LLM response into state candidates."""
        candidates = []
        
        # Handle both list and dict responses
        if isinstance(response, dict):
            response = response.get("candidates", response.get("states", [response]))
        
        if not isinstance(response, list):
            response = [response]
        
        for item in response:
            if isinstance(item, dict):
                state_id = item.get("state_id", item.get("state", ""))
                confidence = float(item.get("confidence", 0.0))
                reasoning = item.get("reasoning", "")
                
                # Validate state exists
                if state_id in self._states:
                    candidates.append(LLMStateCandidate(
                        state_id=state_id,
                        confidence=confidence,
                        reasoning=reasoning,
                    ))
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def _determine_tier(self, confidence: float) -> ConfidenceTier:
        """Determine confidence tier from score."""
        if confidence >= self.confident_threshold:
            return ConfidenceTier.CONFIDENT
        elif confidence >= self.uncertain_threshold:
            return ConfidenceTier.UNCERTAIN
        else:
            return ConfidenceTier.LOST

    def _check_transition_legality(
        self,
        from_state: str,
        to_state: str,
    ) -> bool:
        """Check if transition from current to target state is legal.
        
        If no transitions are defined FROM a state, any transition is legal
        (matching FSM behavior).
        """
        # If no transitions defined from this state, any target is legal
        if from_state not in self._transition_map:
            return True
        
        # Self-transitions are always legal
        if from_state == to_state:
            return True
        
        # Check if to_state is in allowed transitions
        return to_state in self._transition_map.get(from_state, set())

    def _detect_skip_violations(
        self,
        from_state: str,
        to_state: str,
    ) -> list[str]:
        """Detect if intermediate states were skipped.
        
        Uses BFS to find if to_state is reachable from from_state
        and returns the intermediate states that were skipped.
        """
        if from_state not in self._transition_map:
            return []
        
        # BFS to find path
        visited: set[str] = {from_state}
        queue: deque = deque([(from_state, [])])  # (state, path)
        
        while queue:
            current, path = queue.popleft()
            
            for next_state in self._transition_map.get(current, set()):
                if next_state == to_state:
                    # Found path - return intermediate states
                    return path  # These are the skipped states
                
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [next_state]))
        
        # to_state not reachable
        return []

    def _fallback_result(self, current_state: str) -> LLMClassificationResult:
        """Create fallback result when classification fails."""
        return LLMClassificationResult(
            candidates=[LLMStateCandidate(
                state_id=current_state,
                confidence=0.0,
                reasoning="Classification failed, staying in current state",
            )],
            best_state=current_state,
            best_confidence=0.0,
            tier=ConfidenceTier.LOST,
            transition_legal=True,
            skip_violations=[],
            raw_llm_response=None,
        )
