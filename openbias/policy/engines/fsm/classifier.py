"""
Workflow state classifier.

Classification strategies (in priority order):
1. Tool calls: Exact match on function/tool names (instant)
2. Patterns: Regex matching on response content (~1ms)
3. Embeddings: Semantic similarity using sentence-transformers (~50ms)

Performance target: <50ms total for classification.
Model: all-MiniLM-L6-v2 (22MB, ~14k sentences/sec on CPU)
"""

import logging
import re
from typing import Any
from dataclasses import dataclass

from openbias.policy.engines.stateful import StateClassificationResult
from openbias.policy.engines.fsm.workflow.schema import State, ClassificationHint
from openbias.config.settings import ClassifierConfig
from openbias.core.utils import extract_response_content, extract_tool_call_names

logger = logging.getLogger(__name__)

class StateClassifier:
    """
    Classifies LLM responses to workflow states.

    Uses a cascade of classification methods for accuracy and speed:
    1. Tool calls (exact match, instant) - highest confidence
    2. Patterns (regex, ~1ms) - high confidence
    3. Embeddings (semantic similarity, ~50ms) - semantic fallback

    Example:
        ```python
        from openbias.policy.engines.fsm import StateClassifier

        classifier = StateClassifier(workflow.states)

        result = classifier.classify(llm_response)
        print(f"State: {result.state_name}, Confidence: {result.confidence}")
        ```
    """

    def __init__(
        self,
        states: list[State],
        config: ClassifierConfig | None = None,
    ):
        self.states = {s.name: s for s in states}
        self.config = config or ClassifierConfig()

        # Lazy-load embedding model
        self._model = None
        self._state_embeddings: dict[str, Any] | None = None

        # Pre-compile regex patterns for performance
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        for state in states:
            if state.classification.patterns:
                self._compiled_patterns[state.name] = [
                    re.compile(p, re.IGNORECASE) for p in state.classification.patterns
                ]

        logger.debug(f"StateClassifier initialized with {len(states)} states")

    def check_embedding_availability(self) -> bool:
        """Check whether the embedding model can be loaded.

        Returns:
            True if embeddings are available, False otherwise.
        """
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def model(self):
        """Lazy-load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.config.model_name}")
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                )
                logger.info("Embedding model loaded successfully")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed, embedding classification disabled"
                )
                return None
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                return None

        return self._model

    def _get_state_embeddings(self) -> dict[str, Any]:
        """Compute and cache state exemplar embeddings."""
        if self._state_embeddings is not None:
            return self._state_embeddings

        if not self.model:
            return {}

        import numpy as np

        self._state_embeddings = {}

        for name, state in self.states.items():
            if state.classification.exemplars:
                try:
                    # Compute embeddings for all exemplars
                    embeddings = self.model.encode(
                        state.classification.exemplars,
                        convert_to_numpy=True,
                    )
                    # Average embedding of all exemplars
                    self._state_embeddings[name] = np.mean(embeddings, axis=0)
                except Exception as e:
                    logger.error(f"Failed to compute embeddings for state {name}: {e}")

        logger.debug(
            f"Computed embeddings for {len(self._state_embeddings)} states"
        )
        return self._state_embeddings

    def classify(
        self,
        response: Any,
        current_state: str | None = None,
    ) -> StateClassificationResult:
        """
        Classify an LLM response to a workflow state.

        Args:
            response: The LLM response (dict with content, tool_calls, etc.)
            current_state: Current state (for transition-aware classification)

        Returns:
            StateClassificationResult with state name, confidence, and method used.
        """
        # Extract content and tool calls from response
        content = extract_response_content(response)
        tool_calls = extract_tool_call_names(response)

        logger.debug(
            f"Classifying response: {len(content)} chars, "
            f"{len(tool_calls)} tool calls"
        )

        # Strategy 1: Tool call matching (exact, instant)
        if tool_calls:
            result = self._classify_by_tools(tool_calls)
            if result:
                logger.debug(f"Classified by tool call: {result.state_name}")
                return result

        # Strategy 2: Pattern matching (regex, fast)
        if content:
            result = self._classify_by_patterns(content)
            if result:
                logger.debug(f"Classified by pattern: {result.state_name}")
                return result

        # Strategy 3: Embedding similarity (semantic, ~50ms)
        if content and self.model:
            result = self._classify_by_embeddings(content)
            if result:
                logger.debug(
                    f"Classified by embedding: {result.state_name} "
                    f"(similarity={result.confidence:.2f})"
                )
                return result

        # Fallback: Stay in current state, or use workflow's initial state
        if current_state:
            fallback_state = current_state
        else:
            initial = [name for name, s in self.states.items() if s.is_initial]
            fallback_state = initial[0] if initial else next(iter(self.states))
        logger.debug(f"Fallback classification: {fallback_state}")

        return StateClassificationResult(
            state_name=fallback_state,
            confidence=0.0,
            method="fallback",
            details={"reason": "No classification method matched"},
        )

    def _classify_by_tools(
        self,
        tool_calls: list[str],
    ) -> StateClassificationResult | None:
        """Classify by matching tool call names."""
        for state_name, state in self.states.items():
            hint = state.classification
            if hint.tool_calls:
                matches = set(tool_calls) & set(hint.tool_calls)
                if matches:
                    return StateClassificationResult(
                        state_name=state_name,
                        confidence=1.0,
                        method="tool_call",
                        details={"matched_tools": list(matches)},
                    )
        return None

    def _classify_by_patterns(
        self,
        content: str,
    ) -> StateClassificationResult | None:
        """Classify by regex pattern matching."""
        for state_name, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(content)
                if match:
                    return StateClassificationResult(
                        state_name=state_name,
                        confidence=0.9,
                        method="pattern",
                        details={
                            "matched_pattern": pattern.pattern,
                            "match": match.group(),
                        },
                    )
        return None

    def _classify_by_embeddings(
        self,
        content: str,
    ) -> StateClassificationResult | None:
        """Classify by semantic similarity to state exemplars."""
        state_embeddings = self._get_state_embeddings()
        if not state_embeddings:
            return None

        import numpy as np

        try:
            # Compute content embedding
            content_embedding = self.model.encode(content, convert_to_numpy=True)

            # Compute similarity for each state and filter by per-state threshold
            candidates: list[tuple[str, float]] = []

            for state_name, state_embedding in state_embeddings.items():
                # Cosine similarity
                similarity = float(
                    np.dot(content_embedding, state_embedding)
                    / (np.linalg.norm(content_embedding) * np.linalg.norm(state_embedding))
                )
                threshold = self.states[state_name].classification.min_similarity
                if similarity >= threshold:
                    candidates.append((state_name, similarity))

            # Pick highest similarity among candidates that passed their threshold
            if candidates:
                best_state, best_similarity = max(candidates, key=lambda c: c[1])
                return StateClassificationResult(
                    state_name=best_state,
                    confidence=best_similarity,
                    method="embedding",
                    details={"similarity": best_similarity},
                )

        except Exception as e:
            logger.error(f"Embedding classification failed: {e}")

        return None

    def classify_from_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
    ) -> StateClassificationResult | None:
        """
        Quick classification based on tool usage.

        Useful when you only have the tool call info, not full response.
        """
        for state_name, state in self.states.items():
            hint = state.classification
            if hint.tool_calls and tool_name in hint.tool_calls:
                return StateClassificationResult(
                    state_name=state_name,
                    confidence=1.0,
                    method="tool_call",
                    details={
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                    },
                )
        return None
