"""Tests for state classification."""

import pytest

from openbias.policy.engines.stateful import StateClassificationResult
from openbias.policy.engines.fsm.classifier import StateClassifier
from openbias.policy.engines.fsm.workflow.schema import State, ClassificationHint
from openbias.core.utils import extract_response_content, extract_tool_call_names


class TestStateClassifier:
    """Tests for StateClassifier."""

    @pytest.fixture
    def states(self):
        """Create test states with various classification hints."""
        return [
            State(
                name="greeting",
                is_initial=True,
                classification=ClassificationHint(
                    patterns=[r"\bhello\b", r"\bhi\b", r"\bwelcome\b"],
                ),
            ),
            State(
                name="search",
                classification=ClassificationHint(
                    tool_calls=["search_kb", "lookup"],
                    patterns=[r"\bsearching\b", r"\blooking up\b"],
                ),
            ),
            State(
                name="resolution",
                is_terminal=True,
                classification=ClassificationHint(
                    patterns=["resolved", "fixed", "anything else"],
                    exemplars=[
                        "I've resolved your issue.",
                        "Is there anything else I can help with?",
                    ],
                ),
            ),
        ]

    @pytest.fixture
    def classifier(self, states):
        """Create classifier with test states."""
        return StateClassifier(states)

    def test_classify_by_tool_call(self, classifier, mock_llm_response, mock_tool_call):
        """Test classification by tool call."""
        response = mock_llm_response(
            content="Let me search for that.",
            tool_calls=[mock_tool_call("search_kb")],
        )

        result = classifier.classify(response)

        assert result.state_name == "search"
        assert result.confidence == 1.0
        assert result.method == "tool_call"

    def test_classify_by_pattern(self, classifier, mock_llm_response):
        """Test classification by regex pattern."""
        response = mock_llm_response(content="Hello! How can I help you today?")

        result = classifier.classify(response)

        assert result.state_name == "greeting"
        assert result.confidence == 0.9
        assert result.method == "pattern"

    def test_classify_pattern_case_insensitive(self, classifier, mock_llm_response):
        """Test that pattern matching is case insensitive."""
        response = mock_llm_response(content="HELLO there!")

        result = classifier.classify(response)

        assert result.state_name == "greeting"

    def test_classify_fallback(self, classifier, mock_llm_response):
        """Test fallback when no classification matches."""
        response = mock_llm_response(content="Random unrelated text.")

        result = classifier.classify(response, current_state="greeting")

        assert result.state_name == "greeting"  # Falls back to current
        assert result.confidence == 0.0
        assert result.method == "fallback"

    def test_classify_from_tool_call_directly(self, classifier):
        """Test classifying directly from tool call info."""
        result = classifier.classify_from_tool_call("search_kb", {"query": "test"})

        assert result is not None
        assert result.state_name == "search"
        assert result.confidence == 1.0

    def test_classify_unknown_tool(self, classifier):
        """Test classifying unknown tool call."""
        result = classifier.classify_from_tool_call("unknown_tool", {})

        assert result is None

    def test_tool_call_priority_over_pattern(
        self, classifier, mock_llm_response, mock_tool_call
    ):
        """Test that tool calls have priority over patterns."""
        # Content matches "greeting" pattern, but tool matches "search"
        response = mock_llm_response(
            content="Hello, let me help you.",
            tool_calls=[mock_tool_call("search_kb")],
        )

        result = classifier.classify(response)

        # Tool call should win
        assert result.state_name == "search"
        assert result.method == "tool_call"

    def test_extract_content_from_dict(self):
        """Test extracting content from various response formats."""
        # OpenAI format
        response1 = {"choices": [{"message": {"content": "Hello"}}]}
        assert extract_response_content(response1) == "Hello"

        # Simple format
        response2 = {"content": "Hello"}
        assert extract_response_content(response2) == "Hello"

        # Role-based
        response3 = {"role": "assistant", "content": "Hello"}
        assert extract_response_content(response3) == "Hello"

    def test_extract_tool_calls(self, mock_tool_call):
        """Test extracting tool calls from response."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            mock_tool_call("search_kb"),
                            mock_tool_call("lookup"),
                        ],
                    }
                }
            ]
        }

        tools = extract_tool_call_names(response)

        assert tools == ["search_kb", "lookup"]

    def test_multiple_patterns(self, classifier, mock_llm_response):
        """Test matching multiple patterns."""
        # Should match first pattern that hits
        response = mock_llm_response(content="I'm searching for your answer.")

        result = classifier.classify(response)

        assert result.state_name == "search"
        assert result.method == "pattern"


class TestEmbeddingThresholdFiltering:
    """Tests for per-state threshold filtering in embedding classification."""

    def test_lower_similarity_state_wins_when_highest_fails_its_threshold(self):
        """A state with lower similarity but lower threshold should be returned
        when the highest-similarity state doesn't meet its own threshold."""
        import numpy as np
        from unittest.mock import MagicMock, patch

        # State A: high threshold (0.95) — similarity will be 0.85 (fails)
        # State B: low threshold (0.5)  — similarity will be 0.70 (passes)
        states = [
            State(
                name="strict_state",
                classification=ClassificationHint(
                    exemplars=["exemplar for strict"],
                    min_similarity=0.95,
                ),
            ),
            State(
                name="lenient_state",
                classification=ClassificationHint(
                    exemplars=["exemplar for lenient"],
                    min_similarity=0.5,
                ),
            ),
        ]

        classifier = StateClassifier(states)

        # Mock the model and embeddings so we control similarity values
        mock_model = MagicMock()
        classifier._model = mock_model

        # Create unit vectors with known cosine similarities to a query vector
        # query = [1, 0], strict_embedding = [0.85, 0.527], lenient_embedding = [0.70, 0.714]
        query = np.array([1.0, 0.0])
        strict_emb = np.array([0.85, 0.5268])  # cos sim ~ 0.85
        strict_emb = strict_emb / np.linalg.norm(strict_emb)
        lenient_emb = np.array([0.70, 0.7141])  # cos sim ~ 0.70
        lenient_emb = lenient_emb / np.linalg.norm(lenient_emb)

        mock_model.encode.return_value = query

        with patch.object(
            classifier,
            "_get_state_embeddings",
            return_value={"strict_state": strict_emb, "lenient_state": lenient_emb},
        ):
            result = classifier._classify_by_embeddings("test content")

        assert result is not None
        assert result.state_name == "lenient_state"
        assert result.method == "embedding"
        # strict_state had higher similarity but didn't meet its 0.95 threshold
        # lenient_state with ~0.70 similarity meets its 0.5 threshold

    def test_no_candidates_when_all_fail_threshold(self):
        """Returns None when no state meets its threshold."""
        import numpy as np
        from unittest.mock import MagicMock, patch

        states = [
            State(
                name="high_bar",
                classification=ClassificationHint(
                    exemplars=["exemplar"],
                    min_similarity=0.99,
                ),
            ),
        ]

        classifier = StateClassifier(states)
        mock_model = MagicMock()
        classifier._model = mock_model

        query = np.array([1.0, 0.0])
        emb = np.array([0.5, 0.866])  # cos sim ~ 0.5, below 0.99
        emb = emb / np.linalg.norm(emb)

        mock_model.encode.return_value = query

        with patch.object(
            classifier,
            "_get_state_embeddings",
            return_value={"high_bar": emb},
        ):
            result = classifier._classify_by_embeddings("test content")

        assert result is None


class TestEmbeddingAvailability:
    """Tests for embedding availability check."""

    def test_check_embedding_unavailable(self, states):
        """Warning is surfaced when sentence-transformers is not installed."""
        from unittest.mock import patch

        classifier = StateClassifier(states)

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("No module named 'sentence_transformers'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            assert classifier.check_embedding_availability() is False

    def test_check_embedding_available(self, states):
        """Returns True when sentence-transformers is importable."""
        from unittest.mock import patch, MagicMock

        classifier = StateClassifier(states)

        fake_module = MagicMock()
        with patch.dict("sys.modules", {"sentence_transformers": fake_module}):
            assert classifier.check_embedding_availability() is True

    @pytest.fixture
    def states(self):
        return [
            State(
                name="greeting",
                is_initial=True,
                classification=ClassificationHint(patterns=[r"\bhello\b"]),
            ),
        ]


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    @pytest.fixture
    def result(self):
        return StateClassificationResult(
            state_name="test",
            confidence=0.95,
            method="pattern",
            details={"matched_pattern": "test"},
        )

    def test_classification_result_fields(self, result):
        """Test StateClassificationResult fields."""
        assert result.state_name == "test"
        assert result.confidence == 0.95
        assert result.method == "pattern"
        assert "matched_pattern" in result.details
