"""Tests for state classification."""

import pytest

from opensentinel.policy.protocols import StateClassificationResult
from opensentinel.policy.engines.fsm.classifier import StateClassifier
from opensentinel.policy.engines.fsm.workflow.schema import State, ClassificationHint
from opensentinel.core.utils import extract_response_content, extract_tool_call_names


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
