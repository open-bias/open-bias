"""
Tests for pairwise bias mitigation.
"""

import pytest
from openbias.policy.engines.judge.bias import (
    randomize_positions,
    demap_winner,
    demap_pairwise_scores,
)


class TestRandomizePositions:
    def test_returns_three_tuple(self):
        first, second, mapping = randomize_positions("a_content", "b_content")
        assert isinstance(first, str)
        assert isinstance(second, str)
        assert "first" in mapping
        assert "second" in mapping

    def test_mapping_is_valid(self):
        """Mapping should always contain 'a' and 'b'."""
        for _ in range(20):
            _, _, mapping = randomize_positions("a", "b")
            assert set(mapping.values()) == {"a", "b"}

    def test_content_preserved(self):
        """Both contents should appear regardless of order."""
        for _ in range(20):
            first, second, _ = randomize_positions("content_a", "content_b")
            assert {first, second} == {"content_a", "content_b"}


class TestDemapWinner:
    def test_tie_unchanged(self):
        assert demap_winner("tie", {"first": "a", "second": "b"}) == "tie"
        assert demap_winner("tie", {"first": "b", "second": "a"}) == "tie"

    def test_no_swap(self):
        mapping = {"first": "a", "second": "b"}
        assert demap_winner("a", mapping) == "a"
        assert demap_winner("b", mapping) == "b"

    def test_swapped(self):
        mapping = {"first": "b", "second": "a"}
        # "a" (first position) was actually "b"
        assert demap_winner("a", mapping) == "b"
        # "b" (second position) was actually "a"
        assert demap_winner("b", mapping) == "a"


class TestDemapPairwiseScores:
    def test_no_swap_passthrough(self):
        mapping = {"first": "a", "second": "b"}
        scores = [{"criterion": "c1", "score_a": 4, "score_b": 3, "winner": "a"}]
        result = demap_pairwise_scores(scores, mapping)
        assert result[0]["score_a"] == 4
        assert result[0]["score_b"] == 3
        assert result[0]["winner"] == "a"

    def test_swapped_scores_remapped(self):
        mapping = {"first": "b", "second": "a"}
        scores = [{"criterion": "c1", "score_a": 5, "score_b": 2, "winner": "a"}]
        result = demap_pairwise_scores(scores, mapping)
        # Scores should be swapped back
        assert result[0]["score_a"] == 2
        assert result[0]["score_b"] == 5
        # "a" (first position) was "b", so winner should be "b"
        assert result[0]["winner"] == "b"

    def test_multiple_scores(self):
        mapping = {"first": "b", "second": "a"}
        scores = [
            {"criterion": "c1", "score_a": 5, "score_b": 3, "winner": "a"},
            {"criterion": "c2", "score_a": 2, "score_b": 4, "winner": "b"},
        ]
        result = demap_pairwise_scores(scores, mapping)
        assert len(result) == 2
        assert result[0]["score_a"] == 3
        assert result[1]["winner"] == "a"
