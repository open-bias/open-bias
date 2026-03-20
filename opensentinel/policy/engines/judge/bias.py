"""
Bias mitigation for pairwise evaluation.

Implements position randomization to counter position bias
in LLM-as-a-Judge pairwise comparisons.
"""

import random


def randomize_positions(
    response_a: str,
    response_b: str,
) -> tuple[str, str, dict[str, str]]:
    """Randomly swap response positions to mitigate position bias.

    Args:
        response_a: Original Response A content.
        response_b: Original Response B content.

    Returns:
        Tuple of (first, second, mapping) where mapping records
        which original response is in which position.
        mapping["first"] = "a" or "b", mapping["second"] = "a" or "b".
    """
    if random.random() < 0.5:
        # Swap positions
        return response_b, response_a, {"first": "b", "second": "a"}
    else:
        # Keep original order
        return response_a, response_b, {"first": "a", "second": "b"}


def demap_winner(winner: str, mapping: dict[str, str]) -> str:
    """Map the positional winner back to original labels.

    Args:
        winner: "a" (first position), "b" (second position), or "tie".
        mapping: Position mapping from randomize_positions.

    Returns:
        Original label: "a", "b", or "tie".
    """
    if winner == "tie":
        return "tie"

    # winner="a" means first position won -> map back to original
    if winner == "a":
        return mapping["first"]
    elif winner == "b":
        return mapping["second"]

    return winner


def demap_pairwise_scores(
    scores: list,
    mapping: dict[str, str],
) -> list:
    """De-map all pairwise scores back to original positions.

    Swaps score_a/score_b and remaps winner fields if positions
    were randomized.

    Args:
        scores: List of score dicts with score_a, score_b, winner keys.
        mapping: Position mapping from randomize_positions.

    Returns:
        Scores with values mapped back to original a/b labels.
    """
    if mapping["first"] == "a":
        # No swap occurred, return as-is
        return scores

    # Positions were swapped: what was shown as "A" was actually "b"
    remapped = []
    for score in scores:
        remapped_score = dict(score)
        # Swap the scores back
        remapped_score["score_a"] = score.get("score_b", 0)
        remapped_score["score_b"] = score.get("score_a", 0)
        # Remap the winner
        if "winner" in score:
            remapped_score["winner"] = demap_winner(score["winner"], mapping)
        remapped.append(remapped_score)

    return remapped
